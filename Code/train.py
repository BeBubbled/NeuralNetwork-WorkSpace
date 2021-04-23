import os
from collections import OrderedDict
import torch
from tqdm import tqdm
from transformers import AdamW

from tensorboardX import SummaryWriter
import copy

import util

#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir)
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model, name):
        model.save_pretrained(os.path.join(self.path,name))

    # batch should be 1
    def evaluate(self, Experts, gate, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device
        for expert in Experts:
            expert.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # Forward
                outputs = gate(input_ids)
                selected_expert = torch.argmax(outputs)
                batch_size = len(input_ids)
                # total_loss, start_logits, end_logits, distilbert_output.hidden_states
                _, start_logits, end_logits,_ = Experts[selected_expert.item()](input_ids, attention_mask=attention_mask)

                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, Experts, gate, train_dataloader, eval_dataloader, val_dict, num_experts):
        device = self.device
        # get experts
        optims_E = []
        for Expert in Experts:
            Expert.train()
            optims_E.append(AdamW(Expert.parameters(), lr=self.lr))
        # get gatenetwork
        gate.to(device)
        optim_G = torch.optim.Adam(gate.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        # pretrain
        for expert_id, batch in enumerate(train_dataloader):
            if expert_id == num_experts:
                break
            optims_E[expert_id].zero_grad()
            Experts[expert_id].train()
            outputs = Experts[expert_id](batch['input_ids'].to(device),
                                         attention_mask=batch['attention_mask'].to(device),
                                         start_positions=batch['start_positions'].to(device),
                                         end_positions=batch['end_positions'].to(device))
            loss_E = torch.mean(outputs[0])
            loss_E.backward()
            optims_E[expert_id].step()
        # train
        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with tqdm(total=len(train_dataloader.dataset)) as progress_bar:#torch.enable_grad()
                for batch in train_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    # train expert
                    loss_E = []
                    for expert_id in range(num_experts):
                        optims_E[expert_id].zero_grad()
                        Experts[expert_id].train()
                        outputs = Experts[expert_id](input_ids=input_ids.detach(),
                                                     attention_mask=batch['attention_mask'].to(device),
                                                     start_positions=batch['start_positions'].to(device),
                                                     end_positions=batch['end_positions'].to(device))
                        loss_E.append(outputs[0])
                    loss_E = torch.stack(loss_E,dim=0)
                    target_experts = torch.argmin(loss_E,dim=0).detach()
                    for expert_id in range(num_experts):
                        expert_loss = [loss_E[expert_id, i] for i in range(loss_E.shape[1]) if target_experts[i].item() == expert_id]
                        if expert_loss:
                            expert_loss = torch.mean(torch.stack(expert_loss))
                            expert_loss.backward(retain_graph=True)
                            optims_E[expert_id].step()
                    # train gate
                    optim_G.zero_grad()
                    gate.train()
                    outputs = gate(input_ids.detach())
                    loss_G = torch.nn.NLLLoss()(outputs, target_experts)
                    loss_G.backward()
                    optim_G.step()
                    # show logs
                    mean_loss = torch.mean(loss_E).item()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=mean_loss)
                    tbx.add_scalar('train/NLL', mean_loss, global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(Experts, gate, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            for i in range(num_experts):
                                self.save(Experts[i], f'Expert_{i}.ckpt')
                            self.save(gate, 'Gate.ckpt')
                    global_idx += 1
        return best_scores




