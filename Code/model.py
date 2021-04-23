import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import Embeddings

class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=3, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob

class GateNetwork(nn.Module):
    def __init__(self, seq_length, output_size, hidden_size, config):
        super(GateNetwork, self).__init__()
        self.embedding = Embeddings(config)
        self.fc = nn.Sequential(nn.Linear(768,hidden_size),
                                nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(seq_length*hidden_size,output_size),
                                nn.LogSoftmax(dim=0))

    def forward(self, input):
        # (bs, seq_length, dim)
        embedding = self.embedding(input)
        output = self.fc(embedding)
        return output
