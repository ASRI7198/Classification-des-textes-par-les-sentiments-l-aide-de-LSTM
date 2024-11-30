import torch.nn as nn
import torch.nn.functional as F
import torch


class RNN(nn.Module):
    def __init__(self, input_size,emb_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2e = nn.Linear(input_size, emb_size)
        self.i2h = nn.Linear(emb_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input, hidden):
        embedded = self.i2e(input.float())
        hidden = F.tanh(self.i2h(embedded) + self.h2h(hidden))
        hidden = self.dropout(hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
