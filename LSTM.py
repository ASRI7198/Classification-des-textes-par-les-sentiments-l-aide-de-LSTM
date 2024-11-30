import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # Couche du réseau
        self.embelling_layer = nn.Linear(input_size, emb_size)
        self.forget_layer = nn.Linear(emb_size, hidden_size)
        self.input_layer = nn.Linear(emb_size, hidden_size)
        self.cell_layer = nn.Linear(emb_size, hidden_size)
        self.output_layer = nn.Linear(emb_size, hidden_size)

        # Couche cachée
        self.hf = nn.Linear(hidden_size, hidden_size)
        self.hi = nn.Linear(hidden_size, hidden_size)
        self.hc = nn.Linear(hidden_size, hidden_size)
        self.ho = nn.Linear(hidden_size, hidden_size)

        # Couche de sortie
        self.h2end = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_cell):
        hidden, cell = hidden_cell
        embedded = self.embelling_layer(input.float())

        forget = F.sigmoid(self.forget_layer(embedded) + self.hf(hidden))
        input = F.sigmoid(self.input_layer(embedded) + self.hi(hidden))
        cell_state = F.tanh(self.cell_layer(embedded) + self.hc(hidden))
        cell = forget * cell + input * cell_state
        output = F.sigmoid(self.output_layer(embedded) + self.ho(hidden))
        hidden = output * F.tanh(cell)
        output = self.h2end(hidden)
        output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
