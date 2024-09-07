import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

device = "cpu"

class LSTM(nn.Module):
    
    def __init__(self, inpt_size, hidden_size, num_cells, output_size, dev):
        super(LSTM, self).__init__()
        self.inpt_size = inpt_size
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.output_size = output_size
        
        device = dev
        

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(Cell(self.inpt_size, self.hidden_size))
        for l in range(1, self.num_cells):
            self.rnn_cell_list.append(Cell(self.hidden_size,
                                                self.hidden_size))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inpt, hx=None):
        if hx is None:
            h0 = Variable(torch.zeros(self.num_cells, inpt.size(0), self.hidden_size).to(device))
        else:
            h0 = hx

        outs = []

        hidden = []
        for layer in range(self.num_cells):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(inpt.size(1)):

            for layer in range(self.num_cells):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        inpt[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        out = self.fc(out)

        return out

class Cell(nn.Module):
    def __init__(self, inpt_size, hidden_size, bias=True):
        super(Cell, self).__init__()
        self.inpt_size = inpt_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(inpt_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inpt, hx=None):
        if hx is None:
            hx = Variable(inpt.new_zeros(inpt.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self.xh(inpt) + self.hh(hx)

        # Get gates (inpt_t, forget_t, g_t, output_t)
        inpt_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        inpt_t = torch.sigmoid(inpt_gate)
        forget_t = torch.sigmoid(forget_gate)
        cell_t = torch.tanh(cell_gate)
        output_t = torch.sigmoid(output_gate)

        cy = cx * forget_t + inpt_t * cell_t

        hy = output_t * torch.tanh(cy)


        return (hy, cy)