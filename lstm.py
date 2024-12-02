import torch
from torch.autograd import Variable
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
       
        self.fully_connected_1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fully_connected_2 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fully_connected_3 = nn.Linear(int(hidden_size/2), num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.squeeze()
        x = x.float()
        torch.manual_seed(42)
        h0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size).float()).cuda() 
        c0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        y, _ = self.lstm(x, (h0,c0)) 
        y = self.relu(self.fully_connected_1(y[:, -1, :]))
        y = self.relu(self.fully_connected_2(y))
        y = self.fully_connected_3(y) 
        return y