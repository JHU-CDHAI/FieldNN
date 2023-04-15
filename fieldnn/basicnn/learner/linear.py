import torch
import math
import torch.nn.functional as F

class LinearLayer(torch.nn.Module):

    def __init__(self, 
                 input_size  = 200, 
                 output_size = 200.
                 ):

        super(LinearLayer, self).__init__()
    
        self.input_size  = input_size
        self.output_size = output_size
        self.linear  = torch.nn.Linear(self.input_size, self.output_size)
        self.init_weights()
            
    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, info):
        info = self.linear(info)
        return info