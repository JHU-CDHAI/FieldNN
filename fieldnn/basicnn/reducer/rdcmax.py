import torch

class RecuderMaxLayer(torch.nn.Module):
    def __init__(self):
        super(RecuderMaxLayer, self).__init__()

    def forward(self, info, holder):
        leng_mask = holder == 0
        info = info.masked_fill(leng_mask.unsqueeze(-1), -10000) # double check this.
        a, b = info.max(-2) # not necessary, all the values could be smaller than 0.
        return a
    