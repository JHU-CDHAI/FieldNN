import torch

class ReduceSumLayer(torch.nn.Module):
    def __init__(self):
        super(ReduceSumLayer, self).__init__()   

    def forward(self, info, leng_mask):
        # (bs, xxx, l, dim) --> (bs, xxx, dim)
        info = torch.sum(info, -2)
        return info