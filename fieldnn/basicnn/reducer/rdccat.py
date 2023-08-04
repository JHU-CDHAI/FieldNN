import torch

class ReduceConcatLayer(torch.nn.Module):
    def __init__(self):
        super(ReduceConcatLayer, self).__init__()

    def forward(self, info, holder):
        leng_mask = holder == 0
        l, dim = info.shape[-2:]
        info = info.view(*info.shape[:-2],  l*dim)
        return info   
