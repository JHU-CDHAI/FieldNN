
import torch

class ConcatenateLayer(torch.nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, info, leng_mask):
        l, dim = info.shape[-2:]
        info = info.view(*info.shape[:-2],  l*dim)
        return info   
