import torch
import math
import torch.nn.functional as F

class ReduceSumLayer(torch.nn.Module):
    def __init__(self):
        super(ReduceSumLayer, self).__init__()   

    def forward(self, info, leng_mask):
        # (bs, xxx, l, dim) --> (bs, xxx, dim)
        info = torch.sum(info, -2)
        return info

class ReduceMeanLayer(torch.nn.Module):
    def __init__(self):
        super(ReduceMeanLayer, self).__init__()
  
    def forward(self, info, leng_mask):
        leng = (leng_mask == 0).sum(-1).unsqueeze(-1).float()
        leng[leng == 0.] = 1.0 # change pad to any non-zeros to be dominators.
        info = torch.sum(info, -2) # (bs, xxx, l, dim) --> (bs, xxx, dim)
        info = info/leng           # (bs, xxx, dim)    --> (bs, xxx, dim)
        return info

class RecuderMaxLayer(torch.nn.Module):
    def __init__(self):
        super(RecuderMaxLayer, self).__init__()

    def forward(self, info, leng_mask):
        info = info.masked_fill(leng_mask.unsqueeze(-1), -10000) # double check this.
        a, b = info.max(-2) # not necessary, all the values could be smaller than 0.
        return a
    
class ConcatenateLayer(torch.nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, info, leng_mask):
        l, dim = info.shape[-2:]
        info = info.view(*info.shape[:-2],  l*dim)
        return info   

class MergerLayer(torch.nn.Module):
    def __init__(self):
        super(MergerLayer, self).__init__()
    
    def forward(self, tensor_list, order = -2):
        info = torch.cat([i.unsqueeze(order) for i in tensor_list], order)
        return info   
