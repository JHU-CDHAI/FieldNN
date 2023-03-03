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
        info = info/leng    # (bs, xxx, dim)    --> (bs, xxx, dim)
        return info

class RecuderMaxLayer(torch.nn.Module):
    def __init__(self):
        super(RecuderMaxLayer, self).__init__()

    def forward(self, info, leng_mask):
        a, b = info.max(-2)
        # l, dim = info.shape[-2:]
        # info = torch.transpose(info, -1, 1).contiguous()
        # info = F.max_pool1d(info, l).squeeze()
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


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))