import torch

class ReduceMeanLayer(torch.nn.Module):
    def __init__(self):
        super(ReduceMeanLayer, self).__init__()
  
    def forward(self, info, holder):
        leng_mask = holder == 0
        leng = (leng_mask == 0).sum(-1).unsqueeze(-1).float()
        leng[leng == 0.] = 1.0 # change pad to any non-zeros to be dominators.
        info = torch.sum(info, -2) # (bs, xxx, l, dim) --> (bs, xxx, dim)
        info = info/leng           # (bs, xxx, dim)    --> (bs, xxx, dim)
        return info
