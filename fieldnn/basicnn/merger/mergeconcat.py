import torch

class MergeConcatLayer(torch.nn.Module):
    def __init__(self):
        super(MergeConcatLayer, self).__init__()
    
    def forward(self, info_holder_list):
        info_list   = [info   for info, holder in info_holder_list]
        holder_list = [holder for info, holder in info_holder_list]
        info   = torch.cat(info_list,   -2)
        holder = torch.cat(holder_list, -1)
        return info, holder