import torch

class MergeMaxLayer(torch.nn.Module):
    def __init__(self):
        super(MergeMaxLayer, self).__init__()

    def forward(self, info_holder_list):
        info_list = []
        holder_list = []
        for info, holder in info_holder_list:
            
            leng_mask = holder == 0
            info = info.masked_fill(leng_mask.unsqueeze(-1), -10000) # double check this.
            a, b = info.max(-2) # not necessary, all the values could be smaller than 0.
            info_list.append(a.unsqueeze(-2))
            
            a, b = leng_mask.max(-1) # not necessary, all the values could be smaller than 0.
            mask_list.append(a.unsqueeze(-1))
        
        info = torch.cat(info_list, -2)
        mask = torch.cat(mask_list, -2)
        return info, mask
    