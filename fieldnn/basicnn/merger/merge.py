import torch

class MergeLayer(torch.nn.Module):
    def __init__(self):
        super(MergeLayer, self).__init__()
    
    def forward(self, tensor_list, order = -2):
        info = torch.cat([i.unsqueeze(order) for i in tensor_list], order)
        return info   