import torch

class LinearLayer(torch.nn.Module):

    def __init__(self, 
                 input_size  = 200, 
                 output_size = 200,
                 initrange = 0.1
                 ):

        super(LinearLayer, self).__init__()
    
        self.input_size  = input_size
        self.output_size = output_size
        self.linear  = torch.nn.Linear(self.input_size, self.output_size)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    
    def forward(self, holder, info):
        # assert self.input_fullname == fullname
        # (1) learn the info
        info = self.linear(info)
        
        # (2) do the masked_leng because of non-zero bias
        leng_mask = holder == 0
        info = info.masked_fill(leng_mask.unsqueeze(-1), 0)
    
        # (3) post-process
        # for nn_name, layer in self.postprocess.items():
        #     info = layer(info)
            
        # we do not change the fullname and holder
        return holder, info