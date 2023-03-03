import torch
# from fieldnn.nn.op import MergerLayer
from ..nn.op import MergerLayer

class Merger_Layer(torch.nn.Module):
    def __init__(self, input_fullname, output_fullname, merger_layer_para):
        super(Merger_Layer, self).__init__()
        
        # n * (bs, .., c_inp) --> (bs, ..., n, c_inp)
        # Meta Info
        self.input_fullname = input_fullname
        self.output_fullname = output_fullname
        
        # Part 0: sizes
        self.input_size = merger_layer_para['input_size']
        self.output_size = merger_layer_para['output_size']
        
        # Part 1: NN
        nn_name, nn_para = merger_layer_para[input_fullname]
        if nn_name.lower() == 'merger':
            self.Merger = MergerLayer()
        else:
            raise ValueError(f'The NN "{nn_name}" is not available')
        
        # Part 2: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in merger_layer_para['postprocess'].items():
            if method == 'activator':
                activator = config
                if activator.lower() == 'relu': 
                    self.postprocess[method] = torch.nn.ReLU()
                elif activator.lower() == 'tanh': 
                    self.postprocess[method] = torch.nn.Tanh()
                elif activator.lower() == 'gelu':
                    self.postprocess[method] = torch.nn.GELU()
            elif method == 'dropout':
                self.postprocess[method] = torch.nn.Dropout(**config)
            elif method == 'layernorm':
                self.postprocess[method] = torch.nn.LayerNorm(self.output_size, **config)
                
    def forward(self, input_fullname, fullname2data):
        
        fld_list = input_fullname.split('^')
        assert len(fld_list) == len(fullname2data)
        # (1) holder
        holder = self.Merger([data['holder'] for fld, data in fullname2data.items()], -1)
        
        # (2) merge data
        info = self.Merger([data['info'] for fld, data in fullname2data.items()], -2)
        
        # (3) post-process
        for name, layer in self.postprocess.items():
            info = layer(info)
        
        # (4) names
        # prefix = ['-'.join(i.split('-')[:-1]) for i in fullname2data][0]
        # assert prefix == self.fieldname
        # fullname_new = prefix + '2GrnRec2FldType'
        
        return self.output_fullname, holder, info