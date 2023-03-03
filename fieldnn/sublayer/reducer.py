import torch

from ..nn.op import ReduceMeanLayer, ReduceSumLayer, RecuderMaxLayer, ConcatenateLayer
# from fieldnn.nn.op import ReduceMeanLayer, ReduceSumLayer, RecuderMaxLayer, ConcatenateLayer

class Reducer_Layer(torch.nn.Module):
    def __init__(self, input_fullname, output_fullname, reducer_layer_para):
        super(Reducer_Layer, self).__init__()
        # (bs, xxx, l, c_inp) --> (bs, xxx, c_outp)
        
        # Part 0: Meta
        self.input_fullname = input_fullname
        self.output_fullname = output_fullname
        self.input_size = reducer_layer_para['input_size']
        self.output_size = reducer_layer_para['output_size']

        # Part 1: NN
        nn_name, nn_para = reducer_layer_para[input_fullname]
        if nn_name.lower() == 'mean':
            self.reducer = ReduceMeanLayer()
        elif nn_name.lower() == 'sum':
            self.reducer = ReduceSumLayer()
        elif nn_name.lower() == 'max':
            self.reducer = RecuderMaxLayer()
        elif nn_name.lower() == 'concat':
            self.reducer = ConcatenateLayer()
            # TODO: need to assert something
            assert self.output_size % self.input_size == 0
        else:
            raise ValueError(f'There is no layer for "{nn_name}"')
            
        # Part 3: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in reducer_layer_para['postprocess'].items():
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
            
    def forward(self, fullname, holder, info):
        assert self.input_fullname == fullname
        leng_mask = holder == 0
        info = self.reducer(info, leng_mask)
        
        # (3) post-process
        for name, layer in self.postprocess.items():
            info = layer(info)
            
        # fullname = self.fullname
        holder = (leng_mask == 0).sum(-1)
        return self.output_fullname, holder, info