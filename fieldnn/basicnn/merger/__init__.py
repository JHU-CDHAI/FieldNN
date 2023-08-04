import torch
from .mergeconcat import MergeConcatLayer
from .mergemax import MergeMaxLayer
import torch
# from .mergeconcat import MergeConcatLayer
# from .mergemax import MergeMaxLayer


class Merger_Layer(torch.nn.Module):
    def __init__(self, input_names_nnlvl, output_name_nnlvl, merger_layer_para):
        super(Merger_Layer, self).__init__()
        
        # the input_names_nnlvl
        self.input_names_nnlvl = input_names_nnlvl
        # output_name should be generated from the input_names
        self.output_name_nnlvl = output_name_nnlvl
        
        self.input_size = merger_layer_para['input_size']
        self.output_size = merger_layer_para['output_size']
        
        # Part 1: NN
        nn_name = merger_layer_para['nn_name']
        nn_para = merger_layer_para['nn_para']
        
        if nn_name.lower() == 'mergeconcat':
            self.merger = MergeConcatLayer()
        elif nn_name.lower() == 'mergemax':
            self.merger = MergeMaxLayer()
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
                
    def forward(self, input_names_nnlvl, INPUTS_TO_INFODICT):
        
        INPUTS = {k:v for k, v in INPUTS_TO_INFODICT.items() if k in input_names_nnlvl}

        # (1) holder
        # holder = self.merger([data['holder'] for fld, data in INPUTS.items()], -1)
        info_holder_list = [(data['info'], data['holder']) for fld, data in INPUTS.items()]
        
        # (2) merge data
        info, holder = self.merger(info_holder_list)
        
        # (3) post-process
        for name, layer in self.postprocess.items():
            info = layer(info)

        return self.output_name_nnlvl, {'holder': holder, 'info': info}