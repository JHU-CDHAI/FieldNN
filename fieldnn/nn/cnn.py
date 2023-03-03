import torch
import torch.nn.functional as F
# from .helper import reverse_tensor, _addindent, gelu, get_leng_mask


class CNNLayer(torch.nn.Module):
    
    def __init__(self, 
                 type = 'conv1d',
                 n_layers = 1,
                 input_type = 'INPUT-NML',  # ['INPUT-NML']
                 direction_type = 'MIX',    # ['MIX']
                 struct_type = 'LEARNER', # ['LEARNER', 'REDUCER']
                 input_size  = 200, 
                 output_size = 200, 
                 kernel_size = 4, 
                 stride=1, 
                 padding=1, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros'):

        super(CNNLayer, self).__init__()
        
        assert input_type in ['INPUT-NML']
        assert direction_type in ['MIX'] # for mu
        assert struct_type in ['LEARNER']
        
        # cnn_type can be conv1d, and conv2d
        self.type    = type
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        
        self.input_type = input_type
        self.direction_type = direction_type
        self.struct_type = struct_type
        # struct_type can be extractor or reducer
        assert type.lower() in ['conv1d', 'conv2d']
        if type.lower() == 'conv1d':
            self.cnn = torch.nn.Conv1d(input_size, output_size, kernel_size,
                                       stride = stride, padding=padding, dilation=dilation, 
                                       groups=groups, bias=bias, padding_mode=padding_mode)
            self.output = self.extractor # if self.struct_type == 'EXTRACTOR' else self.reducer

        elif type.lower() == 'conv2d':
            assert struct_type == 'EXTRACTOR'
            self.cnn = torch.nn.Conv2d(input_size, output_size, kernel_size, 
                                       stride = stride, padding=padding, dilation=dilation, 
                                       groups=groups, bias=bias, padding_mode=padding_mode)
            self.output = self.extractor
            

        # (+) postprocess here
        self.postprocess = []
        for method, use_config in postprecess.items():
            use, config = use_config
            if use == False: continue
            if method == 'activator':
                activator = config
                if activator.lower() == 'relu': 
                    self.activator = F.relu
                elif activator.lower() == 'tanh': 
                    self.activator = F.tanh
                elif activator.lower() == 'gelu':
                    # TODO: adding gelu here.
                    self.activator =  gelu
                else:
                    self.activator = lambda x: x
                self.postprocess.append(self.activator)
            
            if method == 'dropout':
                self.drop = torch.nn.Dropout(**config)
                self.postprocess.append(self.drop)
                
            elif method == 'layernorm':
                # https://pytorch.org/docs/stable/nn.html
                self.layernorm = torch.nn.LayerNorm(self.output_size, **config)
                self.postprocess.append(self.layernorm)
            
 
    # def reducer(self, info, leng_st_mask, batch_size):
    #     # (BS, S, EmbedSize) --> (BS, EmbedSize, S)
    #     info = torch.transpose(info, -1, 1).contiguous()
    #     return F.max_pool1d(info, info.size(2)).view(batch_size, -1)

    def extractor(self, info, leng_st_mask, batch_size):
        info.masked_fill_(leng_st_mask.unsqueeze(-1).expand(info.shape), value=0)
        return info

    def forward(self, info, leng_st):
        batch_size = info.size(0)
        leng_st_mask = get_leng_mask(leng_st)
        # (BS, S, EmbedSize) --> (BS, EmbedSize, S)
        info = torch.transpose(info, -1, 1).contiguous()
        info = self.cnn(info)
        # (BS, EmbedSize, S) --> (BS, S, EmbedSize)
        info = torch.transpose(info, -1, 1).contiguous()
        info = self.output(info, leng_st_mask,  batch_size)  # precess and restore

        for post_layer in self.postprocess:
            info = post_layer(info)
        return info
    
    
    # def __repr__(self):
    #     # We treat the extra repr like the sub-module, one item per line
    #     extra_lines = []
    #     extra_repr = self.extra_repr()
    #     # empty string will be split into list ['']
    #     if extra_repr:
    #         extra_lines = extra_repr.split('\n')
    #     child_lines = []
    #     for key, module in self._modules.items():
    #         mod_str = repr(module)
    #         mod_str = _addindent(mod_str, 2)
    #         child_lines.append('(' + key + '): ' + mod_str)
    #     lines = extra_lines + child_lines

    #     main_str = self._get_name() + '(' + self.struct_type.upper() + '): ' + '(' + str(self.input_size) + '->' + str(self.output_size) +') ' + '[INPUT] ' + self.input_type.upper() +'; ' + '[DIRECTION] ' + self.direction_type.upper() + '('
    #     if lines:
    #         # simple one-liner info, which most builtin Modules will use
    #         if len(extra_lines) == 1 and not child_lines:
    #             main_str += extra_lines[0]
    #         else:
    #             main_str += '\n  ' + '\n  '.join(lines) + '\n'

    #     main_str += ')'
    #     return main_str