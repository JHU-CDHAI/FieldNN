import torch

# from fieldnn.basicnn.expander.cateembed import CateEmbeddingLayer
# from fieldnn.basicnn.expander.numeembed import NumeEmbeddingLayer 
# from fieldnn.basicnn.expander.llmembed import LLMEmbeddingLayer

from .cateembed import CateEmbeddingLayer
from .numeembed import NumeEmbeddingLayer 
from .llmembed import LLMEmbeddingLayer

class Expander_Layer(torch.nn.Module):
    
    def __init__(self, input_names_nnlvl, output_name_nnlvl, expander_para):
        super(Expander_Layer, self).__init__()
        
        assert len(input_names_nnlvl) == 1
        self.input_names_nnlvl = input_names_nnlvl
        self.input_name_nnlvl = input_names_nnlvl[0]
        
        # output_name should be generated from the input_names
        self.output_name_nnlvl = output_name_nnlvl
        # self.output_name = output_name

        assert 'Grn' in self.input_name_nnlvl
        assert self.input_name_nnlvl.split('Grn')[0] == self.output_name_nnlvl
        
        # the input feature dim size and output feature dim size
        self.input_size = expander_para['input_size']
        self.output_size = expander_para['output_size']
        
        # Part 1: NN
        nn_name, nn_para = expander_para['nn_name'], expander_para['nn_para']
        
        if nn_name.lower() == 'cateembed':
            assert 'idx' in self.input_name_nnlvl and 'LLM' not in self.input_name_nnlvl
            self.Embed = CateEmbeddingLayer(**nn_para)
        elif nn_name.lower() == 'llmembed':
            assert 'idx' in self.input_name_nnlvl and 'LLM' in self.input_name_nnlvl
            self.Embed = LLMEmbeddingLayer(**nn_para)
        elif nn_name.lower() == 'numeembed':
            assert 'wgt' in self.input_name_nnlvl
            self.Embed = NumeEmbeddingLayer(**nn_para)
        else:
            raise ValueError(f'suffix is not correct "{self.input_name_nnlvl}"')

        self.embed_size = self.output_size
        
        # Part 2: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in expander_para['postprocess'].items():
            if method == 'dropout':
                self.postprocess[method] = torch.nn.Dropout(**config)
            elif method == 'activator':
                activator = config
                if activator.lower() == 'relu': 
                    self.postprocess[method] = torch.nn.ReLU()
                elif activator.lower() == 'tanh': 
                    self.postprocess[method] = torch.nn.Tanh()
                elif activator.lower() == 'gelu':
                    self.postprocess[method] = torch.nn.GELU()
            elif method == 'layernorm':
                self.postprocess[method] = torch.nn.LayerNorm(self.embed_size, **config)

    def forward(self, input_names_nnlvl, INPUTS_TO_INFODICT):
        '''
            info_dict
            input_names_nnlvl: full name of field, GRN is here
            holder: holder # i.e., info_idx
            holder_wgt: holder_wgt
        '''
        input_name_nnlvl = input_names_nnlvl[0]
        assert len(input_names_nnlvl) == 1
        assert input_name_nnlvl == self.input_name_nnlvl
        
        info_dict = INPUTS_TO_INFODICT[input_name_nnlvl]
        
        # 1. holder information
        holder = info_dict['holder']
        leng_mask = holder == 0
        embed = self.Embed(holder)
        
        # 2. holder_wgt information
        holder_wgt = info_dict['holder_wgt']
        if type(holder_wgt) != str:
            embed = embed * holder_wgt.unsqueeze(-1)
        
        # 3. post process.
        for nn, layer in self.postprocess.items():
            embed = layer(embed)
        
        return self.output_name_nnlvl, {'holder': holder, 'info': embed}
    