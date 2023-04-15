import torch
import torch.nn.functional as F

# from fieldnn.nn.embedding.cateembed import CateEmbeddingLayer
# from fieldnn.nn.embedding.numeembed import NumeEmbeddingLayer 
# from fieldnn.nn.embedding.llmembed import LLMEmbeddingLayer

from ..nn.embedding.cateembed import CateEmbeddingLayer
from ..nn.embedding.numeembed import NumeEmbeddingLayer 
from ..nn.embedding.llmembed import LLMEmbeddingLayer


class Expander_Layer(torch.nn.Module):
    
    '''Only for Increasing Input_idx's Order'''
    def __init__(self, full_recfldgrn, output_recfld, expander_para):
        super(Expander_Layer, self).__init__()
        
        self.input_size = expander_para['input_size']
        self.output_size = expander_para['output_size']
        
        # Part 1: embedding
        self.input_fullname = full_recfldgrn
        self.output_fullname = output_recfld
        
        assert 'Grn' in full_recfldgrn
        assert full_recfldgrn.split('Grn')[0] == output_recfld
        
        nn_name, embed_para = expander_para[self.input_fullname]
        
        if nn_name.lower() == 'cateembed':
            assert 'idx' in full_recfldgrn and 'LLM' not in full_recfldgrn
            self.Embed = CateEmbeddingLayer(**embed_para)
        elif nn_name.lower() == 'llmembed':
            assert 'idx' in full_recfldgrn and 'LLM' in full_recfldgrn
            self.Embed = LLMEmbeddingLayer(**embed_para)
        elif nn_name.lower() == 'numeembed':
            assert 'wgt' in full_recfldgrn
            self.Embed = NumeEmbeddingLayer(**embed_para)
        else:
            raise ValueError(f'suffix is not correct "{full_recfldgrn}"')

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

    def forward(self, fullname, holder, holder_wgt = 'Empty'):
        '''
            fullname: full name of field, GRN is here
            holder: info_idx
            info: info_wgt
        '''
        assert self.input_fullname == fullname
        leng_mask = holder == 0
        embed = self.Embed(holder)
        
        if type(holder_wgt) != str:
            embed = embed * holder_wgt.unsqueeze(-1)
        
        for nn, layer in self.postprocess.items():
            embed = layer(embed)
        
        return self.output_fullname, holder, embed