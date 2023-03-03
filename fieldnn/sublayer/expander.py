import torch
import torch.nn.functional as F

# from fieldnn.nn.embedding import EmbeddingLayer
# from fieldnn.nn.lmembed import LMEmbedLayer
# from fieldnn.utils.layer import get_Layer2Holder, align_psn_idx

from ..nn.embedding import EmbeddingLayer
from ..nn.lmembed import LMEmbedLayer
from ..utils.layerfn import get_Layer2Holder, align_psn_idx


class Expander_Layer(torch.nn.Module):
    
    '''Only for Increasing Embedding Dimensions'''
    def __init__(self, input_fullname, output_fullname, expander_layer_para):
        super(Expander_Layer, self).__init__()
        
        # tensor.shape[-1]
        self.input_size = expander_layer_para['input_size']
        self.output_size = expander_layer_para['output_size']
        self.Ignore_PSN_Layers = expander_layer_para['Ignore_PSN_Layers']
        
        # Part 1: embedding
        self.input_fullname = input_fullname
        self.output_fullname = output_fullname
        
        assert 'Grn' == input_fullname[-3:]
        assert input_fullname.replace('Grn', '') == output_fullname
        
        nn_name, para = expander_layer_para[input_fullname]
        
        if nn_name.lower() == 'embedding':
            self.Embed = EmbeddingLayer(**para)
            self.embed_size = para['embedding_size']
        elif nn_name.lower() == 'lmembed':
            # TODO
            self.Embed = LMEmbedLayer(**para)
            self.embed_size = para['embedding_size']
        else:
            raise ValueError(f'NN "{nn_name}" is not available')
            
        assert self.embed_size == self.output_size
        
        # Part 2: PSN embedding
        # psn_layers = expander_layer_para['psn_layers']
        # self.PSN_Embed_Dict = torch.nn.ModuleDict()
        # for layername in psn_layers:
        #     para = generate_psn_embed_para(layername, self.embed_size)
        #     self.PSN_Embed_Dict[layername] = EmbeddingLayer(**para)
        
        # Part 3: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in expander_layer_para['postprocess'].items():
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
                
    # def get_psn_embed(self, fullname, holder):
    #     name = fullname.split('-')[-1]
    #     Layer2Idx = {v:idx for idx, v in enumerate(fullname.split('-'))}
    #     Layer2Holder = get_Layer2Holder(fullname, holder, self.Ignore_PSN_Layers)
        
    #     psn_embed = 0
    #     for source_layer, Embed in self.PSN_Embed_Dict.items():
    #         cpsn_idx = align_psn_idx(source_layer, name, Layer2Idx, Layer2Holder)
    #         psn_embed = psn_embed + Embed(cpsn_idx)
        
    #     return psn_embed

    def forward(self, fullname, holder, info = 'Empty'):
        '''
            fullname: full name of field, GRN is here
            holder: info_idx
            info: info_wgt
        '''
        assert self.input_fullname == fullname
        leng_mask = holder == 0
        embed = self.Embed(holder)
        
        if type(info) != str:
            embed = embed * info.unsqueeze(-1)
        
        # Comments: move these to Learner.
        # if len(self.PSN_Embed_Dict):
        #     psn_embed = self.get_psn_embed(fullname, holder)
        #     embed = embed + psn_embed
        
        for nn, layer in self.postprocess.items():
            embed = layer(embed)
        
        return self.output_fullname, holder, embed