import torch

from .cateembed import CateEmbeddingLayer
# # from .numeembed import NumeEmbeddingLayer 
from .llmembed import LLMEmbeddingLayer

class Expander_Layer(torch.nn.Module):
    
    def __init__(self, input_names_nnlvl, output_name_nnlvl, expander_para):
        super(Expander_Layer, self).__init__()
        
        
        # the input feature dim size and output feature dim size
        self.input_size = expander_para['input_size']
        self.output_size = expander_para['output_size']
        
        # input information
        assert len(input_names_nnlvl) == 1
        self.input_names_nnlvl = input_names_nnlvl
        self.input_name_nnlvl = input_names_nnlvl[0]
        
        # input with idx
        self.input_names_nnlvl_idx = [i for i in expander_para if self.input_name_nnlvl in i and 'idx' in i]
        
        # output information
        self.output_name_nnlvl = output_name_nnlvl
    
        # Part 1: NN
        self.EmbedDict = torch.nn.ModuleDict()
        for input_name_nnlvl_idx in self.input_names_nnlvl_idx:
            # for each input_name_nnlvl_idx, we assume we can find them in the INPUTS_TO_INFODICT
            assert 'Grn' in input_name_nnlvl_idx
            embed_para = expander_para[input_name_nnlvl_idx]
            
            nn_name, nn_para = embed_para['nn_name'], embed_para['nn_para']

            # input_name_nnlvl
            if nn_name.lower() == 'cateembed':
                Embed = CateEmbeddingLayer(**nn_para)
                self.EmbedDict[input_name_nnlvl_idx] = Embed
            elif nn_name.lower() == 'llmembed':
                Embed = LLMEmbeddingLayer(**nn_para)
                self.EmbedDict[input_name_nnlvl_idx] = Embed
            # elif nn_name.lower() == 'numeembed':
            #     # TODO: in the future
            else:
                raise ValueError(f'suffix is not correct "{self.input_name_nnlvl}"')

            self.embed_size = self.output_size
            
        self.use_wgt = expander_para['use_wgt']
        
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
        # names
        input_name_nnlvl = input_names_nnlvl[0]
        assert len(input_names_nnlvl) == 1
        assert input_name_nnlvl == self.input_name_nnlvl
    
        # get info_dict
        info_dict = INPUTS_TO_INFODICT[input_name_nnlvl]
        
        
        for input_name_nnlvl_idx in self.input_names_nnlvl_idx:
            assert input_name_nnlvl_idx in info_dict
        if self.use_wgt:
            assert (input_name_nnlvl + '_wgt') in info_dict

        # get embed
        embed_list = []
        for input_name_nnlvl_idx in self.input_names_nnlvl_idx:
            # INPUTS_TO_INFODICT is a part of batch_rfg
            holder = info_dict[input_name_nnlvl_idx]
            holder = holder.long()
            
            # print(holder.shape, f'<----- {input_name_nnlvl}')
            # leng_mask = holder == 0
            embed = self.EmbedDict[input_name_nnlvl_idx](holder)
            
            if self.use_wgt == True: 
                holder_wgt = info_dict[input_name_nnlvl + '_wgt']
                embed = embed * holder_wgt.unsqueeze(-1)
                
            # print(embed.shape, input_name_nnlvl)
            embed_list.append(embed)
            
        embed = torch.stack(embed_list, -2).mean(-2)
        
        for nn, layer in self.postprocess.items():
            embed = layer(embed)# masked_fill(leng_mask.unsqueeze(-1))
        
        return self.output_name_nnlvl, {'holder': holder, 'info': embed}
    