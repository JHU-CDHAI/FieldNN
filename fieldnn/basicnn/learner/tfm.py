import torch
import numpy as np
# from fieldnn.utils.layerfn import orderSeq, restoreSeq
from ...utils.layerfn import orderSeq, restoreSeq

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class TFMLayer(torch.nn.Module):
    def __init__(self, 
                 input_size = 200, 
                 output_size = 200, # d_model
                 nhead = 8,
                 num_encoder_layers = 6, # only have encoder part
                 num_decoder_layers = 0, # in default, we don't need decoder part. 
                 dim_feedforward = 2048, 
                 tfm_dropout = 0.1,
                 tfm_activation = 'relu', 
                 psn_max = 512, 
                 psn_embedprocess = {}):
        
        '''https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py'''

        super(TFMLayer,self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.input_size = input_size
        self.tfm_input_size = input_size
        self.n_directions = 1
        self.output_size = output_size
        assert output_size % self.n_directions == 0 
        self.hidden_size = int(output_size / self.n_directions)
        assert self.hidden_size == self.tfm_input_size
        # self.psn_size = psn_size 
        
        
        
        self.dim_feedforward = dim_feedforward
        
        self.transformer  = torch.nn.Transformer(d_model = self.hidden_size, 
                                                 nhead = nhead,
                                                 num_encoder_layers = self.num_encoder_layers,
                                                 num_decoder_layers = self.num_decoder_layers,
                                                 dim_feedforward = dim_feedforward, 
                                                 dropout = tfm_dropout,
                                                 activation = tfm_activation,
                                                 batch_first = True,
                                                 # src_mask_flag = False, # see all tokens in a sentence 
                                                 # # This IS THE NEW PART. NOT PyTorch.nn.
                                                 )
        
        # Part a: PSN embedding
        # (bs, xxx, psn_max, dim)
        self.psn_max = psn_max
        self.psn_embedding = torch.nn.Embedding(self.psn_max + 1, 
                                                self.input_size, 
                                                padding_idx = 0)
        
        # Part b: PSN EmbedProcess
        self.psn_embedprocess = torch.nn.ModuleDict()
        for method, config in psn_embedprocess.items():
            if method == 'dropout':
                self.psn_embedprocess[method] = torch.nn.Dropout(**config)
            elif method == 'layernorm':
                self.psn_embedprocess[method] = torch.nn.LayerNorm(self.output_size, **config)
            else:
                raise ValueError(f'no avialable embedprocess method {method}')
                
        # you can either choose v1 or v2 forward method.
        # self.forward = self.forward_v1
        self.forward = self.forward_v2
    
    def forward_v1(self, holder, info):
        
        # (1) get the leng_mask
        leng_mask = holder == 0
        
        
        # (2.1) get the psn_embed 
        psn_id = self.generate_psnidx(leng_mask) 
        psn_embed = self.psn_embedding(psn_id)
        # print(psn_embed[0, 0, 0, :, 0], '<------------- psn_embed 1')
        
        # (2.2) TODO: process psn_embed? Do we need the further embed process? 
        for nn, layer in self.psn_embedprocess.items(): 
            psn_embed = layer(psn_embed)
            
        # print(psn_embed[0, 0, 0, :, 0], '<------------- psn_embed 2')
        
        # (2.3) add psn_embed to info
        info = info + psn_embed
        # print(info[0, 0, 0, :, 0], '<------------- info = info + psn_embed')
        
        
        # (3) reshape
        ord_info, ord_leng_mask, r_ix = self.reshape(info, leng_mask)
        
        # print(ord_info[0, :, 0], '<------------- ord_info')
        
        # (4) do the transformer calculator
        ord_info_output = self.transformer(ord_info, ord_info, 
                                           src_key_padding_mask = ord_leng_mask,  
                                           tgt_key_padding_mask = ord_leng_mask)
        
        # print(ord_info_output[0, :, 0], '<------------- ord_info_output')
        
        # (5) restore
        info = self.restore(ord_info_output, leng_mask, r_ix)
        # print(info[0, 0, 0, :, 0], '<------------- info = self.restore(ord_info_output, leng_mask, r_ix)')
            
        return holder, info
    
    
    def forward_v2(self, holder, info):
        # (1) get the leng_mask
        leng_mask = holder == 0
        
        # print(info[0, 0, 0, :, 0], '<------------- info 1')
        
        
        # (2) reshape 
        ord_info, ord_leng_mask, r_ix = self.reshape(info, leng_mask)
        # print(ord_info[0, :, 0], '<------------- ord_info')
        
        # (3.1) get the psn_embed 
        psn_id = self.generate_psnidx(ord_leng_mask) 
        psn_embed = self.psn_embedding(psn_id)
        # print(psn_embed[0, :, 0], '<------------- psn_embed 1')
        
        
        # (3.2) TODO: process psn_embed? Do we need the further embed process? 
        for nn, layer in self.psn_embedprocess.items(): 
            psn_embed = layer(psn_embed)
            
        # print(psn_embed[0, :, 0], '<------------- psn_embed 2')
        
        
        # (3.3) add psn_embed to info
        ord_info = ord_info + psn_embed
        
        # print(ord_info[0, :, 0], '<------------- ord_info 1')
        
    
        # (4) do the transformer calculator
        ord_info_output = self.transformer(ord_info, ord_info, 
                                           src_key_padding_mask = ord_leng_mask,  
                                           tgt_key_padding_mask = ord_leng_mask)
        
        # print(ord_info_output[0, :, 0], '<------------- ord_info_output 1')
        
        
        # (5) restore
        info = self.restore(ord_info_output, leng_mask, r_ix)
        # print(info[0, 0, 0, :, 0], '<------------- info 2')
        
        return holder, info
    
    
    
    def reshape(self, info, leng_mask):
        nbs = int(np.array(info.shape[:-2]).prod())
        ngrn, dim = info.shape[-2:]
        # print(nbs, ngrn, dim)
        
        tmp_info = info.contiguous().view(nbs, ngrn, dim)
        # print(tmp_info.shape)

        tmp_leng_mask = leng_mask.contiguous().view(nbs, ngrn)
        # print(tmp_leng_mask.shape)

        tmp_leng = (tmp_leng_mask == 0).sum(-1)
        # print(tmp_leng.shape)
        
        ord_info,      ord_leng, r_idx = orderSeq(tmp_info, tmp_leng)
        ord_leng_mask, ord_leng, r_idx = orderSeq(tmp_leng_mask, tmp_leng)
        return ord_info, ord_leng_mask, r_idx
    
    def restore(self, ord_info_output, leng_mask, r_idx):
        info_new = restoreSeq(ord_info_output, r_idx)
        output_size = info_new.shape[-1]
        info_output = info_new.view(*list(leng_mask.shape) + [output_size])
        return info_output
        
    def generate_psnidx(self, leng_mask):
        # leng_mask = holder == 0
        psn_idx = (leng_mask == False).cumsum(-1).masked_fill(leng_mask, 0)
        return psn_idx
    
    
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + f'LEARNER(TFM): input({self.input_size}), output({self.output_size}): ('
        lines = [f'(Encoder): EncoderLayer(layers_num={self.num_encoder_layers}, dim_feedforward={self.dim_feedforward})', 
                 f'(Decoder): DecoderLayer(layers_num={self.num_decoder_layers}, dim_feedforward={self.dim_feedforward})']
        main_str += '\n  ' + '\n  '.join(lines) + '\n' + ')'
        
        # if lines:
        #     # simple one-liner info, which most builtin Modules will use
        #     if len(extra_lines) == 1 and not child_lines:
        #         main_str += extra_lines[0]
        #     else:
        #         main_str += '\n  ' + '\n  '.join(lines) + '\n'
        # main_str += ')'
        return main_str
    