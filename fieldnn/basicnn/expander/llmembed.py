import os
import torch
import numpy as np
from ...utils.layerfn import orderSeq, restoreSeq
# from fieldnn.utils.layerfn import orderSeq, restoreSeq
from transformers import AutoModel


class LLMEmbeddingLayer(torch.nn.Module):

    def __init__(self, 
                 tokenizer, 
                 embedding_size, 
                 init, 
                 freeze = False):
        
        super(LLMEmbeddingLayer, self).__init__()
        
        self.tokenizer = tokenizer
        assert init == tokenizer.name_or_path 
        
        self.LLM = AutoModel.from_pretrained(init)
        self.hidden_size = self.LLM.config.hidden_size
        self.embedding_size = embedding_size
        
        self.linear  = torch.nn.Linear(self.hidden_size, self.embedding_size)
        self.init_linear_weights()
            
    def init_linear_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        
        
    def forward(self, holder):
        # 1. get leng_mask
        leng_mask = holder == 0
        
        # 2. get ordered holder
        ord_holder, ord_leng_mask, r_idx = self.reshape(holder, leng_mask)
        
        # 3. embedding ordered holder by LLM
        # expanding by the HuggingFace Language Model
        
        # 3.1 we might want to freeze LLM here
        # print(ord_holder.shape, '<--- ord_holder.shape')
        output = self.LLM(ord_holder)
        # 3.2 adjust the hidden dimension
        ord_info_output = output['last_hidden_state']
        ord_info_output = self.linear(ord_info_output) # bias might not be zeros.
        ord_info_output = ord_info_output.masked_fill(ord_leng_mask.unsqueeze(-1), 0)
    
        # 4. restore orderded output to original shape
        info = self.restore(ord_info_output, leng_mask, r_idx)
        
        return info
    
    
    def reshape(self, holder, leng_mask):
        nbs = np.array(holder.shape[:-1]).prod()
        ngrn = holder.shape[-1]
        # print(nbs, ngrn, dim)
        
        tmp_holder = holder.contiguous().view(nbs, ngrn)
        # print(tmp_info.shape)

        tmp_leng_mask = leng_mask.contiguous().view(nbs, ngrn)
        # print(tmp_leng_mask.shape)

        tmp_leng = (tmp_leng_mask == 0).sum(-1)
        # print(tmp_leng.shape)
        
        ord_holder,    ord_leng, r_idx = orderSeq(tmp_holder, tmp_leng)
        ord_leng_mask, ord_leng, r_idx = orderSeq(tmp_leng_mask, tmp_leng)
        return ord_holder, ord_leng_mask, r_idx
    
    def restore(self, ord_info_output, leng_mask, r_idx):
        info_new = restoreSeq(ord_info_output, r_idx)
        output_size = info_new.shape[-1]
        info_output = info_new.view(*list(leng_mask.shape) + [output_size])
        return info_output
        