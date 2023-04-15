import os
import torch
import numpy as np

from ..nn.rnn import RNNLayer
from ..nn.cnn import CNNLayer
from ..nn.tfm import TFMLayer
from ..nn.linear import LinearLayer
from ..nn.embedding import EmbeddingLayer
from ..utils.layerfn import orderSeq, restoreSeq, align_psn_idx, get_Layer2Holder
from ..utils.parafn import generate_psn_embed_para

# from fieldnn.nn.rnn import RNNLayer
# from fieldnn.nn.cnn import CNNLayer
# from fieldnn.nn.tfm import TFMLayer
# from fieldnn.nn.linear import LinearLayer
# from fieldnn.nn.embedding import EmbeddingLayer
# from fieldnn.utils.layerfn import orderSeq, restoreSeq, align_psn_idx, get_Layer2Holder
# from fieldnn.utils.parafn import generate_psn_embed_para

class Learner_Layer(torch.nn.Module):
    def __init__(self, input_fullname, output_fullname, learner_layer_para):
        super(Learner_Layer, self).__init__()
        
        # Part 0: Meta
        self.fullname = input_fullname
        
        self.input_fullname = input_fullname
        self.output_fullname = output_fullname
        assert self.input_fullname == self.output_fullname
        
        self.input_size = learner_layer_para['input_size']
        self.output_size = learner_layer_para['output_size']
        self.embed_size = self.input_size
        
        # Part 1: NN
        nn_name, para = learner_layer_para[self.fullname]
        if len(input_fullname.split('-')) == 2:
            assert nn_name.lower() == 'linear'
            assert len(learner_layer_para['psn_layers']) == 0
            self.forward = self.forward_ln
        else:
            self.forward = self.forward_tfm
        
        if nn_name.lower() == 'cnn':
            self.Learner = CNNLayer(**para)
        elif nn_name.lower() == 'rnn':
            self.Learner = RNNLayer(**para)
        elif nn_name.lower() == 'tfm':
            assert self.input_size == self.output_size
            self.Learner = TFMLayer(**para)
        elif nn_name.lower() == 'linear':
            self.Learner = LinearLayer(**para)
        else:
            raise ValueError(f'NN "{nn_name}" is not available')

        
        psn_layers = learner_layer_para['psn_layers']
        if len(psn_layers) > 0:
            # Part 2: PSN embedding
            self.PSN_Embed_Dict = torch.nn.ModuleDict()
            for layername in psn_layers:
                para = generate_psn_embed_para(layername, self.embed_size)
                self.PSN_Embed_Dict[layername] = EmbeddingLayer(**para)
            
            # Part 2: EmbedProcess
            self.embedprocess = torch.nn.ModuleDict()
            for method, config in learner_layer_para['embedprocess'].items():
                if method == 'dropout':
                    self.embedprocess[method] = torch.nn.Dropout(**config)
                elif method == 'layernorm':
                    self.embedprocess[method] = torch.nn.LayerNorm(self.output_size, **config)
        
        # Part 3: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in learner_layer_para['postprocess'].items():
            if method == 'dropout':
                self.postprocess[method] = torch.nn.Dropout(**config)
            elif method == 'layernorm':
                self.postprocess[method] = torch.nn.LayerNorm(self.output_size, **config)

        self.Ignore_PSN_Layers = learner_layer_para['Ignore_PSN_Layers']

        
    def get_psn_embed(self, fullname, holder):
        name = fullname.split('-')[-1]
        Layer2Idx = {v:idx for idx, v in enumerate(fullname.split('-'))}
        Layer2Holder = get_Layer2Holder(fullname, holder, self.Ignore_PSN_Layers)
        
        psn_embed = 0
        for source_layer, Embed in self.PSN_Embed_Dict.items():
            cpsn_idx = align_psn_idx(source_layer, name, Layer2Idx, Layer2Holder)
            psn_embed = psn_embed + Embed(cpsn_idx)
        return psn_embed
    
    def reshape(self, info, leng_mask):
        nbs = np.array(info.shape[:-2]).prod()
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
        
    def forward_ln(self, fullname, holder, info):
        # print(fullname, '<--------fullname')
        # print(self.input_fullname, '<--------self.input_fullname')
        assert self.input_fullname == fullname
        info = self.Learner(info)
        for nn_name, layer in self.postprocess.items():
            info = layer(info)
        return self.output_fullname, holder, info
    
    def forward_tfm(self, fullname, holder, info):
        # print(fullname, '<--------fullname')
        # print(self.input_fullname, '<--------self.input_fullname')
        assert self.input_fullname == fullname
        
        # TODO: this step needs close review. 
        # (1) adding psn embed
        if len(self.PSN_Embed_Dict):
            psn_embed = self.get_psn_embed(fullname, holder)
            for nn, layer in self.embedprocess.items():
                psn_embed = layer(psn_embed)
            info = info + psn_embed
            
        # (2) do the calculation
        leng_mask = holder == 0

        # print(holder.shape)
        ord_info, ord_leng_mask, r_ix = self.reshape(info, leng_mask)
        ord_info_output = self.Learner(ord_info, ord_leng_mask)
        info = self.restore(ord_info_output, leng_mask, r_ix)
        
        # (3) post-process
        for nn_name, layer in self.postprocess.items():
            info = layer(info)
            
        # we do not change the fullname and holder
        return self.output_fullname, holder, info