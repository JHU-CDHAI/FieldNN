import torch
import numpy as np
import torch.nn.functional as F

class TFMLayer(torch.nn.Module):
    def __init__(self, 
                 input_size = 512, 
                 output_size = 512, # d_model
                 nhead = 8,
                 num_encoder_layers = 6, # only have encoder part
                 num_decoder_layers = 0, # in default, we don't need decoder part. 
                 dim_feedforward = 2048, 
                 tfm_dropout = 0.1,
                 tfm_activation = 'relu'):
        
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
        # self.postprocess = []
        # for method, use_config in postprecess.items():
        #     use, config = use_config
        #     if use == False: continue
        #     if method == 'activator':
        #         activator = config
        #         if activator.lower() == 'relu': 
        #             self.activator = F.relu
        #         elif activator.lower() == 'tanh': 
        #             self.activator = F.tanh
        #         elif activator.lower() == 'gelu':
        #             self.activator = F.gelu
        #         else:
        #             self.activator = lambda x: x
        #         self.postprocess.append(self.activator)
            
        #     if method == 'dropout':
        #         self.drop = torch.nn.Dropout(**config)
        #         self.postprocess.append(self.drop)
                
        #     elif method == 'layernorm':
        #         self.layernorm = torch.nn.LayerNorm(self.output_size, **config)
        #         self.postprocess.append(self.layernorm)


    def forward(self, info, leng_mask):
        info = self.transformer(info, info, src_key_padding_mask = leng_mask,  tgt_key_padding_mask  = leng_mask)
        # for layer in self.postprocess:
        #     info = layer(info)
        return info