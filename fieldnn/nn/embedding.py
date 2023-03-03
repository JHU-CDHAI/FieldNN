# fieldlm.nn.embedding
import os
import torch
import numpy as np

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, 
                 input_size, 
                 embedding_size, 
                 init = 'init', 
                 freeze = False):
        
        super(EmbeddingLayer, self).__init__()
        
        # (+) self.embedding
        if type(init) == np.ndarray:
            # 1. from given array
            weight = torch.FloatTensor(init)
            assert weight.shape == (input_size, embedding_size)
            self.embedding = torch.nn.Embedding.from_pretrained(weight, freeze = freeze)
            
        elif os.path.isfile(init):
            weight = torch.FloatTensor(np.load(init))
            assert tuple(weight.shape) == (input_size, embedding_size)
            
            self.embedding = torch.nn.Embedding.from_pretrained(weight, freeze = freeze)
            
        else:
            # from random initialization
            self.embedding = torch.nn.Embedding(input_size, embedding_size, padding_idx = 0)
        
    def forward(self, info):
        # tensor0 to tensor1
        info = self.embedding(info)
        return info