# fieldlm.nn.embedding
import os
import torch
import numpy as np

class CatEmbeddingLayer(torch.nn.Module):

    def __init__(self, 
                 vocab_size, 
                 embedding_size, 
                 init = 'random', 
                 freeze = False):
        
        super(CatEmbeddingLayer, self).__init__()
        
        # create embedding
        if init == 'random':
            # c. initial from random initialization
            self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx = 0)
        
        elif type(init) == np.ndarray:
            # a. load from pretrained array. Here init is an array.
            weight = torch.FloatTensor(init)
            assert weight.shape == (vocab_size, embedding_size)
            self.embedding = torch.nn.Embedding.from_pretrained(weight, freeze = freeze)
            
        elif os.path.isfile(init):
            # b. load from the pretrained array file.
            weight = torch.FloatTensor(np.load(init))
            assert tuple(weight.shape) == (vocab_size, embedding_size)
            self.embedding = torch.nn.Embedding.from_pretrained(weight, freeze = freeze)
        
        else:
            raise ValueError(f'In correct init method "{init}"')
        
    def forward(self, holder):
        # info is the grain
        info = self.embedding(holder)
        return info