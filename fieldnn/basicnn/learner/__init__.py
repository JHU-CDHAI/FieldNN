import torch

# from fieldnn.basicnn.learner.tfm import TFMLayer
# from fieldnn.basicnn.learner.linear import LinearLayer

from .tfm import TFMLayer
from .linear import LinearLayer

class Learner_Layer(torch.nn.Module):
    def __init__(self, 
                 input_names_nnlvl, output_name_nnlvl, learner_layer_para,
                ):
        super(Learner_Layer, self).__init__()
        
        # Part 0: Meta
        # here input_names and out_tensor just the tensor name, 
        # intead, the info_dict contains the corresponding real tensors.
        assert len(input_names_nnlvl) == 1
        self.input_names_nnlvl = input_names_nnlvl
        self.input_name_nnlvl = input_names_nnlvl[0]
        
        # output_name should be generated from the input_names
        self.output_name_nnlvl = output_name_nnlvl
        
        # the input feature dim size and output feature dim size
        self.input_size = learner_layer_para['input_size']
        self.output_size = learner_layer_para['output_size']
 
        # Part 1: NN
        nn_name, nn_para = learner_layer_para['nn_name'], learner_layer_para['nn_para']
        
        if nn_name.lower() == 'tfm':
            assert self.input_size == self.output_size
            self.Learner = TFMLayer(**nn_para)
        elif nn_name.lower() == 'linear':
            self.Learner = LinearLayer(**nn_para)
        # elif nn_name.lower() == 'cnn':
        #     self.Learner = CNNLayer(**nn_para)
        # elif nn_name.lower() == 'rnn':
        #     self.Learner = RNNLayer(**nn_para)
        else:
            raise ValueError(f'NN "{nn_name}" is not available')
        
        # Part 2: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in learner_layer_para['postprocess'].items():
            if method == 'dropout':
                self.postprocess[method] = torch.nn.Dropout(**config)
            elif method == 'layernorm':
                self.postprocess[method] = torch.nn.LayerNorm(self.output_size, **config)
        # self.Ignore_PSN_Layers = learner_layer_para['Ignore_PSN_Layers']
    
    
    def forward(self, input_names_nnlvl, INPUTS_TO_INFODICT):
        # information preparation.
        # 'INPUTS_TO_INFODICT` will come from SubUnit Layer.
        assert len(input_names_nnlvl) == 1
        input_name_nnlvl = input_names_nnlvl[0]
        assert self.input_name_nnlvl == input_name_nnlvl
        
        info_dict = INPUTS_TO_INFODICT[input_name_nnlvl]
        holder, info = info_dict['holder'], info_dict['info']
        
        # print(holder.shape, info.shape)
        # the following part is the data proprocessing
        holder, info = self.Learner(holder, info)
        
        for name, layer in self.postprocess.items():
            info = layer(info)
            
        # we don't need to change the holder here.
        return self.output_name_nnlvl, {'holder': holder, 'info': info}
    