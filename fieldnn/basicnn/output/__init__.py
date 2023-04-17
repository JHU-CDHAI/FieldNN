import torch
from .crf import CRF
from .softmax import SoftmaxLayer

class Output_Layer(torch.nn.Module):
    # TODO:
    def __init__(self, input_names_nnlvl, output_name_nnlvl, output_layer_para):
        super(Output_Layer, self).__init__()
        
    def forward(self, input_names_nnlvl, INPUTS_TO_INFODICT):
        return # self.output_name_nnlvl, {'holder': holder, 'info': info}
    