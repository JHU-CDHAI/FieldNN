import os
import torch
import numpy as np

# from .expander import Expander_Layer
# from .learner import Learner_Layer
# from .merger import Merger_Layer

from ..sublayer.expander import Expander_Layer
from ..sublayer.learner import Learner_Layer
from ..sublayer.merger import Merger_Layer
from ..sublayer.reducer import Reducer_Layer


class Pipeline_Layer(torch.nn.Module):
    def __init__(self, pipeline_name, input_fullname, output_fullname, para_dict):
        super(Pipeline_Layer, self).__init__()
        
        self.pipeline_name = pipeline_name
        # self.input_fullname_list = input_fullname.split('^')
        self.input_fullname = input_fullname
        self.output_fullname = output_fullname
        self.para_dict = para_dict
        
        self.Layers = torch.nn.ModuleDict()
        for name, para in para_dict.items():
            if name == 'Expander':
                assert 'Grn' == input_fullname[-3:]
                assert input_fullname.replace('Grn', '') == output_fullname
                self.Layers[input_fullname] = Expander_Layer(input_fullname, output_fullname, para)
        
            elif name == 'Merger':
                assert len(input_fullname.split('^')) > 1
                self.Layers[input_fullname] = Merger_Layer(input_fullname, output_fullname, para)
        
            elif name == 'Reducer':
                self.Layers[input_fullname] = Reducer_Layer(input_fullname, output_fullname, para)
                
            elif name == 'Learner':
                assert output_fullname in para
                self.Layers[output_fullname] = Learner_Layer(output_fullname, output_fullname, para)
            else:
                raise ValueError(f'The sublayer name "{name}" is not available')
                
    def forward(self, fullname2data):
        for input_fullname, Layer in self.Layers.items():
            
            if '^' not in input_fullname:
                # holder, info = fullname2data.pop(input_fullname)
                # print(input_fullname, '<---input_fullname')
                # print(type(Layer), '<---Layer type')
                # print(Layer.input_fullname, '<---Layer type')
                # print(Layer.output_fullname, '<---Layer type')
                data = fullname2data.get(input_fullname)
                holder, info = data['holder'], data['info']
                # print(f'input_fullname: {input_fullname}, Layer Type {type(Layer)}')
                # print(holder.max())
                fullname, holder, info = Layer(input_fullname, holder, info)
                # print(fullname, '<--- output fullname')
                fullname2data[fullname] = {'holder': holder, 'info': info}
            else:
                input_fullname_list = input_fullname.split('^')
                # print(input_fullname)
                # fullname2data_copy = {k: fullname2data.pop(k) for k in input_fullname_list}
                fullname2data_copy = {k: fullname2data.get(k) for k in input_fullname_list}
                fullname, holder, info = Layer(input_fullname, fullname2data_copy)
                fullname2data[fullname] = {'holder': holder, 'info': info}
                
        return fullname2data
    