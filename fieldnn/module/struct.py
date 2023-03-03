import os
import torch
import numpy as np

from .pipeline import Pipeline_Layer
from ..utils.parafn import process_sublayer_name

class Struct_Layer(torch.nn.Module):
    def __init__(self, struct_name, struct_para, meta_para):
        super(Struct_Layer, self).__init__()
        self.struct_name = struct_name
        
        self.final_fullname_output = struct_para['final_fullname_output']
        self.D_model = struct_para['D_model'] 
        self.D_data = struct_para['D_data'] 
        
        
        self.FLD_2_VOCABSIZE = meta_para['FLD_2_VOCABSIZE']
        self.embed_size = meta_para['embed_size']
        self.default_learner_para = meta_para['default_learner_para']
        self.default_reducer_para = meta_para['default_reducer_para']
        self.expander_process = meta_para['expander_process']
        self.default_process = meta_para['default_process']
        self.Ignore_PSN_Layers = meta_para['Ignore_PSN_Layers']
        
        self.Layers = torch.nn.ModuleDict()
        
    
        for input_fullname, pipeline_list in self.D_model.items():
            self.Layers[input_fullname] = torch.nn.ModuleDict() 
            for pipeline_name in pipeline_list:
                input_fullname_list, output_fullname, para_dict = process_sublayer_name(pipeline_name, self.FLD_2_VOCABSIZE, self.embed_size, 
                                                                                        self.default_learner_para,  self.default_reducer_para,
                                                                                        self.expander_process, self.default_process, self.Ignore_PSN_Layers)
                input_fullname = '^'.join(input_fullname_list)
                PipeLine = Pipeline_Layer(pipeline_name, input_fullname, output_fullname, para_dict)
                self.Layers[input_fullname][pipeline_name] = PipeLine

    def forward(self, FLD_2_DATA):
        for input_fullname, output_full_name in self.D_data.items():
            for pipeline_name, Pipeline in self.Layers[input_fullname].items():
                FLD_2_DATA = Pipeline(FLD_2_DATA)
            assert output_full_name in FLD_2_DATA
            
        # update the new output name to final_fullname_output
        assert self.final_fullname_output in output_full_name
        # fullname2data[self.final_fullname_output] = fullname2data.pop(output_full_name)
        FLD_2_DATA[self.final_fullname_output] = FLD_2_DATA.get(output_full_name)
        
        return FLD_2_DATA