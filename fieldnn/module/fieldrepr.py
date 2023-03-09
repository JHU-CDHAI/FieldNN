import os
import torch
import numpy as np

from .struct import Struct_Layer
from ..utils.parafn import process_sublayer_name

class FieldRepr_Layer(torch.nn.Module):
    def __init__(self, FLD_LIST, FLD_END, meta_para):
        super(FieldRepr_Layer, self).__init__()
        
        df_struct = get_structures_from_fldlist(FLD_LIST)
        tmp = df_struct.sort_values('layers', ascending = False)
        layer2structlist = dict(zip(tmp['layers'].to_list(), tmp['struct_name'].to_list()))
        
        NAME_2_FULLNAME = {i.split('-')[-1]:i for i in FLD_LIST}

        self.FLD_LIST = FLD_LIST
        self.FLD_END = FLD_END
        self.NAME_2_FULLNAME = NAME_2_FULLNAME

        self.LAYERS = torch.nn.ModuleDict()
        for layer, structlist in layer2structlist.items():
            self.LAYERS[layer] = torch.nn.ModuleDict()
            for struct_name in structlist:
                
                # construct struct_para
                fullname_input_list, final_fullname_output, struct_model, NAME_2_FULLNAME = get_struct_info(struct_name, NAME_2_FULLNAME)
                fullname_input = '^'.join(fullname_input_list)
                D_model, D_data = generate_structure(fullname_input_list, struct_model)
                struct_para = {}
                struct_para['fullname_input_list'] = fullname_input_list
                struct_para['fullname_input'] = fullname_input
                struct_para['final_fullname_output'] = final_fullname_output
                struct_para['struct_model'] = struct_model
                struct_para['D_model'] = D_model
                struct_para['D_data'] = D_data
                self.LAYERS[layer][struct_name] = Struct_Layer(struct_name, struct_para, meta_para)

    def forward(self, FLD_2_DATA):
        for layer, LayerDict in self.LAYERS.items():
            for struct_name, StructLayer in LayerDict.items():
                print(f'\nstruct_name <---------- {struct_name} ')
                print([i for i in FLD_2_DATA])
                FLD_2_DATA = StructLayer(FLD_2_DATA)
                print([i for i in FLD_2_DATA])
                
        assert self.FLD_END in FLD_2_DATA
        return FLD_2_DATA[self.FLD_END]