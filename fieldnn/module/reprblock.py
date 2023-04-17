import torch
from .subunit import SubUnit_Layer


class ReprBlockLayer(torch.nn.Module):
    
    def __init__(self, df_SubUnit):
        super(ReprBlockLayer, self).__init__()
        self.df_SubUnit = df_SubUnit
        self.SubUnitDict = torch.nn.ModuleDict()
        
        for idx, SubUnit_info in df_SubUnit.iterrows():
            output_name = SubUnit_info['output_name']
            SubUnitLayer = SubUnit_Layer(SubUnit_info)
            self.SubUnitDict[output_name] = SubUnitLayer

    def forward(self, RECFLD_TO_EMBEDTESNOR):
        
        OUTPUT_TO_TENSOR = RECFLD_TO_EMBEDTESNOR.copy()
        
        for output_name, SubUnitLayer in self.SubUnitDict.items():
            input_names = SubUnitLayer.SubUnit_input_names
            SubUnit_output_name, info_dict = SubUnitLayer(input_names, OUTPUT_TO_TENSOR)
            
            assert output_name == SubUnit_output_name
            OUTPUT_TO_TENSOR[SubUnit_output_name] = info_dict
        
        return OUTPUT_TO_TENSOR
    