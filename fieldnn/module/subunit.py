import torch

# from fieldnn.basicnn.expander import Expander_Layer
# from fieldnn.basicnn.reducer import Reducer_Layer
# from fieldnn.basicnn.merger import Merger_Layer
# from fieldnn.basicnn.learner import Learner_Layer

from ..basicnn.expander import Expander_Layer
from ..basicnn.reducer import Reducer_Layer
from ..basicnn.merger import Merger_Layer
from ..basicnn.learner import Learner_Layer


class SubUnit_Layer(torch.nn.Module):
    '''Currently, it is not latest version'''
    
    def __init__(self, SubUnit_info):
        super(SubUnit_Layer, self).__init__()
        
        # the input names for the SubUnit
        self.SubUnit_input_names = SubUnit_info['input_names']
        
        # the output name for this SubUnit
        self.SubUnit_output_name = SubUnit_info['output_name']
        
        # get the SubUnit's BasicNN Config List
        self.SubUnit_BasicNN_Config_List = SubUnit_info['SubUnit_BasicNN_Config_List']
        
        
        # construct the LayersDict to hold all BasicNN within this SubUnit.
        self.LayersDict = torch.nn.ModuleDict()
        
        # initialize all the BasicNN for this SubUnit.
        for idx, BasicNN_Config_Dict in enumerate(self.SubUnit_BasicNN_Config_List):
            
            # nn_type_nn_name: like reducer-Max, learner-TFM, merger-Merger, expander-llmembed.
            nn_type_nn_name = BasicNN_Config_Dict['nn_type_nn_name']
            
            # Basic_Config: the Config for this NN.
            Basic_Config = BasicNN_Config_Dict['Basic_Config']
            
            input_names_nnlvl = Basic_Config['input_names_nnlvl']
            output_name_nnlvl = Basic_Config['output_name_nnlvl']
                
            if 'expander' in nn_type_nn_name:
                expander_para = Basic_Config['expander_para']
                NN = Expander_Layer(input_names_nnlvl, output_name_nnlvl, expander_para)
                self.LayersDict[f'{idx}_{nn_type_nn_name}'] = NN
                
            elif 'reducer' in nn_type_nn_name:
                reducer_para = Basic_Config['reducer_para']
                NN = Reducer_Layer(input_names_nnlvl, output_name_nnlvl, reducer_para)
                self.LayersDict[f'{idx}_{nn_type_nn_name}'] = NN
                
            elif 'merger' in nn_type_nn_name:
                merger_para = Basic_Config['merger_para']
                NN = Merger_Layer(input_names_nnlvl, output_name_nnlvl, merger_para)
                self.LayersDict[f'{idx}_{nn_type_nn_name}'] = NN
                
                
            elif 'learner' in nn_type_nn_name:
                learner_para = Basic_Config['learner_para']
                NN = Learner_Layer(input_names_nnlvl, output_name_nnlvl, learner_para)
                self.LayersDict[f'{idx}_{nn_type_nn_name}'] = NN
                
            else:
                raise ValueError(f'Current BasicNN {nn_type_nn_name} is not available')


    def forward(self, SubUnit_input_names, RECFLD_TO_TENSOR):

        INPUTS_TO_INFODICT = {}
        
        for idx, BasicNN_Config_Dict in enumerate(self.SubUnit_BasicNN_Config_List):
            
            nn_type_nn_name = BasicNN_Config_Dict['nn_type_nn_name']
            Basic_Config = BasicNN_Config_Dict['Basic_Config']
            input_names_nnlvl = Basic_Config['input_names_nnlvl']
            output_name_nnlvl = Basic_Config['output_name_nnlvl']
            
            NN = self.LayersDict[f'{idx}_{nn_type_nn_name}']
            
            # prepare the input.
            if idx == 0:
                for tensor_name in input_names_nnlvl:
                    INPUTS_TO_INFODICT[tensor_name] = RECFLD_TO_TENSOR[tensor_name]
            else:
                for input_name in input_names_nnlvl:
                    assert input_name in INPUTS_TO_INFODICT
                
            output_name_nnlvl, info_dict = NN(input_names_nnlvl, INPUTS_TO_INFODICT)
            
            # current output will be the input in the next round. 
            INPUTS_TO_INFODICT[output_name_nnlvl] = info_dict
            
        # pick up the SubUnit_output_name and its info_dict
        final_output_name_nnlvl = output_name_nnlvl
        SubUnit_output_name = self.SubUnit_output_name
        if not SubUnit_output_name in final_output_name_nnlvl:
            print(f'Mismatched SubUnit Output and Final NN Output: {final_output_name_nnlvl} vs {output_name_nnlvl}')
        
        info_dict = INPUTS_TO_INFODICT[final_output_name_nnlvl]
        
        return SubUnit_output_name, info_dict
    