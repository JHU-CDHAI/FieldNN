# write it as a function

# Convert the above block to a function
from .reducerfn import get_reducer_para
from .mergerfn import get_merger_para
# from .expanderfn import get_expander_para


############################################# Hyperparameters
default_BasicNNtype_To_NNName = {
    'expander': None, # will be updated according to the Grn Type
    'reducer': 'Max',
    'merger': 'Merger',
    'learner': None, # TODO: ignore this currently
    
}
#############################################

def mapping_SubUnitName_to_SubUnitNNList(SubUnitName, input_names, 
                                         default_BasicNNtype_To_NNName):
    SubUnitNNList = []
    for NAME in SubUnitName:
        if NAME == 'E':
            nn_type = 'expander'
            input_name = input_names[0]
            
            assert 'Grn' in input_name
            assert len(input_names) == 1
            
            if 'Tknz' in input_name:
                nn_name = 'LLMEmbed'
            elif '_wgt' in input_name:
                nn_name = 'NumeEmbed'
            else:
                nn_name = 'CateEmbed'
            
        elif NAME == 'R':
            assert len(input_names) == 1
            nn_type = 'reducer'
            nn_name = default_BasicNNtype_To_NNName[nn_type]
            
        elif NAME == 'M':
            nn_type = 'merger'
            nn_name = default_BasicNNtype_To_NNName[nn_type]
            
        elif NAME == 'L':
            assert len(input_names) == 1
            nn_type = 'learner'
            nn_name = default_BasicNNtype_To_NNName[nn_type]
            
        else:
            raise ValueError(f'The BasicNN is not correct {NAME}')
        
        SubUnitNNList.append(nn_type + '-' + nn_name)
    return SubUnitNNList


def generate_BasicNN_Config(nn_type_nn_name, 
                            input_names_nnlvl, 
                            default_nnpara, 
                            embed_size, 
                            process):
    '''
        please notince here, this function is not the final version yet.
    
    '''
    nn_type, nn_name = nn_type_nn_name.split('-')

    # Image here you are in the loop already.
    # Current Iteration is 0

    # actually, the following code only deals with nn_type == 'reducer'
    # We will add the following condition in the whole loop.
    if nn_type == 'reducer': 
        # assert len(SubUnit_input_names) == 1
        # Get the input_names_nnlvl
        # input_names_nnlvl = SubUnit_input_names # this assigments only works for the first iteration
        assert len(input_names_nnlvl) == 1
        fld = input_names_nnlvl[0].split('-')[-1]
        # Get the output_name_nnlvl
        output_name_nnlvl = input_names_nnlvl[0].replace('-' + fld, '@' + fld)

        # Prepare the para for the NN layer
        nn_para = default_nnpara # this will be updated.

        # Get the input_size
        input_size = embed_size

        # Get the output_size
        if nn_name.lower() == 'concat':
            # this types of merger only happened in the R in MLRL. 
            # you can skip this part safely if you haven't encountered M. 
            fld_childflds = input_names_nnlvl[0].split('-')[-1]
            assert '@' in fld_childflds
            childflds = fld_childflds.split('@')[-1]
            childflds = childflds.split('&')
            output_size = len(childflds) * embed_size
        else:
            # usually the most case. 
            output_size = embed_size 

        # Get the postprocess
        postprocess = process

        # Derive the para
        para = get_reducer_para(nn_name, nn_para, input_size, output_size, postprocess)  
        
    elif nn_type == 'merger':
        # generate the output name
        
        assert len(input_names_nnlvl) > 1
        childflds = [i.split('-')[-1] for i in input_names_nnlvl]
        fld_childflds = '&'.join(childflds)
        output_prefix = input_names_nnlvl[0].replace('-' + childflds[0], '')
        output_name_nnlvl = output_prefix + '@' + fld_childflds
        
        # Prepare the para for the NN layer
        nn_para = default_nnpara # this will be updated.

        # Get the input_size
        input_size = embed_size
        output_size = embed_size

        # Get the postprocess
        postprocess = process

        # Derive the para
        para = get_merger_para(nn_name, nn_para, input_size, output_size, postprocess) 
        
    else:
        raise ValueError(f'nn_type {nn_type} is not available yet')

    # have a look at the para here. 
    # print(input_names_nnlvl, '<--------- input_names_nnlvl')
    # print(output_name_nnlvl, '<--------- output_name_nnlvl')
    # para

    Basic_Config = {'input_names_nnlvl': input_names_nnlvl, 
                      'output_name_nnlvl': output_name_nnlvl, 
                      f'{nn_type}_para': para}
    
    return Basic_Config



# the question is how to create the SubUnit_BasicNN_List for each SubUnit
# we will turn to this later. 
# print(SubUnit_BasicNN_List)

def get_SubUnit_BasicNN_Config_List(SubUnit_BasicNN_List, 
                                    SubUnit_input_names, 
                                    SubUnit_output_name, 
                                    default_nnpara, 
                                    embed_size, 
                                    process
                                   ):
    
    
    # TODO: also add the layer_idx in order to deal with the learn_layer_para.
    # This function also needs to be updated. 
    
    SubUnit_BasicNN_Config_List = []
    for basic_nn_idx, nn_type_nn_name in enumerate(SubUnit_BasicNN_List):

        # Get the input_names_nnlvl
        if basic_nn_idx == 0:
            input_names_nnlvl = SubUnit_input_names # this assigments only works for the first iteration
        else:
            input_names_nnlvl = [output_name_nnlvl]

        # print(nn_type_nn_name)
        Basic_Config = generate_BasicNN_Config(nn_type_nn_name, 
                                               input_names_nnlvl, 
                                               default_nnpara, 
                                               embed_size, 
                                               process)
        output_name_nnlvl = Basic_Config['output_name_nnlvl']
        BasicNN_Config = {'nn_type_nn_name': nn_type_nn_name, 'Basic_Config': Basic_Config}
        SubUnit_BasicNN_Config_List.append(BasicNN_Config)
        
        # also check the input_size of dim and output_size of dim

    final_output_name_nnlvl = output_name_nnlvl

    # print(final_output_name_nnlvl, '<------- final_output_name_nnlvl')
    # print(SubUnit_output_name, '<------- SubUnit_output_name')
    # assert SubUnit_output_name in final_output_name_nnlvl
    return SubUnit_BasicNN_Config_List