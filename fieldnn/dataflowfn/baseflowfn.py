# to fieldnn.dataflowfn.baseflow.py

from ..configfn.expanderfn import get_expander_para
from ..configfn.mergerfn import get_merger_para
from ..configfn.reducerfn import get_reducer_para
from ..configfn.learnerfn import get_learner_para
from .embedflowfn import get_Default_ExpanderNNPara

def mapping_SubUnitName_to_SubUnitNNList(SubUnitName, input_names, 
                                         default_BasicNNtype_To_NNName):
    SubUnitNNList = []
    for idx, NAME in enumerate(SubUnitName):
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
            if idx == 0: assert len(input_names) == 1
            nn_type = 'reducer'
            nn_name = default_BasicNNtype_To_NNName[nn_type]
            
        elif NAME == 'M':
            nn_type = 'merger'
            nn_name = default_BasicNNtype_To_NNName[nn_type]
            
        elif NAME == 'L':
            # print(input_names)
            assert idx != 0
            # assert len(input_names) == 1; not necessary, L can follow M
            nn_type = 'learner'
            input_name = input_names[0]
            layers_num = len(input_name.split('-'))
            if layers_num >= 3:
                nn_name = 'TFM'
            elif layers_num == 2:
                nn_name = 'Linear'
            else:
                raise ValueError(f'incorrect layers_num {layers_num}')
            
        else:
            raise ValueError(f'The BasicNN is not correct {NAME}')
        
        SubUnitNNList.append(nn_type + '-' + nn_name)
    return SubUnitNNList


def get_SubUnit_Default_NNPara_List(SubUnit_BasicNN_List, SubUnit_input_names, 
                                    fldgrn_folder, learner_default_dict):
    
    SubUnit_DefaultBasicNN_List = []
    for basic_nn_idx, nn_type_nn_name in enumerate(SubUnit_BasicNN_List):
        nn_type, nn_name = nn_type_nn_name.split('-')
        
        if basic_nn_idx == 0:
            # the first BasicNN in the SubUnit
            # expander and merger will only be here.
            # this assigments only works for the first iteration
            # let's make a village rule: only E, R, M can be the first. 
            assert nn_type in ['expander', 'reducer', 'merger']
        
            if nn_type == 'expander':
                input_names_nnlvl = SubUnit_input_names 
                assert len(input_names_nnlvl) == 1
                full_recfldgrn = input_names_nnlvl[0]
                default_para = get_Default_ExpanderNNPara(full_recfldgrn, fldgrn_folder)
                
            else:
                default_para = {}
            
        else:
            assert nn_type in ['reducer', 'learner']
            if nn_type == 'learner':
                default_para = learner_default_dict[nn_name] # TODO
            else:
                default_para = {}
        
        SubUnit_DefaultBasicNN_List.append(default_para)    
        
    return SubUnit_DefaultBasicNN_List
   
def generate_BasicNN_Config(nn_type_nn_name, 
                            input_names_nnlvl, 
                            default_nnpara, 
                            embed_size, 
                            process):
    '''
        please notince here, this function is not the final version yet.
    '''
    nn_type, nn_name = nn_type_nn_name.split('-')

    if nn_type == 'expander':
        
        assert len(input_names_nnlvl) == 1
        fld = input_names_nnlvl[0].split('-')[-1]
        # Get the output_name_nnlvl
        output_name_nnlvl = input_names_nnlvl[0].split('Grn')[0]
        
        # Get the input_size and output_size
        input_size = None
        output_size = embed_size 
        
        # Get the postprocess
        postprocess = process
        
        # Derive the para
        vocab_tokenizer = default_nnpara['vocab_tokenizer']
        init = default_nnpara['init']
        para = get_expander_para(nn_name, default_nnpara, 
                                 embed_size, vocab_tokenizer, init, postprocess)

    elif nn_type == 'reducer': 
        
        assert len(input_names_nnlvl) == 1
        fld = input_names_nnlvl[0].split('-')[-1]
        # Get the output_name_nnlvl
        output_name_nnlvl = input_names_nnlvl[0].replace('-' + fld, '@' + fld)

        # Get the para for the NN layer
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
        para = get_reducer_para(nn_name, default_nnpara, input_size, output_size, postprocess)  
        
    elif nn_type == 'merger':
        # generate the output name
        
        assert len(input_names_nnlvl) > 1
        # input_names_nnlvl: ['B-P-EC-A1C@DT', 'B-P-EC-A1C@V']
        
        childflds = [i.split('-')[-1] for i in input_names_nnlvl]
        # childflds: ['A1C@DT', 'A1C@V']
        
        fld_childflds = '&'.join([i.split('@')[-1] for i in childflds])
        # fld_childflds: DT&V
        
        output_prefix = input_names_nnlvl[0].replace('@' + childflds[0].split('@')[-1], '')
        # output_prefix: B-P-EC-A1C
        
        output_name_nnlvl = output_prefix + '-' + fld_childflds
        # output_name_nnlvl: B-P-EC-A1C-DT&V
        
        # Prepare the para for the NN layer
        nn_para = default_nnpara # this will be updated.

        # Get the input_size
        input_size = embed_size
        output_size = embed_size

        # Get the postprocess
        postprocess = process

        # Derive the para
        para = get_merger_para(nn_name, default_nnpara, input_size, output_size, postprocess) 
        
        
    elif nn_type == 'learner':
        # generate the output name
        assert len(input_names_nnlvl) == 1
        output_name_nnlvl = input_names_nnlvl[0] # just the same name as before
        
        # Prepare the para for the NN layer
        nn_para = default_nnpara # this will be updated.

        # Get the input_size
        input_size = embed_size
        output_size = embed_size

        # Get the postprocess
        postprocess = process

        # Derive the para
        para = get_learner_para(nn_name, default_nnpara, input_size, output_size, postprocess)
        
        
    else:
        raise ValueError(f'nn_type {nn_type} is not available yet')

    Basic_Config = {'input_names_nnlvl': input_names_nnlvl, 
                    'output_name_nnlvl': output_name_nnlvl, 
                    f'{nn_type}_para': para}
    
    return Basic_Config


# from fieldnn.configfn.expanderfn import get_expander_para
# from fieldnn.configfn.mergerfn import get_merger_para
# from fieldnn.configfn.reducerfn import get_reducer_para
# from fieldnn.configfn.learnerfn import get_learner_para



def generate_BasicNN_Config(nn_type_nn_name, 
                            input_names_nnlvl, 
                            default_nnpara, 
                            embed_size, 
                            process):
    '''
        please notince here, this function is not the final version yet.
    '''
    nn_type, nn_name = nn_type_nn_name.split('-')

    if nn_type == 'expander':
        
        assert len(input_names_nnlvl) == 1
        # fld = input_names_nnlvl[0].split('-')[-1]
        # Get the output_name_nnlvl
        output_name_nnlvl = input_names_nnlvl[0].split('Grn')[0]
        
        # Get the input_size and output_size
        input_size = None
        output_size = embed_size 
        
        # Get the postprocess
        postprocess = process
        
        # Derive the para
        full_recfldgrn = default_nnpara['full_recfldgrn']
        Info = default_nnpara['Info']
        # para = get_expander_para(nn_name, default_nnpara, embed_size, vocab_tokenizer, init, postprocess)
        para = get_expander_para(full_recfldgrn, Info, embed_size, postprocess)

    elif nn_type == 'reducer': 
        
        assert len(input_names_nnlvl) == 1
        fld = input_names_nnlvl[0].split('-')[-1]
        # Get the output_name_nnlvl
        output_name_nnlvl = input_names_nnlvl[0].replace('-' + fld, '@' + fld)

        # Get the para for the NN layer
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
        para = get_reducer_para(nn_name, default_nnpara, input_size, output_size, postprocess)  
        
    elif nn_type == 'merger':
        # generate the output name
        
        assert len(input_names_nnlvl) > 1
        # input_names_nnlvl: ['B-P-EC-A1C@DT', 'B-P-EC-A1C@V']
        
        childflds = [i.split('-')[-1] for i in input_names_nnlvl]
        # childflds: ['A1C@DT', 'A1C@V']
        
        fld_childflds = '&'.join([i.split('@')[-1] for i in childflds])
        # fld_childflds: DT&V
        
        output_prefix = input_names_nnlvl[0].replace('@' + childflds[0].split('@')[-1], '')
        # output_prefix: B-P-EC-A1C
        
        output_name_nnlvl = output_prefix + '-' + fld_childflds
        # output_name_nnlvl: B-P-EC-A1C-DT&V
        
        # Prepare the para for the NN layer
        nn_para = default_nnpara # this will be updated.

        # Get the input_size
        input_size = embed_size
        output_size = embed_size

        # Get the postprocess
        postprocess = process

        # Derive the para
        para = get_merger_para(nn_name, default_nnpara, input_size, output_size, postprocess) 
        
        
    elif nn_type == 'learner':
        # generate the output name
        assert len(input_names_nnlvl) == 1
        output_name_nnlvl = input_names_nnlvl[0] # just the same name as before
        
        # Prepare the para for the NN layer
        nn_para = default_nnpara # this will be updated.

        # Get the input_size
        input_size = embed_size
        output_size = embed_size

        # Get the postprocess
        postprocess = process

        # Derive the para
        para = get_learner_para(nn_name, default_nnpara, input_size, output_size, postprocess)
        
        
    else:
        raise ValueError(f'nn_type {nn_type} is not available yet')

    Basic_Config = {'input_names_nnlvl': input_names_nnlvl, 
                    'output_name_nnlvl': output_name_nnlvl, 
                    f'{nn_type}_para': para}
    
    return Basic_Config


def get_SubUnit_BasicNN_Config_List(SubUnit_BasicNN_List, 
                                    SubUnit_DefaultBasicNN_List, 
                                    SubUnit_input_names, 
                                    SubUnit_output_name, 
                                    embed_size, 
                                    process, 
                                   ):
    
    
    # TODO: also add the layer_idx in order to deal with the learn_layer_para.
    # This function also needs to be updated. 
    
    SubUnit_BasicNN_Config_List = []
    
    # print('\n\n************** SubUnit_BasicNN_List ****************')
    # print(SubUnit_BasicNN_List)
    # print(SubUnit_input_names)
    # print(SubUnit_output_name)
    for basic_nn_idx, nn_type_nn_name in enumerate(SubUnit_BasicNN_List):

        # Get the input_names_nnlvl
        if basic_nn_idx == 0:
            input_names_nnlvl = SubUnit_input_names # this assigments only works for the first iteration
        else:
            input_names_nnlvl = [output_name_nnlvl]
        
        ##############
        default_nnpara = SubUnit_DefaultBasicNN_List[basic_nn_idx] 
        ##############
        
        
        Basic_Config = generate_BasicNN_Config(nn_type_nn_name, 
                                               input_names_nnlvl, 
                                               default_nnpara, 
                                               embed_size, 
                                               process)
        output_name_nnlvl = Basic_Config['output_name_nnlvl']
        BasicNN_Config = {'nn_type_nn_name': nn_type_nn_name, 'Basic_Config': Basic_Config}
        SubUnit_BasicNN_Config_List.append(BasicNN_Config)
        
        # print('==========================')
        # print(basic_nn_idx, nn_type_nn_name)
        # print(input_names_nnlvl, '<-------- input_names_nnlvl')
        # print(output_name_nnlvl, '<-------- output_name_nnlvl')
        
        
        # also check the input_size of dim and output_size of dim

    final_output_name_nnlvl = output_name_nnlvl

    # if not SubUnit_output_name in final_output_name_nnlvl:
    #     print('xxx errors xxx')
    #     print(final_output_name_nnlvl, '<------- final_output_name_nnlvl')
    #     print(SubUnit_output_name, '<------- SubUnit_output_name')
    #     print('xxx errors xxx')
    assert SubUnit_output_name in final_output_name_nnlvl
    # print('\n************** End SubUnit_BasicNN_List ****************')
    return SubUnit_BasicNN_Config_List



