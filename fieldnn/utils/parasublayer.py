# from fieldnn.utils.parafn import get_expander_para, get_learner_para, get_reducer_para, get_merger_para
# from fieldnn.utils.parafn import get_fullname_from_inputs

from .parafn import get_expander_para, get_learner_para, get_reducer_para, get_merger_para
from .parafn import get_fullname_from_inputs

default_learner_para  = {
    'nn_name': 'TFM',
    'nn_para': {'num_encoder_layers': 6}
}

default_reducer_para  = {
    'nn_name': 'Max',
}

def get_EL_sublyaer_para_list(input_fullname_list, 
                              fld_to_vocabsize, 
                              embed_size,
                              default_learner_para, 
                              expander_process, 
                              default_process, 
                              Ignore_PSN_Layers):
    print(input_fullname_list)
    assert len(input_fullname_list) == 1
    fullname = input_fullname_list[0]

    ###########
    nn_name = 'Embedding'
    vocab_size = fld_to_vocabsize[fullname]
    nn_para = {'input_size': vocab_size}
    postprocess = expander_process
    ###########
    expander_layer_para = get_expander_para(fullname, nn_name, nn_para, embed_size, 
                                            Ignore_PSN_Layers, 
                                            postprocess
                                           )
    # print(expander_layer_para)
    ###########
    nn_name = default_learner_para['nn_name']
    nn_para = default_learner_para['nn_para']
    input_size = embed_size
    output_size = embed_size
    embedprocess = default_process
    postprocess = default_process
    ###########
    learner_layer_para  = get_learner_para(fullname, 
                                           nn_name, nn_para, 
                                           input_size, output_size, 
                                           Ignore_PSN_Layers, 
                                           embedprocess, postprocess
                                          )
    # print(learner_layer_para)
    para_dict = {'Expander': expander_layer_para, 'Learner': learner_layer_para}
    return para_dict



def get_RL_sublayer_para_list(input_fullname_list, 
                              embed_size, 
                              default_learner_para,
                              default_reducer_para,
                              default_process,
                              Ignore_PSN_Layers):
    assert len(input_fullname_list) == 1
    fullname = input_fullname_list[0]

    #########
    nn_name = default_reducer_para['nn_name'] # 'Max'
    nn_para = {}
    input_size = embed_size
    output_size = embed_size if nn_name != 'concat' else embed_size * 3
    postprocess = default_process
    #########

    reducer_layer_para = get_reducer_para(fullname, nn_name, nn_para, input_size, output_size, postprocess)
    # print(reducer_layer_para)

    ###########
    nn_name = default_learner_para['nn_name']
    nn_para = default_learner_para['nn_para']
    input_size = embed_size
    output_size = embed_size
    embedprocess = default_process
    postprocess = default_process
    ###########
    learner_layer_para  = get_learner_para(fullname, 
                                           nn_name, nn_para, 
                                           input_size, output_size, 
                                           Ignore_PSN_Layers, 
                                           embedprocess, postprocess
                                          )
    # print(learner_layer_para)
    para_dict = {'Reducer': reducer_layer_para, 'Learner': learner_layer_para}
    return para_dict
    

def get_ML_sublayer_para_list(input_fullname_list,
                              embed_size, 
                              default_learner_para,
                              default_process, 
                              Ignore_PSN_Layers):
    assert len(input_fullname_list) > 1
    fullname = get_fullname_from_inputs(input_fullname_list)

    #########
    nn_name = 'Merger'
    nn_para = {}
    input_size = embed_size
    output_size = embed_size
    postprocess = default_process
    #########

    merger_layer_para = get_merger_para(fullname, nn_name, nn_para, input_size, output_size, postprocess)
    # print(merger_layer_para)

    ###########
    nn_name = default_learner_para['nn_name']
    nn_para = default_learner_para['nn_para']
    input_size = embed_size
    output_size = embed_size
    embedprocess = default_process
    postprocess = default_process
    ###########
    learner_layer_para  = get_learner_para(fullname, 
                                           nn_name, nn_para, 
                                           input_size, output_size, 
                                           Ignore_PSN_Layers, 
                                           embedprocess, postprocess
                                          )
    # print(merger_layer_para)
    para_dict = {'Merger': merger_layer_para, 'Learner': learner_layer_para}
    return para_dict

    

def process_sublayer_name(sublayer_name, fld_to_vocabsize, embed_size, 
                          default_learner_para,  default_reducer_para,
                          expander_process, default_process, Ignore_PSN_Layers):
    # print('   ----> (sublayer name)', sublayer_name)   
    sublayer_type, input_output = sublayer_name.split('**')
    input_fullnames, output_fullname = input_output.split('=>')
    input_fullname_list = input_fullnames.split('^')
    # print('       ----> (+)', sublayer_type)
    # print('       ----> (+)', input_fullname_list)
    # print('       ----> (+)', output_fullname)

    if 'EL' == sublayer_type:
        para_dict = get_EL_sublyaer_para_list(input_fullname_list, 
                                              fld_to_vocabsize, 
                                              embed_size,
                                              default_learner_para, 
                                              expander_process, 
                                              default_process, 
                                              Ignore_PSN_Layers)
    elif 'RL' == sublayer_type:
        para_dict = get_RL_sublayer_para_list(input_fullname_list, 
                                              embed_size, 
                                              default_learner_para,
                                              default_reducer_para,
                                              default_process,
                                              Ignore_PSN_Layers)
    elif 'ML' == sublayer_type:
        para_dict = get_ML_sublayer_para_list(input_fullname_list,
                                              embed_size, 
                                              default_learner_para,
                                              default_process, 
                                              Ignore_PSN_Layers)
    else:
        raise ValueError(f'The sublayer type {sublayer_type} is not available')
        
        
    return input_fullname_list, output_fullname, para_dict