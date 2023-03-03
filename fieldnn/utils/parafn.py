import pandas as pd

def get_fullname_from_inputs(fullname_list):
    names = [i.split('-')[-1] for i in fullname_list]
    
    prefix = ['-'.join(i.split('-')[:-1]) for i in fullname_list][0]
    
    table_row_indicator = list(set([i.count(':') for i in names]))[0]
    
    if table_row_indicator == 0:
        fullname = f'{prefix}-{"&".join(names)}'
    elif table_row_indicator >= 1:
        table_name = [':'.join(i.split(':')[:-1]) for i in names][0]
        columns =  [i.split(':')[-1] for i in names]
        fullname = f'{prefix}-{table_name}-{"&".join(columns)}'
    else:
        raise ValueError(f'"table_row_indicator" {table_row_indicator} if not correct')
    return fullname

def generate_psn_embed_para(layername, embed_size):
    
    if 'Grn' not in layername: 
        vocab_size = 100
    else:
        vocab_size = 512
        
    embed_para = {'embedding_size': embed_size,
                  'init': 'random', 
                  'input_size': vocab_size + 1 }
    return embed_para



def get_expander_para(fullname, 
                      nn_name, nn_para, 
                      embed_size, 
                      Ignore_PSN_Layers,
                      postprocess
                     ):
    
    expander_layer_para = {}
    #(0) Input size, output size
    expander_layer_para['input_size'] = None
    expander_layer_para['output_size'] = embed_size

    
    #(1) Main NN
    if nn_name.lower() == 'embedding':
        para = {'embedding_size': embed_size,
                'init': 'random', 
                'input_size': 5000} # will be updated in input_size
        # print(nnpara)
        assert 'input_size' in nn_para
        for k, v in nn_para.items():
            if k in para: para[k] = v
    else:
        raise ValueError(f'The NN "{nn_name}" is not available yet')
    expander_layer_para[fullname] = nn_name, para
    
    # (2) PSN Embed
    # name = fullname.split('-')[-1]
    # Layer2Idx = {v:idx for idx, v in enumerate(fullname.split('-'))}
    # psn_layers = list(reversed([i for i in Layer2Idx if i not in Ignore_PSN_Layers]))
    # expander_layer_para['psn_layers'] = psn_layers
    
    
    # (3) Post Process
    expander_layer_para['postprocess'] = postprocess
    
    expander_layer_para['Ignore_PSN_Layers'] = Ignore_PSN_Layers
    
    return expander_layer_para



def get_learner_para(fullname, 
                     nn_name, nn_para, 
                     input_size, output_size, 
                     Ignore_PSN_Layers, 
                     embedprocess,
                     postprocess):
    learner_layer_para = {}
    # (0) Size
    learner_layer_para['input_size'] = input_size
    learner_layer_para['output_size'] = output_size
    
    
    # (1) NN
    if nn_name.lower() == 'tfm':
        assert input_size == output_size
        para =  {'input_size': output_size,
                 'output_size': output_size,
                 'nhead': 8,
                 'num_encoder_layers': 6,
                 'num_decoder_layers': 0,
                 'dim_feedforward': 2048,
                 'tfm_dropout': 0.1,
                 'tfm_activation': 'relu'}
        for k, v in nn_para.items():
            if k in para: para[k] = v


    elif nn_name.lower() == 'linear':
        para =  {'input_size': input_size,
                 'output_size': output_size}
        for k, v in nn_para.items():
            if k in para: para[k] = v
            
    else:
        raise ValueError(f'The NN "{nn_name}" is not available yet')
    
    learner_layer_para[fullname] = nn_name, para
    
    # (2) PSN Embed
    if nn_name.lower() == 'linear':
        learner_layer_para['psn_layers'] = {}
        learner_layer_para['embedprocess'] = {}
        learner_layer_para['Ignore_PSN_Layers'] = {}
    else:
        name = fullname.split('-')[-1]
        Layer2Idx = {v:idx for idx, v in enumerate(fullname.split('-'))}
        psn_layers = list(reversed([i for i in Layer2Idx if i not in Ignore_PSN_Layers]))
        learner_layer_para['psn_layers'] = psn_layers
        learner_layer_para['embedprocess'] = embedprocess
        learner_layer_para['Ignore_PSN_Layers'] = Ignore_PSN_Layers
        
    # (3) Post Process
    learner_layer_para['postprocess'] = postprocess

    return learner_layer_para


def get_merger_para(fullname, nn_name, nn_para, input_size, output_size, postprocess):
    merger_layer_para = {}
    
    merger_layer_para[fullname] = nn_name, nn_para
    
    merger_layer_para['input_size'] = input_size
    merger_layer_para['output_size'] = output_size

    merger_layer_para['postprocess'] = postprocess
    
    return merger_layer_para



def get_reducer_para(fullname, nn_name, nn_para, input_size, output_size, postprocess):
    reducer_layer_para = {}
    
    reducer_layer_para[fullname] = nn_name, nn_para
    
    reducer_layer_para['input_size'] = input_size
    reducer_layer_para['output_size'] = output_size

    reducer_layer_para['postprocess'] = postprocess
    
    return reducer_layer_para




# from fieldnn.utils.parafn import get_expander_para, get_learner_para, get_reducer_para, get_merger_para
# from fieldnn.utils.parafn import get_fullname_from_inputs
# from .parafn import get_expander_para, get_learner_para, get_reducer_para, get_merger_para
# from .parafn import get_fullname_from_inputs

# default_learner_para  = {
#     'nn_name': 'TFM',
#     'nn_para': {'num_encoder_layers': 6}
# }

# default_reducer_para  = {
#     'nn_name': 'Max',
# }


def get_EL_sublayer_para_list(input_fullname_list, 
                              FLD_2_VOCABSIZE, 
                              embed_size,
                              default_learner_para, 
                              expander_process, 
                              default_process, 
                              Ignore_PSN_Layers):
    # print(input_fullname_list)
    assert len(input_fullname_list) == 1
    fullname = input_fullname_list[0]
    output_fullname = fullname.replace('Grn', '')
    ###########
    nn_name = 'Embedding'
    vocab_size = FLD_2_VOCABSIZE[fullname]
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
    learner_layer_para  = get_learner_para(output_fullname, 
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
    output_fullname = '-'.join(fullname.split('-')[:-1]) + ':' + fullname.split('-')[-1]
    
    nn_name = default_learner_para['nn_name']
    nn_para = default_learner_para['nn_para']
    input_size = embed_size
    output_size = embed_size
    embedprocess = default_process
    postprocess = default_process
    ###########
    learner_layer_para  = get_learner_para(output_fullname, 
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
    fullname = '^'.join(input_fullname_list) # input of M
    # fullname = get_fullname_from_inputs(input_fullname_list)

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
    # print(input_fullname_list, '<----get_ML_sublayer_para_list') 
    fullname = get_fullname_from_inputs(input_fullname_list) # input of L
    # print(fullname, '<----get_ML_sublayer_para_list') 
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

    

def process_pipeline_name(sublayer_name, FLD_2_VOCABSIZE, embed_size, 
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
        para_dict = get_EL_sublayer_para_list(input_fullname_list, 
                                              FLD_2_VOCABSIZE, 
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