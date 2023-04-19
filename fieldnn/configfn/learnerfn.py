#######################
# psn_embedprocess = {
#     # 'activator': 'gelu',
#     # 'dropout': {'p': 0.5, 'inplace': False}, # maybe the dropout?
#     # 'layernorm': {'eps': 1e-05, 'elementwise_affine': True}
# }

# embed_size = 128
# default_tfm_para = {'psn_max': 512, 
#                     'psn_embedprocess': psn_embedprocess}
#######################

def get_tfm_para(input_size, default_tfm_para):
    assert 'input_size' not in default_tfm_para
    
    tfm_para =  {'input_size': input_size,
                 'output_size': input_size,
                 'nhead': 8,
                 'num_encoder_layers': 6,
                 'num_decoder_layers': 0,
                 'dim_feedforward': 2048,
                 'tfm_dropout': 0.1,
                 'tfm_activation': 'relu',
                 'psn_max': 512, 
                 'psn_embedprocess': {}
                }
    
    for k, v in default_tfm_para.items(): tfm_para[k] = v
    return tfm_para

# default_linear_para = {'initrange': 0.1}

def get_linear_para(input_size, output_size, default_linear_para):
    linear_para =  {'input_size': input_size,
                    'output_size': output_size}
    
    for k, v in default_linear_para.items(): linear_para[k] = v
    return linear_para


def get_learner_para(nn_name, default_nn_para, 
                     input_size, output_size, 
                     postprocess):
    
    learner_para = {}
    
    # (1) 
    learner_para['nn_type'] = 'learner'  
    learner_para['nn_name'] = nn_name # TFM or linear
    
    # (2)
    if nn_name.lower() == 'tfm':
        para = get_tfm_para(input_size, default_nn_para)
    elif nn_name.lower() == 'linear':
        para = get_linear_para(input_size, output_size, default_nn_para)
    else:
        raise ValueError(f'The NN "{nn_name}" is not available yet')
    learner_para['nn_para'] = para
    
    # (3)
    learner_para['input_size'] = input_size
    learner_para['output_size'] = output_size
    

    # (4) Post Process
    learner_para['postprocess'] = postprocess
    
    return learner_para