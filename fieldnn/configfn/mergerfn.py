def get_merger_para(nn_name, nn_para, input_size, output_size, postprocess):
    para = {}
    para['nn_type'] = 'merger'
    para['nn_name'] = nn_name
    para['nn_para'] = nn_para
    para['input_size'] = input_size
    para['output_size'] = output_size
    para['postprocess'] = postprocess
    return para