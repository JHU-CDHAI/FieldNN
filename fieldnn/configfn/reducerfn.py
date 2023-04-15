
def get_reducer_para(nn_name, nn_para, input_size, output_size, postprocess):
    reducer_layer_para = {}
    reducer_layer_para['nn_type'] = 'reducer'
    reducer_layer_para['nn_name'] = nn_name
    reducer_layer_para['nn_para'] = nn_para
    reducer_layer_para['input_size'] = input_size
    reducer_layer_para['output_size'] = output_size
    reducer_layer_para['postprocess'] = postprocess
    return reducer_layer_para