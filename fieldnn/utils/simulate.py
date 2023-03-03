import numpy as np
from .layerfn import traverse

def get_next_info(layer, current_info, max_list):
    current_max = max_list[layer]
    next_max = max_list[layer + 1]
    # to do: get next_info
    next_info = np.zeros(list(np.array(current_info).shape) + [current_max]).astype(int)
    # print(next_info.shape)
    for element in list(traverse(current_info, nest_layer = layer)):
        idx, leng, value = element
        # print(next_info[tuple(idx)])
        next_info[tuple(idx)][:value] = np.random.randint(1, next_max + 1, value)

    # print(next_info)
    return next_info


def get_simulated_tensor_from_fldname(fld_name, B_lenP, B2P_lnEC, prefix_layers_num, vocab_size):
    layers_num = len(fld_name.split('-'))
    # print(layers_num)

    # max_list = df['max'].astype(int).to_list()
    prefix = [np.array(B_lenP).max(), max(B2P_lnEC)] 
    max_list = [np.array(B_lenP).max(), max(B2P_lnEC)] + list(np.random.randint(1, 10, layers_num - len(prefix) - 1)) + [vocab_size -1]
    # print(max_list)

    init_info = np.array(B2P_lnEC)
    
    for layer_idx in range(prefix_layers_num - 1, layers_num - 1):
        print(layer_idx)
        current_info = init_info
        next_info = get_next_info(layer_idx, current_info, max_list)
        # print(next_info)
        init_info = next_info

        print(layer_idx, '-->', current_info.shape)
        print(layer_idx + 1, '-->', next_info.shape)


    fld_tensor_idx = next_info
    # print(fld_tensor.shape)
    return fld_tensor_idx



