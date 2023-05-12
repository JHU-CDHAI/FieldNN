import torch
import numpy as np


def traverse(o, tree_types=(list, tuple, np.ndarray), index = None, nest_layer = 100):
    if isinstance(o, tree_types) and nest_layer > 0:
        for idx, value in enumerate(o):
            new_index = index + [idx] if type(index) == list else [idx]
            for subvalue in traverse(value, tree_types, new_index, nest_layer - 1):
                yield subvalue
    else:
        if not isinstance(o, tree_types): 
            length = None
        else:
            length = len(o)
        yield index, length, o



def convert_relational_list_to_numpy(values_list, new_full_recfldgrn, suffix):
    o = values_list
    layer_num = len(new_full_recfldgrn.split('-'))
    layers = new_full_recfldgrn.replace(suffix, '').split('-')
    # L = [len(values_list)] 

    D = {}

    # (1) from first layer: 0
    idx = 0
    layer_parents = layers[:idx + 1]
    layer_children = layers[idx + 1]
    len_name = f'{"-".join(layer_parents)}@ln{layer_children}'
    len_np = np.array(len(values_list))
    len_shapes = [len_np.max()] # from layer 0, prepare for layer 1. 
    D[len_name] = len_np

    # (2) from 1 - last one layers
    for idx in range(1, layer_num - 1):
        output = list(traverse(o, nest_layer = idx))
        # print(output)
        # data = np.zeros(L)
        # print('\n\n')
        # print(idx)
        # print(output)

        layer_parents = layers[:idx + 1]
        layer_children = layers[idx + 1]
        len_name = f'{"-".join(layer_parents)}_ln{layer_children}'
        # print(len_name)

        locidx  = [i[0] for i in output]
        length = [i[1] for i in output]
        # values = [i[2] for i in output]

        len_np = np.zeros(len_shapes).astype(int)
        # print(len_np.shape, '<---- len_np.shape')
        for locidx, length, _ in output:
            len_np[tuple(locidx)] = int(length)
        # print(len_np)
        len_shapes.append(len_np.max())
        # print(len_shapes, '<---- next len_np.shape')
        # print(length)
        # print()
        D[len_name] = len_np

    # (3) for the data
    idx = layer_num
    name = new_full_recfldgrn
    data = np.zeros(len_shapes) # don't convert it to int for now. 
    output = list(traverse(o, nest_layer = idx))
    for locidx, _, value in output:
        data[tuple(locidx)] = value
        # print(locidx, value)
        # print(data[tuple(locidx)])# = value
    data.shape
    D[name] = data.astype(int)
    
    return D

def orderSeq(seq_unordered, leng_unordered):
    # leng_unordered is a tensor
    # seq_unordered is a numpy
    leng_ordered, seq_index = leng_unordered.sort(descending=True) 
    _, reverse_index = seq_index.sort()
    leng_ordered = leng_ordered[leng_ordered>0]
    seq_index    = seq_index[:len(leng_ordered)]
    seq_ordered  = seq_unordered[seq_index.cpu()]
    return seq_ordered, leng_ordered, reverse_index


def restoreSeq(seq_ordered, reverse_index):
    # shape = list(seq_ordered.shape)
    data_type = seq_ordered.type()
    shape = list(seq_ordered.shape)
    shape[0] = len(reverse_index) - shape[0]
    t = torch.cat([seq_ordered, torch.zeros(shape).type(data_type)])
    seq_restored = t[reverse_index]
    return seq_restored

def get_Layer2Holder(fullname, holder, Ignore_PSN_Layers = ['B', 'P']):
    # holder = holder
    d = {}
    for layername in list(reversed(fullname.split('-'))): # from 2 to -
        if layername in Ignore_PSN_Layers: continue
        leng_mask = holder == 0
        leng = (leng_mask == 0).sum(-1)
        psn_idx = (leng_mask == False).cumsum(-1).masked_fill(leng_mask, 0)
        d[layername] = {'holder': holder, 
                        'leng_mask': leng_mask, 
                        'leng': leng, 
                        'psn_idx': psn_idx}
        # d[layername] = holder, psn_idx
        holder = leng
    Layer2Hoder = d
    return Layer2Hoder


def align_psn_idx(source_layer, current_layer, Layer2Idx, Layer2Holder):
    if source_layer == current_layer:
        psn_idx = Layer2Holder[current_layer]['psn_idx']
        return psn_idx
    else:
        source_psn_idx = Layer2Holder[source_layer]['psn_idx']
        current_leng_mask = Layer2Holder[current_layer]['leng_mask']
        gaps = Layer2Idx[current_layer] - Layer2Idx[source_layer]
        # print(gaps)
        # print(layername)
        # print(prev_info.shape)
        # print(leng_mask.shape)
        # print(leng.shape)
        # print(psn_idx.shape)
        shape0 = list(source_psn_idx.shape) + [1] * gaps
        shape1 = current_leng_mask.shape
        psn_idx = source_psn_idx.view(*shape0).expand(shape1).masked_fill(current_leng_mask, 0)
        # print(cpsn_idx.shape)
        return psn_idx