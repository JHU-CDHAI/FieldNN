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



    