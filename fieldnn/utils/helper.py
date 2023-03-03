# Utils.py put it on the other modules
import torch
import torch.nn.functional as F
import numpy as np
import math


# order sequences and restore sequences according to their lenghts
# TODO: test the speed of orderSeq and restoreSeq
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


# padding sequences 
def pad_packed_sequence(var_data, batch_sizes, batch_first=True, padding_value=0.0):

    # var_data, batch_sizes = sequence
    max_batch_size = int(batch_sizes[0])
    max_seq_length = batch_sizes.size(0)

    output = var_data.data.new(max_seq_length, max_batch_size, *var_data.size()[1:]).fill_(padding_value)
    lengths = []
    data_offset = 0
    prev_batch_size = int(batch_sizes[0])
    prev_i = 0
    for i, batch_size in enumerate(batch_sizes.tolist() + [0]):
        if batch_size != prev_batch_size:
            l = prev_batch_size * (i - prev_i)
            tmp = var_data[data_offset:data_offset + l]
            output[prev_i:i, :prev_batch_size] = tmp.view(i - prev_i, prev_batch_size, *tmp.size()[1:])
            data_offset += l
            prev_i = i
        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size
    lengths.reverse()
    if batch_first:
        output = output.transpose(0, 1)
    return output, torch.LongTensor(lengths)


# ------------------------ skip
def change_feat_location(x):
    return torch.transpose(x, -1, 1).contiguous()



# reordering tensor to simulate the backward lstm or gpt.
def reverse_tensor(input_tensor, leng_st):
    # only leng_st is valid
    input_tensor_reversed = torch.zeros_like(input_tensor)
    # leng_st: based on batch
    for sent_idx, leng_sent in enumerate(leng_st):
        # print(origin_tensor)
        input_tensor_reversed[sent_idx][:leng_sent] = torch.flip(input_tensor[sent_idx][:leng_sent], (0,))
    return input_tensor_reversed

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s



