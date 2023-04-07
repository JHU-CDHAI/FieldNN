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



def update_df_dataflow(df):
    df_dataflow = df.copy()


    for layer_idx in df_dataflow.columns:
        # (1) first deal with the fld in each layer
        s = df_dataflow[layer_idx]
        current_fullrfg_list = s[-s.isna()].to_list()
        fld_with_at = [i for i in current_fullrfg_list if '@' in i.split('-')[-1]]
        fld_with_at_to_keep = []
        fld_with_at_to_check = [i for i in fld_with_at if i not in fld_with_at_to_keep]
        
        while len(fld_with_at_to_check) > 0:
            for index in df_dataflow.index:
                full_recfldgrn = df_dataflow.loc[index, layer_idx]
                if pd.isna(full_recfldgrn): continue
                if full_recfldgrn == 'ToFill': continue
                curfld = full_recfldgrn.split('-')[-1]
                prefix = '-'.join(full_recfldgrn.split('-')[:-1])
                headfld, tailfld = get_curfld_recinfo(curfld)
                if headfld != None:
                    full_recfldgrn_new = prefix + '-' + headfld
                    same_rfg_new = [i for i in current_fullrfg_list if full_recfldgrn_new in i]
                    if len(same_rfg_new) == 1:
                        print(full_recfldgrn)
                        df_dataflow.loc[index, layer_idx] = full_recfldgrn_new
            
            # update s for one iteration
            s = df_dataflow[layer_idx]
            current_fullrfg_list = s[-s.isna()].to_list()
            fld_with_at = [i for i in current_fullrfg_list if '@' in i.split('-')[-1]]
            fld_with_at_to_check = [i for i in fld_with_at if i not in fld_with_at_to_keep]
            
        print(f'Finish drop single @ fld in layer {layer_idx}')
        
            
            
        s = df_dataflow[layer_idx]
        current_fullrfg_list = s[-s.isna()].to_list()
        fld_with_at = [i for i in current_fullrfg_list if '@' in i.split('-')[-1]]
        fld_with_at_to_check = [i for i in fld_with_at if i not in fld_with_at_to_keep]
        
        print('Current fld_with_at_to_check is:', fld_with_at_to_check)
        
        
        while len(fld_with_at_to_check) > 0:
            for index in df_dataflow.index:
                full_recfldgrn = df_dataflow.loc[index, layer_idx]
                if pd.isna(full_recfldgrn): continue
                if full_recfldgrn == 'ToFill': continue
                curfld = full_recfldgrn.split('-')[-1]
                prefix = '-'.join(full_recfldgrn.split('-')[:-1])
                headfld, tailfld = get_curfld_recinfo(curfld)
                if headfld != None:
                    full_recfldgrn_new = prefix + '-' + headfld
                    same_rfg_new = [i for i in current_fullrfg_list if full_recfldgrn_new in i]
                    if len(same_rfg_new) > 1:
                        tailfld_merged = '&'.join([i.replace(full_recfldgrn_new +'@', '') for i in same_rfg_new])
                        new_index = '(Merge)' + headfld + '@' + tailfld_merged
                        # new_fldgrn = full_recfldgrn_new
                        df_dataflow.loc[new_index, layer_idx] = full_recfldgrn_new
                        fld_with_at_to_keep.extend(same_rfg_new)
                        fld_with_at_to_keep.append(full_recfldgrn_new)
                        # print(full_recfldgrn_new)
                        # print(same_rfg_new)
                        break
        
        if layer_idx == df_dataflow.columns[-1]: continue
        # then update the next layer information
        for index in df_dataflow.index:
            last = df_dataflow.loc[index, layer_idx]
            if pd.isna(last): continue
            if '@' in last.split('-')[-1]: continue
            full_recfldgrn_next = '-'.join(last.split('-')[:-1]) + '@' + last.split('-')[- 1]
            df_dataflow.loc[index, layer_idx - 1] = full_recfldgrn_next
            
    return df_dataflow




    


def update_df_dataflow(df):
    df_dataflow = df.copy()


    for layer_idx in df_dataflow.columns:
        
        # (1) first deal with the fld in each layer
        s = df_dataflow[layer_idx]
        current_fullrfg_list = s[-s.isna()].to_list()
        fld_with_at = [i for i in current_fullrfg_list if '@' in i.split('-')[-1]]
        fld_with_at_to_keep = []
        fld_with_at_to_check = [i for i in fld_with_at if i not in fld_with_at_to_keep]
        
        while len(fld_with_at_to_check) > 0:
            for index in df_dataflow.index:
                full_recfldgrn = df_dataflow.loc[index, layer_idx]
                if pd.isna(full_recfldgrn): continue
                if full_recfldgrn == 'ToFill': continue
                curfld = full_recfldgrn.split('-')[-1]
                prefix = '-'.join(full_recfldgrn.split('-')[:-1])
                headfld, tailfld = get_curfld_recinfo(curfld)
                if headfld != None:

                    full_recfldgrn_new = prefix + '-' + headfld
                    
                    
                    same_rfg_new = [i for i in current_fullrfg_list if full_recfldgrn_new in i]
                    
                    if len(same_rfg_new) == 1:
                        print(full_recfldgrn)
                        df_dataflow.loc[index, layer_idx] = full_recfldgrn_new


                    elif len(same_rfg_new) > 1:
                        tailfld_merged = '&'.join([i.replace(full_recfldgrn_new +'@', '') for i in same_rfg_new])
                        new_index = '(Merge)' + headfld + '@' + tailfld_merged
                        # new_fldgrn = full_recfldgrn_new
                        df_dataflow.loc[new_index, layer_idx] = full_recfldgrn_new
                        fld_with_at_to_keep.extend(same_rfg_new)
                        fld_with_at_to_keep.append(full_recfldgrn_new)
                        # print(full_recfldgrn_new)
                        # print(same_rfg_new)
                        # break
                    
        
        if layer_idx == df_dataflow.columns[-1]: continue
        # then update the next layer information
        for index in df_dataflow.index:
            last = df_dataflow.loc[index, layer_idx]
            if pd.isna(last): continue
            if '@' in last.split('-')[-1]: continue
            full_recfldgrn_next = '-'.join(last.split('-')[:-1]) + '@' + last.split('-')[- 1]
            df_dataflow.loc[index, layer_idx - 1] = full_recfldgrn_next
            
    return df_dataflow