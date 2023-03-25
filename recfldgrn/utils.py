import itertools
import pandas as pd
import numpy as np
import os

############## <s>
def generate_grain_vocab_info(s):
    value_list = itertools.chain(*s.to_list())
    d = pd.Series(value_list).value_counts().to_dict()
    v2freq = {'_padding': 0}
    if '_missing' in d:
        v2freq['_missing'] = d.pop('_missing') # I should have some prefix here. 
    for k, v in d.items(): v2freq[k] = v

    idx2v = {idx: v for idx, v in enumerate(v2freq.keys())}
    v2idx = {v:k for k, v in idx2v.items()}
    
    return {'idx2v': idx2v, 'v2idx': v2idx, 'v2freq': v2freq}
############## <e>


def get_compressed_df(df, full_recfldgrn_name, prefix_ids):
    df_rec = df
    CompressParent_ID = prefix_ids[-1]
    prefix_ids_new = prefix_ids[:-1]

    df_rec_new = pd.DataFrame(df_rec.groupby(CompressParent_ID).apply(lambda x: x.to_dict('list')).to_list())

    for col in prefix_ids: 
        df_rec_new[col] = df_rec_new[col].apply(lambda x: x[0])
        
    full_recfldgrn_name_new = CompressParent_ID.replace('ID', '') + '-' + full_recfldgrn_name
    df_rec_new = df_rec_new.rename(columns = {full_recfldgrn_name: full_recfldgrn_name_new})
    df_rec_new = df_rec_new[prefix_ids + [full_recfldgrn_name_new]]
    return df_rec_new, full_recfldgrn_name_new, prefix_ids_new



def get_highorder_input_idx(df, recfield2grain, 
                            v2idx, prefix_ids, focal_ids, 
                            Field2Grain_2_get_GrnIdxList_Fn, 
                            Field2Grain_2_get_GrnWgtList_Fn):
    
    RecName, field = recfield2grain.split('-')[0].split('@')
    grain = recfield2grain.split('-')[-1]
    
    suffix = '_idx' if 'Nume' not in recfield2grain else '_wgt'
    if suffix == '_idx':
        get_grn_idx_fn = Field2Grain_2_get_GrnIdxList_Fn[recfield2grain]
        df[recfield2grain + suffix] = df[field].apply(lambda x: get_grn_idx_fn(x, v2idx))
    elif suffix == '_wgt':
        get_grn_wgt_fn = Field2Grain_2_get_GrnWgtList_Fn[recfield2grain]
        df[recfield2grain + suffix] = df[field].apply(lambda x: get_grn_wgt_fn(x))
    else:
        raise ValueError(f'Incorrect suffix "{suffix}"')
    
    
    full_recfldgrn_name_c = recfield2grain + suffix
    top_id = df.columns[0] # need to double check this.
    
    if len(prefix_ids) > 0:
        prefix_ids_c = prefix_ids
        df_Rec = df[prefix_ids_c + [full_recfldgrn_name_c]]
        for i in range(len(prefix_ids_c)):
            df_Rec, full_recfldgrn_name_c, prefix_ids_c = get_compressed_df(df_Rec, full_recfldgrn_name_c, prefix_ids_c)
    else:
        df_Rec = df[focal_ids + [full_recfldgrn_name_c]]

    df_p = df_Rec.reset_index(drop = True)# .set_index(top_id)
    return df_p


def traverse(o, tree_types=(list, tuple), index = None, nest_layer = 100):
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
    len_name = f'{"2".join(layer_parents)}_ln{layer_children}'
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
