import itertools
import pandas as pd

############## <s>
def generate_grain_vocab_info(s):
    value_list = itertools.chain(*s.to_list())
    d = pd.Series(value_list).value_counts().to_dict()
    v2freq = {}
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
