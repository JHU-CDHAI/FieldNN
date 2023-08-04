import numpy as np
import pandas as pd
from ..utils.datanamefn import get_curfld_recinfo

def get_Repr_dataflow_table(full_recfldgrn_list):
    '''
        input: a list of full_recfldgrn
        output: the dataframe that shows the data flow from the grain-embedding-tensor to final feature vector.
    '''
    max_layer = max([len(i.split('-')) for i in full_recfldgrn_list])
    L = []
    for full_recfldgrn in full_recfldgrn_list:
        d = {}
        recfldgrn = '-'.join(full_recfldgrn.split('-')[-2:])
        current_layer_id = len(full_recfldgrn.split('-'))
        d['recfldgrn'] = recfldgrn
        for layer_id in range(max_layer, 1, -1):
            if layer_id > current_layer_id:
                d[layer_id] = np.nan
            elif layer_id == current_layer_id:
                d[layer_id] = full_recfldgrn
            else:
                d[layer_id] = np.nan
        L.append(d)
    df_dataflow = pd.DataFrame(L).set_index('recfldgrn')
    return df_dataflow


def update_df_Repr_dataflow(df):
    
    df_dataflow = df.copy()

    for layer_idx in df_dataflow.columns:
        
        # from layerid: full_recfldgrn to layerid-1: pfx_rec if pfx_rec is unique.
        for index in df_dataflow.index:
            full_recfldgrn = df_dataflow.loc[index, layer_idx] 
            if pd.isna(full_recfldgrn): continue
            # if full_recfldgrn == 'ToFill': continue
            pfx_rec = '-'.join(full_recfldgrn.split('-')[:-1])
            cur_rec = full_recfldgrn.split('-')[-1]
            s = df_dataflow.loc[df_dataflow.index != index, layer_idx]
            current_fullrfg_list = s[-s.isna()].to_list()
            current_pfxrec_list = ['-'.join(i.split('-')[:-1]) for i in current_fullrfg_list]

            # print(current_pfxrec_list)
            if pfx_rec not in current_pfxrec_list:
                pfx_rec_list = pfx_rec.split('-')
                pfx_rec_list[-1] = pfx_rec_list[-1].replace('@', '')
                output_recfldgrn = '-'.join(pfx_rec_list)
                df_dataflow.loc[index, layer_idx - 1] = output_recfldgrn
        
        # create new merge index.
        s = df_dataflow.loc[:, layer_idx]
        current_fullrfg_list = s[-s.isna()].to_list()
        if len(current_fullrfg_list) == 0: continue
        current_pfxrec_list = [{'i': i, 'j': '-'.join(i.split('-')[:-1])} for i in current_fullrfg_list]
        dfx = pd.DataFrame(current_pfxrec_list)
        s = dfx.groupby('j').apply(lambda x: x['i'].to_list()).to_dict()
        s = {k: v for k, v in s.items() if len(v) > 1}
        # print(s)
        for pfx_rec, fullrec_list in s.items():
            
            full_recfldgrn_new = pfx_rec + '-' + '&'.join([i.split('-')[-1] for i in fullrec_list])
            new_index = '(Merge)' + full_recfldgrn_new
            l = list(df_dataflow.index)
            
            # print(fullrec_list, '<----')
            # print(df_dataflow[layer_idx].to_list(), '<----')
            
            selected_index_list = df_dataflow[df_dataflow[layer_idx].isin(fullrec_list)].index
            # print(selected_index_list)
            loc = max([l.index(i) for i in selected_index_list])
            l.insert(loc + 1, new_index)
            
            df_dataflow.loc[new_index, layer_idx] = full_recfldgrn_new
            df_dataflow.loc[new_index, layer_idx-1] = pfx_rec
            df_dataflow = df_dataflow.reindex(l)
            
            
    df_dataflow = df_dataflow.iloc[:, :-1]
            
    return df_dataflow


def update_df_Repr_dataflow_completename(df):
    df_dataflow = df.copy()
    L = []
    for recfldgrn, row in df_dataflow.iterrows():
        # print(recfldgrn)
        new_row = {}
        new_row['recfldgrn'] = recfldgrn
        full_recfldgrn = [i for i in row.values if not pd.isna(i)][0]
        curlayer_idx = len(full_recfldgrn.split('-'))
        for layer_idx, full_recfldgrn in row.items():
            if curlayer_idx < layer_idx:
                new_row[layer_idx] = np.nan
            elif curlayer_idx == layer_idx:
                new_row[layer_idx] = full_recfldgrn
            elif curlayer_idx > layer_idx:
                last = new_row[layer_idx+1]
                output_fullname = '-'.join(last.split('-')[:-1]) + '@' + last.split('-')[- 1]
                new_row[layer_idx] = output_fullname
            else:
                raise ValueError('wrong information')
        L.append(new_row)
    df_dataflow_filled = pd.DataFrame(L).set_index('recfldgrn')
    return df_dataflow_filled



def get_Repr_SubUnit_List(df_dataflow, 
                          default_R_subunit_name = 'RL', 
                          default_MR_subunit_name = 'ML', # or 'MLRL'
                         ):
    layeridx_list = list(df_dataflow.columns)
    
    SubUnit_List = []
    for idx in range(len(layeridx_list) - 1):
        A_layerid = layeridx_list[idx]
        B_layerid = layeridx_list[idx + 1]
        # print(A_layerid, B_layerid)

        A_tensors = df_dataflow[A_layerid]
        A_tensors = A_tensors[-A_tensors.isna()].to_dict()

        B_tensors = df_dataflow[B_layerid]
        B_tensors = B_tensors[-B_tensors.isna()].to_dict()
        
        
        # Deal with the Merge First.
        # from B tensor: potential there are some Merger NNs. 
        # print('B', B_tensors)
        # check whether these is a '(Merger') in the key
        # merger_tensors = [v for k, v in B_tensors.items() if '(Merge)' in k and '@' not in v]
        # merger_tensors = [v for k, v in B_tensors.items() if '(Merge)' in k]
        # print('B-merger_tensors', merger_tensors)
        
        merger_tensors = [tensor for index, tensor in A_tensors.items() 
                          if '(Merge)' in index and '&' in tensor]
        
        # print(merger_tensors, '<----- merger_tensors', A_layerid)

        for output_tensor in merger_tensors:
            # print(B_tensors, '<---- B_tensor')
            input_tensors = [i for k, i in A_tensors.items() 
                             if '-'.join(output_tensor.split('-')[:-1]) == '-'.join(i.split('-')[:-1])]
            input_tensors = [i for i in input_tensors if i != output_tensor]
            # print(output_tensor, ':', input_tensors)

            d = {}
            d['SubUnitName'] = default_MR_subunit_name
            d['input_names'] = input_tensors
            d['output_name'] = output_tensor
            
            d['input_layerid'] = A_layerid
            d['output_layerid'] = A_layerid
        
            SubUnit_List.append(d)
        

        # print(f'\nFrom Layer {A_layerid} to {B_layerid}:')

        # from A tensor to B tensor, we have the Reducer NNs.
        # also notice that some 

        # print('A', A_tensors)
        for k in A_tensors:
            # pass the merged tensors. 
            if k not in B_tensors: continue

            input_name = A_tensors[k]
            output_name = B_tensors[k]

            if pd.isna(output_name) == True: continue # pass it. 

            d = {}
            d['SubUnitName'] = default_R_subunit_name
            d['input_names'] = [input_name]
            d['output_name'] = output_name
            
            d['input_layerid'] = A_layerid
            d['output_layerid'] = B_layerid
        
            SubUnit_List.append(d)
            
    df_SubUnit = pd.DataFrame(SubUnit_List)
    return df_SubUnit


