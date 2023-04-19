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


def update_df_Repr_dataflow(df, style = 'Reducer&Merger'):
    
    '''
        This function try to auto-fill the data flow from each field to the end field 'B-P'.
        There are two styles:
            style 'Reducer&Merger': Merging the tensors along the data flow.
            style 'ReducerOnly': Don't merge the tensors until the end of the models. 
    
    '''
    
    
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
                
                if '@' in full_recfldgrn.split('-')[-1]:
                    
                    prefix = '-'.join(full_recfldgrn.split('-')[:-1])
                    curfld = full_recfldgrn.split('-')[-1] # PNSectSent@Sentence@Tk
                    headfld, tailfld = get_curfld_recinfo(curfld) # head: PNSectSent@Sentence, tail: Tk
                    
                    full_recfldgrn_new = full_recfldgrn.replace('@' + tailfld, '')
                    same_rfg_new = [i for i in current_fullrfg_list if full_recfldgrn_new in i]
                    
                    # print(f'for Layer {layer_idx}')
                    # print(full_recfldgrn, '<--- full_recfldgrn')
                    # print(full_recfldgrn_new, '<--- full_recfldgrn_new')
                    # print(same_rfg_new, '<--- same_rfg_new')
                    
                    if len(same_rfg_new) == 1:
                        # print(full_recfldgrn)
                        df_dataflow.loc[index, layer_idx] = full_recfldgrn_new
                        ############# need to think about this
                        s = df_dataflow[layer_idx]
                        current_fullrfg_list = s[-s.isna()].to_list()
                        #############
                        

                    elif len(same_rfg_new) > 1:
                        
                        if style == 'Reducer&Merger':
                            fld_with_at_to_keep.extend(same_rfg_new)
                            
                            # To update a new Column
                            tailfld_merged = '&'.join([i.replace(full_recfldgrn_new +'@', '') for i in same_rfg_new])
                            new_index = '(Merge)' + headfld + '@' + tailfld_merged#
                            if new_index in df_dataflow.index: continue

                            selected_index_list = [df_dataflow[df_dataflow[layer_idx] == i].index[0] for i in same_rfg_new]
                            # print(selected_index_list)

                            l = list(df_dataflow.index)
                            # print(l, '<--- l original columns')
                            loc = max([l.index(i) for i in selected_index_list])
                            l.insert(loc + 1, new_index)
                            # print(l, '<--- l with new index columns')

                            df_dataflow.loc[new_index, layer_idx] = full_recfldgrn_new
                            df_dataflow = df_dataflow.reindex(l)
                            # print(list(df_dataflow.index), '<--- df_dataflow with new index columns')

                            ############# need to think about this
                            # s = df_dataflow[layer_idx]
                            # current_fullrfg_list = s[-s.isna()].to_list()
                            #############
                            fld_with_at_to_keep.append(full_recfldgrn_new)
                            # print(fld_with_at_to_keep, '<----------------'
                            
                        elif style == 'ReducerOnly':
                            fld_with_at_to_keep.extend(same_rfg_new)
                            
                        else:
                            raise ValueError('Wrong style, only for Reducer&Merger and ReducerOnly')
                            
            
            # (2) update conditions
            s = df_dataflow[layer_idx]
            current_fullrfg_list = s[-s.isna()].to_list()
            fld_with_at = [i for i in current_fullrfg_list if '@' in i.split('-')[-1]]
            fld_with_at_to_check = [i for i in fld_with_at if i not in fld_with_at_to_keep]
            
        if layer_idx == df_dataflow.columns[-1]: continue
        # then update the next layer information
        for index in df_dataflow.index:
            last = df_dataflow.loc[index, layer_idx]
            if pd.isna(last): continue
            if '@' in last.split('-')[-1] and style == 'Reducer&Merger': continue
            full_recfldgrn_next = '-'.join(last.split('-')[:-1]) + '@' + last.split('-')[- 1]
            df_dataflow.loc[index, layer_idx - 1] = full_recfldgrn_next
            
            
    if style == 'ReducerOnly':
        df_dataflow.loc['(Merge)P@All', 2] = 'B-P' 
        
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
                          default_MR_subunit_name = 'MRL', # or 'MLRL'
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

        # print(f'\nFrom Layer {A_layerid} to {B_layerid}:')

        # from A tensor to B tensor, we have the Reducer NNs.
        # also notice that some 

        # print('A', A_tensors)
        for k in A_tensors:
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


        # from B tensor: potential there are some Merger NNs. 
        # print('B', B_tensors)
        # check whether these is a '(Merger') in the key
        merger_tensors = [v for k, v in B_tensors.items() if '(Merge)' in k and '@' not in v]
        # print('B-merger_tensors', merger_tensors)

        for output_tensor in merger_tensors:
            input_tensors = [i for k, i in B_tensors.items() if output_tensor + '@' in i]
            # print(output_tensor, ':', input_tensors)

            d = {}
            d['SubUnitName'] = default_MR_subunit_name
            d['input_names'] = input_tensors
            d['output_name'] = output_tensor
            
            d['input_layerid'] = B_layerid
            d['output_layerid'] = B_layerid
        
            SubUnit_List.append(d)
            
    df_SubUnit = pd.DataFrame(SubUnit_List)
    return df_SubUnit


