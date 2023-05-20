import os
import pandas as pd


def get_EmbeddingBlock_SubUnit(full_recfldgrn_list, default_E_subunit_name = 'E'):
    SubUnit_List = []
    for input_name in full_recfldgrn_list:
        d = {}
        d['SubUnitName'] = default_E_subunit_name
        
        recfldgrn = '-'.join(input_name.split('-')[-2:])
        output_layerid = len(input_name.split('-'))
        output_name = input_name.split('Grn')[0]
        
        d['input_names'] = [input_name]
        d['output_name'] = output_name
        
        d['output_layerid'] = output_layerid
        
        SubUnit_List.append(d)
        
    df_SubUnit = pd.DataFrame(SubUnit_List)
    
    return df_SubUnit


def get_Default_ExpanderNNPara(full_recfldgrn, fldgrn_folder):
    
    # (1) get basic information
    recfld = [i for i in full_recfldgrn.split('-') if '@' in i][0]
    # rec, fld = recfld.split('@')
    # grn_suffix = [i for i in full_recfldgrn.split('-') if 'Grn' in i][0]
    # grn, suffix = grn_suffix.split('_')
    # prefix_ids = [i for i in full_recfldgrn.split('-') if 'Grn' not in i and '@' not in i]
    # recfldgrn = rec + '@' + fld + '-' + grn

    # (2) get vocab information
    # fldgrn_folder = 'data/ProcData/FldGrnInfo'
    fullfldgrn_file = os.path.join(fldgrn_folder, full_recfldgrn[2:] + '.p')
    Info = pd.read_pickle(fullfldgrn_file)


    # (3) get vocab information
    # no matter what type of grain, in the end, we will have vocab_tokenizer.
    # if 'LLM' in full_recfldgrn:
    #     vocab_tokenizer = df_FieldGrainInfo[df_FieldGrainInfo['recfield2grain'] == recfldgrn].iloc[0]['Vocab']['v2idx']
    #     init = vocab_tokenizer.name_or_path
    # else:
    #     vocab_tokenizer = df_FieldGrainInfo[df_FieldGrainInfo['recfield2grain'] == recfldgrn].iloc[0]['Vocab']['v2idx']
    #     init = 'random'

    d = {'full_recfldgrn': full_recfldgrn, 'Info': Info}
    return d