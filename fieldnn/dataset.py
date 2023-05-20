import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
import os 
from recfldgrn.datapoint import load_df_data_from_folder
from .utils.layerfn import convert_relational_list_to_numpy


class RFGDataset(Dataset):
    def __init__(self, Tensor_folder, recfldgrn_list, Elig_Set, RecRootID = 'PID'):
        self.recfldgrn_list = recfldgrn_list
        self.Tensor_folder = Tensor_folder
        self.Elig_Set = Elig_Set
        self.RecRootID = RecRootID
        
        L = []
        for recfldgrn in recfldgrn_list:
            # (1) get tensor_folder
            tensor_folder = os.path.join(Tensor_folder, recfldgrn)
            # (2) get df_Pat and full_recfldgrn
            df_tensor_fnl = load_df_data_from_folder(tensor_folder)
            df_tensor_fnl = df_tensor_fnl[df_tensor_fnl[RecRootID].isin(Elig_Set)].reset_index(drop = True).set_index(RecRootID) 
            L.append(df_tensor_fnl)
        data = reduce(lambda left, right: pd.merge(left, right, on=RecRootID), L)
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index]# [full_recfldgrn]
        y = np.random.choice([0,1]) # go back to label later.
        return {'x': x, 'y': y} # torch.tensor(x), torch.tensor(y)
    
# dataset = RFGDataset(Tensor_folder, recfldgrn_list, Elig_Set, RecRootID = 'PID')
# len(dataset)


def my_collate_fn(batch_input):
    ##############
    # inputs: you can check the following inputs in the above cells.
    # (1): relational_list
    # (2): new_full_recfldgrn
    # (3): suffix
    ##############
    df_batch = pd.DataFrame([i['x'].to_dict() for i in batch_input])
    recfldgrn_sfx_list = [i for i in df_batch.columns]
    
    batch_rfg = {}

    for recfldgrn_sfx in recfldgrn_sfx_list:
        relational_list = df_batch[recfldgrn_sfx].to_list()
        B_recfldgrn_sfx = 'B-' + recfldgrn_sfx # B- means Batch. 
        suffix = '_' + B_recfldgrn_sfx.split('_')[-1]
        D = convert_relational_list_to_numpy(relational_list, B_recfldgrn_sfx, suffix)
        tensor_idx = D[B_recfldgrn_sfx]
        # print(B_recfldgrn_sfx, '<--- B_recfldgrn_suffix')
        # print(tensor_idx.shape, '<------- the shape of tensor_idx')
        batch_rfg[B_recfldgrn_sfx] = torch.Tensor(tensor_idx)
        # print('\n')
        
    # batch_y = torch.LongTensor([i['y'] for i in batch_input])  # ignore this
    batch_y = torch.LongTensor([[i['y']] for i in batch_input])  # ignore this
    return batch_rfg, batch_y