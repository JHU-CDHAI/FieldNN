import os
import random
import numpy as np
import pandas as pd

RANGE_SIZE = 10000


def get_df_rec(attr, path, old_columns, new_columns):
    df = pd.read_csv(path, low_memory = False)
    print(df.columns)
    df['RID'] = df.index
    for i in old_columns:
        if i not in  df.columns:
            df[i] = None
            
    df = df[old_columns]
    df.columns = new_columns
    # df = df.reset_index().rename(columns = {'index': 'RID'})
    DT_col = [i for i in new_columns if 'DT' in i][0]
    df[DT_col] = pd.to_datetime(df[DT_col])
    df = df.sort_values(['PID', DT_col]).reset_index(drop = True)
    # print(df.shape)
    df = df[df['ECID'] != 0].reset_index(drop = True)
    # print(df.shape)
    return df



def convert_PID_to_PIDgroup(x, RANGE_SIZE):
    int_value = int(x.replace('P', ''))
    range_size = RANGE_SIZE
    group_idx = int(int_value / range_size)
    s = group_idx * range_size
    e = (group_idx + 1) * range_size - 1
    groupname = f'PIDGroup_s{s}_e{e}'
    return groupname

def write_df_to_folders(DataName, data_folder, df):
    dfy = df.copy()
    # data_folder = f'../Data/B-EHRDataSet/{DataName}'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    dfy['GroupName'] = dfy['PID'].apply(lambda x: convert_PID_to_PIDgroup(x , RANGE_SIZE))
    for groupname, dfx in dfy.groupby('GroupName'):
        file = os.path.join(data_folder, f'{groupname}.p')
        # print(file)
        dfx = dfx.drop(columns= ['GroupName'])
        dfx.to_pickle(file)

def load_df_data_from_folder(data_folder):
    file_list = os.listdir(data_folder)
    # file_lis
    df = pd.concat([pd.read_pickle(os.path.join(data_folder, i)) for i in file_list]).reset_index(drop = True)
    return df# .shape


def get_dfx_p_filename(PID, DataName, data_folder, RANGE_SIZE):
    group_name = convert_PID_to_PIDgroup(PID, RANGE_SIZE) 
    p_filename = f'{data_folder}/{DataName}/{group_name}.p'
    return p_filename
    
def load_patient_data(PID, dfx):
    dfx = dfx[dfx['PID'] == PID].reset_index(drop = True)
    return dfx

def get_dfx_from_buffer(PID, DataName, data_folder, RANGE_SIZE, BUCKET_buffer):
    p_filename = get_dfx_p_filename(PID, DataName, data_folder, RANGE_SIZE)
    if p_filename not in BUCKET_buffer:
        dfx = pd.read_pickle(p_filename)
        BUCKET_buffer[p_filename] = dfx
        if len(BUCKET_buffer) > 100:
            for key in random.sample(BUCKET_buffer.keys(), 4):
                del BUCKET_buffer[key]
    else:
        dfx = BUCKET_buffer[p_filename]
    return dfx, BUCKET_buffer


class PatientDP(object):
    BUCKET_buffer = {}
    
    
    def __init__(self, PID, data_folder, RANGE_SIZE):
        self.PID = PID
        self.data_folder = data_folder
        self.df_rec_dict = {}
        self.RANGE_SIZE = RANGE_SIZE
        
        
    def get_df_rec(self, DataName):
        if DataName not in self.df_rec_dict:
           # df = load_field_data(self.PID, DataName, self.data_folder)
            
            dfx, BUCKET_buffer = get_dfx_from_buffer(self.PID, DataName, 
                                                     self.data_folder, 
                                                     self.RANGE_SIZE, 
                                                     self.BUCKET_buffer)
            self.BUCKET_buffer = BUCKET_buffer
            df_p = load_patient_data(self.PID, dfx)
            self.df_rec_dict[DataName] = df_p
            return df_p
        else:
            return self.df_rec_dict[DataName]
    
        
        
    
    