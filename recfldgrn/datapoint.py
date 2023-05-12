# from recfldgrn.datapoint import convert_PID_to_PIDgroup
# from recfldgrn.datapoint import RANGE_SIZE, write_df_to_folders, load_df_data_from_folder
import os 
import random
import pandas as pd

def convert_PID_to_PIDgroup(x, RANGE_SIZE):
    if type(x) == str:
        if 'P' in x:
            int_value = int(x.replace('P', ''))
    else:
        int_value = int(x)
    range_size = RANGE_SIZE
    group_idx = int(int_value / range_size)
    s = group_idx * range_size
    e = (group_idx + 1) * range_size - 1
    groupname = f'Group_s{s}_e{e}'
    return groupname

def write_df_to_folders(DataName, data_folder, df, IDname):
    dfy = df.copy()
    # data_folder = f'../Data/B-EHRDataSet/{DataName}'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    dfy['GroupName'] = dfy[IDname].apply(lambda x: convert_PID_to_PIDgroup(x , RANGE_SIZE))
    for groupname, dfx in dfy.groupby('GroupName'):
        file = os.path.join(data_folder, f'{groupname}.p')
        # print(file)
        dfx = dfx.drop(columns= ['GroupName'])
        dfx.to_pickle(file)
        
        
RANGE_SIZE = 10000


def load_df_data_from_folder(data_folder, IDName = None):
    file_list = os.listdir(data_folder)
    # file_lis
    df = pd.concat([pd.read_pickle(os.path.join(data_folder, i)) for i in file_list if '.p' in i]).reset_index(drop = True)
    # df = df.sort_values(IDName).reset_index(drop = True)
    return df# .shape


def get_dfx_p_filename(IDValue, DataName, data_folder, RANGE_SIZE):
    group_name = convert_PID_to_PIDgroup(IDValue, RANGE_SIZE) 
    p_filename = f'{data_folder}/{DataName}/{group_name}.p'
    return p_filename
    
def load_patient_data(RID, IDValue, dfx):
    dfx = dfx[dfx[RID] == IDValue].reset_index(drop = True)
    return dfx

def get_dfx_from_buffer(RID, IDValue, DataName, data_folder, RANGE_SIZE, BUCKET_buffer):
    p_filename = get_dfx_p_filename(IDValue, DataName, data_folder, RANGE_SIZE)
    if p_filename not in BUCKET_buffer:
        dfx = pd.read_pickle(p_filename)
        BUCKET_buffer[p_filename] = dfx
        if len(BUCKET_buffer) > 100:
            for key in random.sample(BUCKET_buffer.keys(), 4):
                del BUCKET_buffer[key]
    else:
        dfx = BUCKET_buffer[p_filename]
    return dfx, BUCKET_buffer


class DataPoint(object):

    '''
    DataPoint: stands for one data point. 
    one instance is a patient.
    the instance of the patient could quickly the his/her record information by using `get_df_rec`.
    '''
    BUCKET_buffer = {}

    def __init__(self, RID, IDValue, data_folder, RANGE_SIZE):
        self.RID = RID
        self.IDValue = IDValue
        self.data_folder = data_folder
        self.df_rec_dict = {}
        self.RANGE_SIZE = RANGE_SIZE
        
    def get_df_rec(self, DataName):
        '''
        DataName: the name of the record, like: P, EC, BMI, PN, PNSect, etc.
        '''
        if DataName not in self.df_rec_dict:
            dfx, BUCKET_buffer = get_dfx_from_buffer(self.RID, self.IDValue, 
                                                     DataName, 
                                                     self.data_folder, 
                                                     self.RANGE_SIZE, 
                                                     self.BUCKET_buffer)
            self.BUCKET_buffer = BUCKET_buffer
            df_p = load_patient_data(self.RID, self.IDValue, dfx)
            self.df_rec_dict[DataName] = df_p
            return df_p
        else:
            return self.df_rec_dict[DataName]
        
    def __repr__(self):
        return f'<{self.RID} [{self.IDValue}]>'
    