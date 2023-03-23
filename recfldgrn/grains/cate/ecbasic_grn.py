import pandas as pd

def get_basicInfoGrn_list(x):
    if pd.isna(x):
        return ['_missing']
    else:
        return x.split('&')