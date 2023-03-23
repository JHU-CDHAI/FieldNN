import pandas as pd

def get_SectNmDftGrn_list(x):
    if pd.isna(x):
        return ['_missing']
    else:
        return x.split('/')
    
