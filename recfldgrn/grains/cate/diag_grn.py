import pandas as pd

def func_convert_DiagVgrn(x):
    '''x here is icd-10'''
    if pd.isna(x): 
        return ['_missing']
    else:
        li = ['A-' + x[0], 'B-' + x[:3], 'C-' + x.split('.')[-1], 
              # 'D-' +x
              ]
        return li