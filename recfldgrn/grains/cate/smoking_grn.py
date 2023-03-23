import pandas as pd
from ._base import process_input_str

def func_convert_Smokinggrn(x):
    if pd.isna(x):
        return ['_missing']
    else:
        return [x]