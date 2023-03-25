import pandas as pd


def load_DTgrn_Dictionary():
    Y = ['Yx'] + [f'Y{i}' for i in range(2010, 2030)]
    M = ['Mx'] +[f'M{i}' for i in range(1, 13)]
    D = ['Dx'] + [f'D{i}' for i in range(1, 32)]
    S = ['Sx'] +[f'S{i}' for i in range(1, 5)]
    W = ['Wx'] + [f'W{i}' for i in range(1, 8)]
    WY = ['WYx'] + [f'WY{i}' for i in range(1, 56)]
    H = ['Hx'] + [f'H{i}' for i in range(0, 24)]
    value_list = ['_padding', '_missing'] + Y + M + D + S + W + WY + H 
    v2freq = {k:0 for k in value_list}
    idx2v = {idx: v for idx, v in enumerate(v2freq.keys())}
    v2idx = {v:k for k, v in idx2v.items()}
    DT_v2idx2freq = {'idx2v': idx2v, 'v2idx': v2idx, 'v2freq': v2freq}
    return DT_v2idx2freq


def func_convert_DTgrn(x):
    if pd.isna(x):
        nan = 'x'
        return f'Y{nan} M{nan} D{nan} S{nan} W{nan} WY{nan} H{nan}'.split()
    
    # dfx['DT'] = pd.to_datetime(dfx['DT'])
    DT = x
    year = DT.year
    month = DT.month
    date = DT.day
    season = int(month / 4)+ 1
    weekday = DT.weekday() + 1
    weekofyear = int(DT.isocalendar()[1]) + 1
    hour = DT.hour
    
    return f'Y{year} M{month} D{date} S{season} W{weekday} WY{weekofyear} H{hour}'.split()
    