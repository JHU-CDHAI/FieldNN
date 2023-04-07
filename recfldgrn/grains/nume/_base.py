import numpy as np
import pandas as pd

def func_convert_Num_factory(start, end, Min, Max, scale):
    
    def func_convert_NUMgrn(v):
    
        miss, more, less, bottom, top = 0, 0, 0, 0, 0; base = 1
        if pd.isna(v) == True:
            miss = 1; base = 0
            num_embed = int((end - start) / scale)
            x = np.zeros(num_embed + 1)# .astype(int)
            x = list(x)
            li = [miss, more, less, bottom, top, base] + x
            num_embed = num_embed + 7
        else:
            if type(v) == str:
                if v[0] not in '<>-':
                    v = float(v)
                elif v[0] == '<':
                    less = 1
                    v = float(v[1:])
                elif v[0] == '>':
                    more = 1
                    v = float(v[1:])
                
            if v < start: 
                bottom = (start - v) / (start - Min)
                base = 0
                v = start
                
            elif v > end: 
                top = (v-end) / (Max-end)
                v= end

            num_embed = int((end - start) / scale)
            x = np.zeros(num_embed + 1)# .astype(f)
            num = int((v - start) / scale)
            # print(num)
            x[:num] = 1
            # print(v, start, scale,  v - start, num)
            # print( (v-start) - num * scale)
            x[num] = ((v-start) - num * scale) / scale
            x = x.round(4)
            # print((v - start) % scale, v, start, scale)
            # print(x[num])
            # print(x)
            x = list(x) # value part

            li = [miss, more, less, bottom, top, base] + x
            num_embed = num_embed + 7
        return li
    
    return func_convert_NUMgrn