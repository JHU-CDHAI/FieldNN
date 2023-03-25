from transformers import AutoTokenizer
import pandas as pd
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# print(type(tokenizer.backend_tokenizer))

def func_convert_SentTkLLM(x, tokenizer):

    if pd.isna(x): 
        x = 'No information here.'
        return tokenizer(x)['input_ids']
    else:
        return tokenizer(x)['input_ids']
