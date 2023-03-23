from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# print(type(tokenizer.backend_tokenizer))

def func_convert_SentTkLLM(x, tokenizer):
    return tokenizer(x)['input_ids']
