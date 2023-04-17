
def get_cateembed_para(embed_size, vocab_tokenizer, init = 'random'):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'vocab_size': len(vocab_tokenizer)}
    return embed_para

def get_numeembed_para(embed_size, vocab_tokenizer, init = 'random'):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'vocab_size': len(vocab_tokenizer)}
    return embed_para

def get_llmembed_para(embed_size, tokenizer, init):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'tokenizer': tokenizer}
    return embed_para


def get_expander_para(nn_name, nn_para,
                      embed_size, vocab_tokenizer, init, 
                      postprocess):
    
    expander_para = {}
    
    expander_para['nn_type'] = 'expander'
    expander_para['nn_name'] = nn_name

    # (1) get the parameters

    if nn_name.lower() == 'cateembed':
        para = get_cateembed_para(embed_size, vocab_tokenizer, init)
    elif nn_name.lower() == 'numeembed':
        para = get_numeembed_para(embed_size, vocab_tokenizer, init)
    elif nn_name.lower() == 'llmembed':
        para = get_llmembed_para(embed_size,  vocab_tokenizer, init)
    else:
        raise ValueError(f'The NN "{nn_name}" is not available yet')
        
    expander_para['nn_para'] = para

    #(2) Input size, output size
    expander_para['input_size'] = None
    expander_para['output_size'] = embed_size

    # (3) Post Process
    expander_para['postprocess'] = postprocess
    
    return expander_para