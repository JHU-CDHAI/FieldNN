

def get_recfldgrn_embedding(full_recfldgrn, embed_size, vocab_size, init = 'random'):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'vocab_size': vocab_size}
    return embed_para


def get_recfldgrn_llmembedding(full_recfldgrn, embed_size, tokenizer, init):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'tokenizer': tokenizer}
    return embed_para


def get_expander_para(full_recfldgrn, 
                      nn_name,
                      embed_size, vocabsize_tokenizer, init, 
                      postprocess):
    
    expander_para = {}
    #(0) Input size, output size
    expander_para['input_size'] = None
    expander_para['output_size'] = embed_size
    expander_para['nn_name'] = nn_name

    #(1) Main NN
    # print(nn_name)
    if nn_name.lower() == 'catembeddinglayer':
        para = get_recfldgrn_embedding(full_recfldgrn, embed_size, vocabsize_tokenizer, init)
    elif nn_name.lower() == 'numembeddinglayer':
        para = get_recfldgrn_embedding(full_recfldgrn, embed_size, vocabsize_tokenizer, init)
    elif nn_name.lower() == 'llmembeddinglayer':
        para = get_recfldgrn_llmembedding(full_recfldgrn, embed_size,  vocabsize_tokenizer, init)
    else:
        raise ValueError(f'The NN "{nn_name}" is not available yet')
    
    # print(para, '<--------- inside get_expander_para')
    expander_para[full_recfldgrn] = nn_name, para

    # (2) Post Process
    expander_para['postprocess'] = postprocess
    
    return expander_para