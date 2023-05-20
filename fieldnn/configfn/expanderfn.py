def get_cateembed_para(embed_size, vocab_size, init = 'random'):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'vocab_size': vocab_size}
    return embed_para


def get_llmembed_para(embed_size, tokenizer, init):
    embed_para =  {'embedding_size': embed_size,
                   'init': init, 
                   'tokenizer': tokenizer}
    return embed_para


def get_expander_para(full_recfldgrn, Info, embed_size, postprocess):
    
    expander_para = {}
    #(1) Input size, output size
    expander_para['input_size'] = None
    expander_para['output_size'] = embed_size
    expander_para['nn_type'] = 'expander'
    
    # (2) Loop Embed Info for each sfx
    EmbedDict = Info['EmbedDict']
    
    for sfx in EmbedDict:
        # embed_para = expander_para[input_name_nnlvl]
        input_name_nnlvl = full_recfldgrn + '_' + sfx + 'idx'
        
        EmbedConf = EmbedDict[sfx]
        nn_name = EmbedConf['embed_type']
        
        embed_para = {}
        embed_para['nn_name'] = nn_name
        
        if nn_name.lower() == 'cateembed':
            vocab_size = EmbedConf['vocab_size']
            init = 'random' # init = EmbedConf['init'] # TODO: to update in the future.
            para = get_cateembed_para(embed_size, vocab_size, init)
            embed_para['nn_para'] = para
        elif nn_name.lower() == 'llmembed':
            tokenizer = EmbedConf['tknz']
            init = tokenizer.name_or_path
            para = get_llmembed_para(embed_size, tokenizer, init)
            embed_para['nn_para'] = para
        else:
            raise ValueError(f'The NN "{nn_name}" is not available yet')
        
        expander_para[input_name_nnlvl] = embed_para
    
    # (3) Post Process
    expander_para['postprocess'] = postprocess
    
    
    # (4) use_wgt
    expander_para['use_wgt'] = Info.get('use_wgt', True)
    
    return expander_para