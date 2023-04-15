import torch
import torch.nn.functional as F
# from fieldlm.nn.helper import reverse_tensor, _addindent, gelu
# from .helper import reverse_tensor, _addindent, gelu

class RNNLayer(torch.nn.Module):
    
    def __init__(self, 
                 type = 'lstm', 
                 n_layers = 1,
                 input_type = 'INPUT-NML',  # ['INPUT-NML', 'INPUT-SEP']
                 direction_type = 'FWD',    # ['FWD', 'BI-MIX', 'BI-SEP']
                 struct_type = 'EXTRACTOR', # ['EXTRACTOR', 'REDUCER']
                 input_size = 200, output_size = 200, rnn_dropout = 0.5,
                 ):

        super(RNNLayer, self).__init__()
        
        assert input_type in ['INPUT-NML', 'INPUT-SEP']
        assert direction_type in ['FWD', 'BI-MIX', 'BI-SEP'] # for mu
        assert struct_type in ['EXTRACTOR', 'REDUCER']
        
        # rnn_type can be lstm, gru, rnn
        self.type = type 
        
        # produce n identical layers.
        # when n_layers = 1, and bi= True, BI-MIX and BI-SEP are the same.
        self.n_layers = n_layers 
        if self.n_layers == 1: rnn_dropout = 0
        
        # (+) input_type: ['INPUT-NML', 'INPUT-SEP']
        self.input_type = input_type
        self.input_size = input_size
        if self.input_type == 'INPUT-SEP':
            assert input_size % 2 == 0 
            self.rnn_input_size = int( input_size / 2)
        elif self.input_type == 'INPUT-NML':
            self.rnn_input_size = input_size
        
        # (+) direction_type:  ['FWD', 'BI-MIX', 'BI-SEP']
        self.direction_type = direction_type
        self.n_directions = 1 if direction_type == 'FWD' else 2 
        
        # (+) struct_type: ['EXTRACTOR', 'REDUCER']
        self.struct_type = struct_type
        
        # (+) output size   
        assert output_size % self.n_directions == 0 
        self.output_size = output_size
        self.hidden_size = int(output_size / self.n_directions)
        
        # (+) dropout rate
        self.rnn_dropout = rnn_dropout
        
        # (+) initialize self.rnn here.
        if self.type.lower() == 'lstm':
            
            # (+) BUILD RNN
            if direction_type == 'FWD':
                # 1. initialize lstm
                self.rnn = torch.nn.LSTM(self.rnn_input_size, self.hidden_size, self.n_layers, 
                                         dropout = rnn_dropout, 
                                         bidirectional=False, # pay attention when doing LM
                                         batch_first = True)  # don't miss batch_first = True
                self.rnn_op = self.lstm_op_allnormal
                # output can be extractor or reducer
                
            
            elif direction_type == 'BI-MIX':
                # 1. initialize lstm
                self.rnn = torch.nn.LSTM(self.rnn_input_size, self.hidden_size, self.n_layers, 
                                         dropout = rnn_dropout, 
                                         bidirectional=True, # pay attention when doing LM
                                         batch_first = True) # don't miss batch_first = True
                
                self.rnn_op = self.lstm_op_allnormal
                # output can be extractor or reducer

                
            elif direction_type == 'BI-SEP':
                
                # fed with fwd input
                self.rnn_fwd = torch.nn.LSTM(self.rnn_input_size, self.hidden_size, self.n_layers, 
                                              dropout = rnn_dropout, 
                                              bidirectional=False, # pay attention when doing LM
                                              batch_first = True)  # don't miss batch_first = True
                # fed with bwd input 
                self.rnn_bwd = torch.nn.LSTM(self.rnn_input_size, self.hidden_size, self.n_layers, 
                                              dropout = rnn_dropout, 
                                              bidirectional=False, # pay attention when doing LM
                                              batch_first = True)  # don't miss batch_first = True
                
                if self.input_type == 'INPUT-SEP':
                    self.rnn_op = self.lstm_op_bisep_inputsep
                    # output can be extractor or reducer
                elif self.input_type == 'INPUT-NML':
                    self.rnn_op = self.lstm_op_bisep_inputnml
                    # output can be extractor or reducer
                else:
                    raise ValueError('Not a valid lstm input type, must in [INPUT-SEP, INPUT-NML]!')
                
            else:
                raise ValueError("Not a valid lstm direction type, must in ['FWD', 'BI-MIX', 'BI-SEP']!")
                
            # (+) RUDUCER OR EXTRACTOR
            if self.struct_type == 'LEARNER':
                self.output = self.lstm_extractor
            # elif self.struct_type == 'REDUCER':
            #     self.output = self.lstm_reducer
            else:
                raise ValueError('Not a valid struct type, must be either extactor or reducer!')
            
            
    def lstm_op_allnormal(self, info, leng_st):
        # extractor type
        # (bs, a, b, c1) --> (bs, a, b, c2), where c1, c2 are input_size, output_size
        # print(info.device)
        info = torch.nn.utils.rnn.pack_padded_sequence(info, leng_st, batch_first = True) 
        
        info, (hn, cn) = self.rnn(info)
        info, _ = torch.nn.utils.rnn.pad_packed_sequence(info, batch_first = True)
        return info, (hn, cn)
    
    
    def lstm_op_bisep_inputnml(self, info, leng_st):
        # prepare data
        info_fwd = info
        info_bwd = reverse_tensor(info, leng_st) # helper_tools.reverse_tensor(info, leng_st)
        
        # get fwd output
        info_fwd = torch.nn.utils.rnn.pack_padded_sequence(info_fwd, leng_st, batch_first = True) 
        info_fwd, (hn_fwd, cn_fwd) = self.rnn_fwd(info_fwd)
        info_fwd, _ = torch.nn.utils.rnn.pad_packed_sequence(info_fwd, batch_first = True)
        
        # get bwd output
        info_bwd = torch.nn.utils.rnn.pack_padded_sequence(info_bwd, leng_st, batch_first = True) 
        info_bwd, (hn_bwd, cn_bwd) = self.rnn_bwd(info_bwd)
        info_bwd, _ = torch.nn.utils.rnn.pad_packed_sequence(info_bwd, batch_first = True)
        
        # reorder bwd
        info_bwd = reverse_tensor(info_bwd, leng_st) # info_bwd = helper_tools.reverse_tensor(info_bwd, leng_st)
        
        # concat output
        info = torch.cat([info_fwd, info_bwd], -1)
        
        # concat hidden output
        hn = torch.cat([hn_fwd.unsqueeze(1), hn_bwd.unsqueeze(1)], 1)
        batch_size, hidden_size  = hn.size(-2), hn.size(-1)
        hn = hn.view(-1, batch_size, hidden_size)
        
        # concat cell output
        cn = torch.cat([cn_fwd.unsqueeze(1), cn_bwd.unsqueeze(1)], 1)
        batch_size, hidden_size  = cn.size(-2), cn.size(-1)
        cn = cn.view(-1, batch_size, hidden_size)
        
        return info, (hn, cn)
        
        
    def lstm_op_bisep_inputsep(self, info, leng_st):
        # prepare data
        info_fwd, info_bwd = info.chunk(2, -1)
        info_bwd = reverse_tensor(info_bwd, leng_st) # info = helper_tools.reverse_tensor(info, leng_st)
        
        # get fwd output
        info_fwd = torch.nn.utils.rnn.pack_padded_sequence(info_fwd, leng_st, batch_first = True) 
        info_fwd, (hn_fwd, cn_fwd) = self.rnn_fwd(info_fwd)
        info_fwd, _ = torch.nn.utils.rnn.pad_packed_sequence(info_fwd, batch_first = True)
        
        # get bwd output
        info_bwd = torch.nn.utils.rnn.pack_padded_sequence(info_bwd, leng_st, batch_first = True) 
        info_bwd, (hn_bwd, cn_bwd) = self.rnn_bwd(info_bwd)
        info_bwd, _ = torch.nn.utils.rnn.pad_packed_sequence(info_bwd, batch_first = True)
        
        # reorder bwd
        info_bwd = reverse_tensor(info_bwd, leng_st) # info_bwd = helper_tools.reverse_tensor(info_bwd, leng_st)
        
        # concat output
        info = torch.cat([info_fwd, info_bwd], -1)
        
        # concat hidden output
        hn = torch.cat([hn_fwd.unsqueeze(1), hn_bwd.unsqueeze(1)], 1)
        batch_size, hidden_size  = hn.size(-2), hn.size(-1)
        hn = hn.view(-1, batch_size, hidden_size)
        
        # concat cell output
        cn = torch.cat([cn_fwd.unsqueeze(1), cn_bwd.unsqueeze(1)], 1)
        batch_size, hidden_size  = cn.size(-2), cn.size(-1)
        cn = cn.view(-1, batch_size, hidden_size)
        
        return info, (hn, cn)
    

    def lstm_extractor(self, info, hidden):
        # do nothing
        return info

    # def lstm_reducer(self, info, hidden):
    #     hn, cn = hidden
    #     batch_size, hidden_size = hn.size(-2), hn.size(-1) 
    #     hn = hn.view(self.n_layers, self.n_directions, batch_size, hidden_size)[-1] # only get the last layer ones.
    #     hn = torch.cat([hn[i] for i in range(self.n_directions)], dim = 1)
    #     return hn

    def forward(self, info, leng_st):
        # prepare `self.rnn_op` and `self.output` at the very first time.
        # print(info.device)
        # print(info)
        info, hidden = self.rnn_op(info, leng_st)
        info = self.output(info, hidden)
        
        for post_layer in self.postprocess:
            info = post_layer(info)
        return info