import torch

from ..nn.operator.op import ReduceMeanLayer, ReduceSumLayer, RecuderMaxLayer, ConcatenateLayer
# from fieldnn.nn.operator.op import ReduceMeanLayer, ReduceSumLayer, RecuderMaxLayer, ConcatenateLayer

class Reducer_Layer(torch.nn.Module):
    def __init__(self, input_names_nnlvl, output_name_nnlvl, reducer_layer_para):
        super(Reducer_Layer, self).__init__()
        
        # Part 0: Meta
        # here input_names and out_tensor just the tensor name, 
        # they are not the real tensors
        # intead, the info_dict contains the corresponding real tensors.
        assert len(input_names_nnlvl) == 1
        self.input_names_nnlvl = input_names_nnlvl
        self.input_name_nnlvl = input_names_nnlvl[0]
        
        # output_name should be generated from the input_names
        self.output_name_nnlvl = output_name_nnlvl
        # self.output_name = output_name

        # the input feature dim size and output feature dim size
        self.input_size = reducer_layer_para['input_size']
        self.output_size = reducer_layer_para['output_size']

        # Part 1: NN
        nn_name, nn_para = reducer_layer_para['nn_name'], reducer_layer_para['nn_para']
        if nn_name.lower() == 'mean':
            self.reducer = ReduceMeanLayer()
        elif nn_name.lower() == 'sum':
            self.reducer = ReduceSumLayer()
        elif nn_name.lower() == 'max':
            self.reducer = RecuderMaxLayer()
        elif nn_name.lower() == 'concat':
            self.reducer = ConcatenateLayer()
            # TODO: need to assert something
            assert self.output_size % self.input_size == 0
        else:
            raise ValueError(f'There is no layer for "{nn_name}"')
            
        # Part 2: PostProcess
        self.postprocess = torch.nn.ModuleDict()
        for method, config in reducer_layer_para['postprocess'].items():
            if method == 'activator':
                activator = config
                if activator.lower() == 'relu': 
                    self.postprocess[method] = torch.nn.ReLU()
                elif activator.lower() == 'tanh': 
                    self.postprocess[method] = torch.nn.Tanh()
                elif activator.lower() == 'gelu':
                    self.postprocess[method] = torch.nn.GELU()
            elif method == 'dropout':
                self.postprocess[method] = torch.nn.Dropout(**config)
            elif method == 'layernorm':
                self.postprocess[method] = torch.nn.LayerNorm(self.output_size, **config)
            
    def forward(self, input_names_nnlvl, INPUTS_TO_INFODICT):
        # information preparation.
        # 'INPUTS_TO_INFODICT` will come from SubUnit Layer.
        assert len(input_names_nnlvl) == 1
        input_name_nnlvl = input_names_nnlvl[0]
        assert self.input_name_nnlvl == input_name_nnlvl
        
        info_dict = INPUTS_TO_INFODICT[input_name_nnlvl]
        holder, info = info_dict['holder'], info_dict['info']
        
        # print(holder.shape, info.shape)
        # the following part is the data proprocessing
        leng_mask = holder == 0
        info = self.reducer(info, leng_mask)
        
        for name, layer in self.postprocess.items():
            info = layer(info)
            
        holder = (leng_mask == 0).sum(-1)
        
        # output_name_nnlvl is not necessarily to be stored in the 
        return self.output_name_nnlvl, {'holder': holder, 'info': info}
    