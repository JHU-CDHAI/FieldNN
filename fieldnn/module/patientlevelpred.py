import torch
from .embedblock import EmbedBlockLayer
from .reprblock import ReprBlockLayer


class PatientLevelPredictionModel(torch.nn.Module):
    """ Naive softmax-layer """
    def __init__(self, df_Embed_SubUnit, df_Repr_SubUnit, 
                 actn_fn_name, loss_fn_name,
                 output_name, embed_size, output_size):
        
        '''
            df_Embed_SubUnit: df_SubUnit for the Embed Block.
            df_Repr_SubUnit: df_Repr_SubUnit for the Repr Block.
            actn_fn_name: like sigmoid, softmax
            loss_fn_name: like CrossEntropy, BCELoss.
            output_name: the featvec name.
            embed_size: the featvec dim.
            output_size: the output dim size. if actn_fn is softmax, then label sets. if actn_fn is sigmoid, then 1. 
            
        '''
        super(PatientLevelPredictionModel, self).__init__()
        
        self.output_name = output_name
        
        self.EmbedBlock = EmbedBlockLayer(df_Embed_SubUnit)
        self.ReprBlock  = ReprBlockLayer(df_Repr_SubUnit)
        self.OutputBlock = torch.nn.Linear(embed_size, output_size)
        
        self.actn_fn_name = actn_fn_name
        if self.actn_fn_name == 'Sigmoid':
            self.actn_method = torch.nn.Sigmoid()
            self.actn_fn = lambda outputvecs: self.actn_method(outputvecs) # will return probs 
        elif self.actn_fn_name == 'Softmax':
            self.actn_method = torch.nn.Softmax()
            self.actn_fn = lambda outputvecs: self.actn_method(outputvecs, dim = 1) # will return probs 
        else:
            raise ValueError(f'Activation Function Name {actn_fn_name} is not available yet')
        
        self.loss_fn_name = loss_fn_name
        if self.loss_fn_name == 'BCELoss':
            assert self.actn_fn_name == 'Sigmoid'
            self.loss_method = torch.nn.BCELoss()
            self.loss_fn = lambda probs, targets: self.loss_method(probs, targets) # will return loss
        
        elif self.loss_fn_name == 'CrossEntropyLoss':
            assert self.actn_fn_name == 'Softmax'
            self.loss_method = torch.nn.CrossEntropyLoss()
            self.loss_fn = lambda probs, targets: self.loss_method(probs, targets) # will return loss
        
        else:
            raise ValueError(f'Loss Function Name {loss_fn_name} is not available yet')
        

    def get_REPR_TENSOR(self, batch_rfg):
        # get the full_recfldgrn_list
        full_recfldgrn_list = [i for i in batch_rfg]

        # prepare RECFLD_TO_TENSOR
        RECFLD_TO_TENSOR = {}
        for full_recfldgrn in full_recfldgrn_list:
            # (1) get the info_raw from batch_rfg
            info_raw = batch_rfg[full_recfldgrn]

            # (2) get the holder (input_idx) and holder_wgt (for nume embedding only)
            if '_idx' in full_recfldgrn:
                holder_wgt = 'Empty'
                holder = torch.LongTensor(info_raw)
            elif '_wgt' in full_recfldgrn:
                holder_wgt = torch.FloatTensor(info_raw)
                # ATTENTION: here holder_wgt could contain zeros in some valid positions.
                holder = torch.ones_like(holder_wgt).cumsum(-1).masked_fill(holder_wgt == 0, 0).long()
            else:
                raise ValueError(f'Invalid full_recfldgrn "{full_recfldgrn}"')

            info_dict = {'holder': holder, 'holder_wgt': holder_wgt}
            RECFLD_TO_TENSOR[full_recfldgrn] = info_dict
    
        # get RECLD_TO_EMBEDTENSOR from RECFLD_TO_TENSOR and EmbedBlock
        RECFLD_TO_EMBEDTESNOR = self.EmbedBlock(RECFLD_TO_TENSOR)
        
        # update the full_recfldgrn_list
        full_recfldgrn_list = [i for i in RECFLD_TO_EMBEDTESNOR]

        # update the names of full_recfldgrn_list
        fld_updates_dict = {}
        for i in RECFLD_TO_EMBEDTESNOR:
            layernum = len(i.split('-'))
            fld = i.split('-')[-1]
            if '@' not in fld: continue
            
            # print(fld)
            neat_i = '-'.join(i.split('-')[:-1]) + '-' + fld.split('@')[0]
            # print(neat_i)
            same_neat_list = [t for t in RECFLD_TO_EMBEDTESNOR if neat_i + '@' in t]
            # print(same_neat_list)
            if len(same_neat_list) == 1: fld_updates_dict[i] = neat_i

        for old, new in fld_updates_dict.items():
            RECFLD_TO_EMBEDTESNOR[new] = RECFLD_TO_EMBEDTESNOR.pop(old)
            
        # get the OUTPUT_TO_TENSOR data holder
        REPR_TENSOR = self.ReprBlock(RECFLD_TO_EMBEDTESNOR)
        return REPR_TENSOR

    def loss_funciton(self, batch_rfg, batch_targets):
        # batch -> Embed -> Repr -> repr tensor
        REPR_TENSOR = self.get_REPR_TENSOR(batch_rfg)
        info_dict = REPR_TENSOR[self.output_name]
        featvecs = info_dict['info']
        
        outputvecs = self.OutputBlock(featvecs)
        
        probs = self.actn_fn(outputvecs) 
        l = self.loss_fn(probs, batch_targets)
        return l
        
    def forward(self, batch_rfg):
        # forward is the method for the inference.
        # batch -> Embed -> Repr -> repr tensor
        REPR_TENSOR = self.get_REPR_TENSOR(batch_rfg)
        info_dict = REPR_TENSOR[self.output_name]
        featvecs = info_dict['info']
        
        outputvecs = self.OutputBlock(featvecs)
        
        probs = self.actn_fn(outputvecs) 
        return probs