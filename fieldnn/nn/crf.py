
from .helper import pad_packed_sequence
import numpy as np

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.

    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """
    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        # </pad> is in num_tags.
        # n_tags are n_classes + 1
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # num_special_tokens: = 1
        a = torch.zeros(num_tags); a[0] = -10000.0 
        b = torch.zeros(num_tags); b[0] = -10000.0 
        self.start_transitions = nn.Parameter(a)
        self.end_transitions = nn.Parameter(b)
        
        # nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        # nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        init_transitions = torch.zeros(self.num_tags, self.num_tags)
        init_transitions[:,0]   = -10000.0 
        init_transitions[0,:]   = -10000.0 
        
        self.transitions = torch.nn.Parameter(init_transitions)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def loss_function(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        # print(mask[:, 0])
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)  # prob
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask) # forwards_algorithm
        # shape: (batch_size,)
        # llh = numerator - denominator # seriously?
        llh = denominator - numerator 

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, 
               emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:

            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, 
            emissions: torch.Tensor, 
            tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list



class CRFLayer(torch.nn.Module):

    def __init__(self, tag_size, init_method = 'zeros', 
                 PAD_TAG = 0, START_TAG = 1, END_TAG = 2):
        '''
            The Difference between tagset_size and tag_size
            tag_size = tagset_size + 2
            input_size = tagset_size
        '''
        super(CRFLayer, self).__init__()

        self.tag_size = tag_size # including pad, start, end: 0, 1, 2
        

        if init_method == 'zeros':
            init_transitions = torch.zeros(self.tag_size, self.tag_size)
            # print('init with zeros')
        else:
            init_transitions = torch.randn(self.tag_size, self.tag_size)
            # print('init with random')
            
        self.PAD_TAG   = PAD_TAG
        self.START_TAG = START_TAG
        self.END_TAG   = END_TAG
        
        init_transitions[:,START_TAG] = -10000.0
        init_transitions[END_TAG,:]   = -10000.0
        
        if type(self.PAD_TAG) == int:
            init_transitions[:,PAD_TAG]   = -10000.0 
            init_transitions[PAD_TAG,:]   = -10000.0 
        
        self.transitions = torch.nn.Parameter(init_transitions)

    
    # TODO IT in the future in order to decrease the training time
    def set_feats(self, feats, leng):

        tag_size  = self.tag_size
        START_TAG = self.START_TAG
        END_TAG   = self.END_TAG

        packedSeq = torch.nn.utils.rnn.pack_padded_sequence(feats, leng, batch_first = True)
        data = packedSeq.data
        
        valid_leng  = data.size(0)
        batch_sizes = packedSeq.batch_sizes# .numpy()
        # batch_sizes  = list(packedSeq.batch_sizes.numpy()) 
        batch_lengs = torch.cumsum(batch_sizes, 0)
        # batch_lengs  = list(np.cumsum(batch_sizes))

        transitions_all = self.transitions.view(1, tag_size, tag_size).expand(valid_leng, tag_size, tag_size)  
        obs_all         = data.view(valid_leng, 1, tag_size).expand(valid_leng, tag_size, tag_size)
        local_trans_all = transitions_all + obs_all
        
        idex, _ = pad_packed_sequence(torch.tensor(range(leng.sum())), batch_sizes, batch_first=True)

        ends_index = torch.zeros(leng.size(0)).type(leng.type())
        

        for idx, seq in enumerate(idex):
            ends_index[idx] = seq[leng[idx]-1]

        for end_index in ends_index:
            local_trans_all[end_index] = local_trans_all[end_index] - 10000
            local_trans_all[end_index][:, END_TAG ] = local_trans_all[end_index][:, END_TAG ] + 10000
            
    
        self.leng            = leng
        self.batch_sizes     = batch_sizes.type(leng.type())
        self.batch_lengs     = batch_lengs.type(leng.type())
        self.ends_index      = ends_index.type(leng.type())
        self.local_trans_all = local_trans_all
    
    
    def sentence_score(self, feats, tags, leng, use_cache = False, SE_in_feats = True):
        #### Prepare the Data #### 
        # START_TAG = self.START_TAG
        
        tag_size = self.tag_size
        START_TAG = self.START_TAG
        END_TAG = self.END_TAG
        
        if use_cache == False:
            self.set_feats(feats, leng)
        
        # reverse_idx     = self.reverse_idx
        batch_sizes     = self.batch_sizes
        batch_lengs     = self.batch_lengs
        ends_index      = self.ends_index
        local_trans_all = self.local_trans_all
        leng            = self.leng

        # PackedTags, tags_reverse_idx = utils.pack(tags, leng)
        PackedTags = torch.nn.utils.rnn.pack_padded_sequence(tags, leng, batch_first = True)
        tags = PackedTags.data
        # // assert tags_reverse_idx == self.reverse_idx
        
        prob = torch.zeros(batch_sizes[0]).type(feats.type())
        # // curIdx = 0
        # // curStart, curEnd  = 0, batch_lengs[curIdx]
        # // curNSent = batch_sizes[curIdx]
        # // local_trans = local_trans_all[curStart: curEnd]
        # print("=====>", curIdx)
        # print('curStart', curStart, 'curEnd', curEnd)
        # print('curNCont', batch_sizes[curIdx])
        # print('cur Num of Sents:', curNSent)
        # print(local_trans_all[curStart: curEnd]) 
        # print('To  ', tags[curStart : curEnd])
        # print(prob)

        # zero_batch_lengs = np.concatenate([[0], batch_lengs.numpy()])
        # print(batch_lengs)
        # print(batch_lengs.shape)
        zero_batch_lengs = torch.zeros(batch_lengs.size(0) + 1).type(batch_lengs.type())
        # print(zero_batch_lengs.shape)
        zero_batch_lengs[1:] = batch_lengs
        # print(zero_batch_lengs)
        

        for curIdx in range(1, len(batch_sizes)):
            curStart, curEnd  = batch_lengs[curIdx-1], batch_lengs[curIdx]
            lastStart,lastEnd = zero_batch_lengs[1 + curIdx-2], batch_lengs[curIdx-1]
            curNSent = batch_sizes[curIdx]
            local_trans = local_trans_all[curStart: curEnd]
            
            # print("=====>",curIdx)
            # print('lastStart', lastStart, 'lastEnd', lastEnd)
            # print('curStart', curStart, 'curEnd', curEnd)
            # print('cur Num of Sents:', curNSent)
            # print(local_trans_all[curStart: curEnd]) 
            # print('From', tags[lastStart: lastEnd])
            # print('To  ', tags[curStart : curEnd])
            transIdx = torch.cat([tags[lastStart : lastEnd][:curNSent].unsqueeze(1), 
                                  tags[curStart  : curEnd ].unsqueeze(1)], 
                                  dim = 1)
            
            for sentIdx in range(curNSent):
                fromIdx       = transIdx[sentIdx][0]
                toIdx         = transIdx[sentIdx][1]
                prob[sentIdx] = prob[sentIdx] + local_trans[sentIdx][fromIdx, toIdx]
            # print(transIdx)
            # print(prob)

        # // prob = prob[reverse_idx]
        # print('In sentence_score: Prob Score for: prob(tag|feat)')
        # print('prob:\n', prob.type())
        return prob # a batch's scores 
        
    def forward_algorithm(self, feats, leng, use_cache = False):
        #### Prepare the Data #### 

        tag_size = self.tag_size
        START_TAG = self.START_TAG
        END_TAG = self.END_TAG
        
        if use_cache == False:
            self.set_feats(feats, leng)
        
        # reverse_idx     = self.reverse_idx
        batch_sizes     = self.batch_sizes
        ends_index      = self.ends_index
        batch_lengs     = self.batch_lengs
        local_trans_all = self.local_trans_all
        leng            = self.leng
        
        par_history = torch.zeros(batch_lengs[-1], tag_size).type(feats.type())

        # // curIdx = 0
        # // curStart, curEnd = 0, batch_lengs[curIdx]
        # // curNSent  = batch_sizes[curIdx]
        # // par = torch.tensor([[0.] * tag_size] * batch_sizes[0])

        curIdx = 0
        curStart, curEnd = 0, batch_lengs[curIdx]
        # curNSent  = batch_sizes[curIdx]


        par = - torch.ones(batch_sizes[0], tag_size) * 10000.; par = par.type(feats.type())
        par[:, START_TAG] = 0.
        par_history[curStart:curEnd] = par


        # // par_history[curStart:curEnd] = par  # could be commented out, logicall displayed here.
        # print("=====>", curIdx)
        # print('curStart', curStart, 'curEnd', curEnd)
        # print('curNCont', batch_sizes[curIdx])
        # print('cur Num of Sents:', curNSent)
        # print(local_trans_all[curStart: curEnd]) 
        # print(par)

        for curIdx in range(1, len(batch_sizes)):
            curStart, curEnd = batch_lengs[curIdx-1], batch_lengs[curIdx]
            curNSent = batch_sizes[curIdx]
            pre      = par[:curNSent]
            local_trans= local_trans_all[curStart: curEnd]

            # for i in [local_trans, pre, par]:
            #     print(i.type())
            scores = local_trans + pre.view(curNSent, tag_size, 1).expand(curNSent, tag_size, tag_size)
            # par = log_sum_exp(scores)
            par = torch.logsumexp(scores, dim=1)
            par_history[curStart: curEnd]= par
            # print("=====>", curIdx)
            # print('curStart', curStart, 'curEnd', curEnd)
            # print('curNCont', batch_sizes[curIdx])
            # print('cur Num of Sents:', curNSent)
            # print(scores) 
            # print(par)

        final_par = par_history[ends_index][:, END_TAG]
        # // final_par = torch.flip(final_par[reverse_idx])

        if final_par.size(0) != batch_sizes[0]:
            print('Errors in generating final_par')
            print(feats.shape)
            print(final_par.shape)
            print(ends_index)
            print(batch_sizes)

        # final_par = torch.flip(final_par, [0])
        # print('In forward_algorithm: Prob Score for: prob(feat)             (i.e., sum_all_tag-_s prob(tag-|feat)')
        # print(final_par, '\n')
        
        return final_par
                       
    def neg_log_likelihood_loss(self, feats, tags, leng, use_cache = False):

        # In this stage, SE in feats is True
        final_par = self.forward_algorithm(feats, leng, use_cache = False) # not SE_in_feats
        prob = self.sentence_score(feats, tags, leng, use_cache = True)

        forward_score, gold_score = final_par.sum(), prob.sum()
        # llh = final_par - prob
        # loss = llh.sum().item()
        
        # print(final_par.size(), prob.size())
        if forward_score - gold_score < 0:
            print('\n---->Errors Occur:')
            print('Forwards     :', final_par.detach().numpy())
            print('Tag_Score    :', prob.detach().numpy())
            tag_seq, tag_scores = self.viterbi_decode(feats, leng, use_cache = False)
            print('Viterbi_Score:', tag_scores.detach().numpy())

        # for i in [forward_score, gold_score]:
        #     print(i.type())
        return forward_score - gold_score
    
    def viterbi_decode(self, feats, leng, use_cache = False):

        #### Prepare the Data #### 
        tag_size = self.tag_size
        START_TAG = self.START_TAG
        END_TAG = self.END_TAG
        
        if use_cache == False:
            self.set_feats(feats, leng)
        
        # reverse_idx     = self.reverse_idx
        # // valid_leng      = self.valid_leng
        batch_sizes     = self.batch_sizes
        batch_lengs     = self.batch_lengs
        local_trans_all = self.local_trans_all
        leng            = self.leng

        ########################   STAGE 0 ###########################

        # total valid number = batch_lengs[-1]
        bp_history  = torch.zeros(batch_lengs[-1], tag_size).type(leng.type())
        out_history = torch.zeros(batch_lengs[-1], tag_size).type(feats.type())

        #// curIdx = 0
        #// curStart, curEnd = 0, batch_lengs[curIdx]
        #// curNSent  = batch_sizes[curIdx]
        #// out = torch.tensor([[0.] * tag_size] * batch_sizes[0])


        curIdx = 0
        curStart, curEnd = 0, batch_lengs[curIdx]
        # curNSent  = batch_sizes[curIdx]


        out = - torch.ones(batch_sizes[0], tag_size) * 10000.; out = out.type(feats.type())
        out[:, START_TAG] = 0.
        out_history[curStart:curEnd] = out


        #// bp  = torch.tensor([[0]  * tag_size] * batch_sizes[0])
        #// out_history[curStart:curEnd] = out 
        #// bp_history[curStart:curEnd] = bp

        # print("=====>", curIdx)
        # print('curStart', curStart, 'curEnd', curEnd)
        # print('curNCont', batch_sizes[curIdx])
        # print('cur Num of Sents:', curNSent)
        # print(local_trans_all[curStart: curEnd]) 
        # print(out)
        # print(bp)

        for curIdx in range(1, len(batch_sizes)):
            curStart, curEnd = batch_lengs[curIdx-1], batch_lengs[curIdx]
            curNSent = batch_sizes[curIdx]
            pre      = out[:curNSent]
            local_trans = local_trans_all[curStart: curEnd] # [:curNCont]
            scores   = local_trans + pre.view(curNSent, tag_size, 1).expand(curNSent, tag_size, tag_size)
            out, bp  = torch.max(scores, 1)
            out_history[curStart:curEnd] = out 
            bp_history [curStart:curEnd] = bp
            # print("=====>", curIdx)
            # print('curStart', curStart, 'curEnd', curEnd)
            # print('curNCont', batch_sizes[curIdx])
            # print('cur Num of Sents:', curNSent)
            # print(scores) 
            # print(out)
            # print(bp)

        ########################   STAGE 1 ###########################
        # valid_leng = batch_lengs[-1]
        tag_seq   = torch.zeros(batch_lengs[-1]).type(leng.type())
        tag_scores= torch.Tensor([]).type(feats.type())
        lastTag   = torch.Tensor([]).type(leng.type())
        lastNSent = 0
        for idx in range(len(batch_sizes)-1):
            curIdx   = len(batch_sizes) - idx - 1
            curStart, curEnd = batch_lengs[curIdx-1], batch_lengs[curIdx]
            curNSent = batch_sizes[curIdx]
            curNEnd  = curNSent - lastNSent
            # print(idx)
            # print('======', curIdx)
            # print('curStart',curStart, 'curEnd', curEnd)
            # print('curNSent',curEnd-curStart)
            # print('curNEnd', curNEnd)
            # print(out_history[curStart: curEnd])
            # print(bp_history[curStart: curEnd])
            
            curTag = lastTag
            
            if curNEnd > 0:
                # print('----------> For Ends --------------')
                curout = out_history[curStart: curEnd][-curNEnd:]
                # print(curout)
                seq_scores, curTagEnd  = torch.max(curout, 1)
                # print(seq_scores, curTagEnd)
                curTag     = torch.cat([curTag, curTagEnd])
                tag_scores = torch.cat([tag_scores, seq_scores])
                
            
            tag_seq[curStart: curEnd] = curTag
            curbp = bp_history[curStart: curEnd]
            # // assert len(curbp)  == curNSent
            # // assert len(curTag) == curNSent
            # print('----------> For CurTag --------------')
            # print(curTag)
            
            lastTag = curbp.gather(1, curTag.view(curNSent, 1)).view(curNSent)
            # print('--------> For Last Sent Tag -----------')
            # print(lastTag)
            lastNSent = curNSent

        curIdx = 0
        curStart, curEnd = 0, batch_lengs[curIdx]
        # // curNSent = batch_sizes[curIdx]
        # // curNEnd  = curNSent - lastNSent
        # print('curStart',curStart, 'curEnd', curEnd)
        # print('curNSent',curEnd-curStart)
        # print('curNEnd', curNEnd)
        # print(out_history[curStart: curEnd])
        # print(bp_history[curStart: curEnd])
        
        curTag = lastTag
        tag_seq[curStart: curEnd] = curTag 
        # print(curTag)  

        tag_seq, _ = pad_packed_sequence(tag_seq, batch_sizes, batch_first=True)
        # tag_seq    = tag_seq[reverse_idx]
        # tag_scores = tag_scores[reverse_idx]
        # print('Prob Score for: prob(tag|feats)', tag_scores)

        # print('In viterbi_decode: Prob Score for: best prob(tag|feat)')
        # print(tag_scores, '\n')
            
        return tag_seq, tag_scores
    
    def forward(self, feats, leng, use_cache = False):
        tag_seq, tag_scores = self.viterbi_decode(feats, leng, use_cache = use_cache)
        return tag_seq, tag_scores
    