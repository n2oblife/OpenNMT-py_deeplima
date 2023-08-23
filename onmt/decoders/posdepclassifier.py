import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt
import onmt.decoders as dec

from trankit.utils.base_utils import *
from trankit.utils.conll import *
from trankit.utils.chuliu_edmonds import chuliu_edmonds_one_root
from trankit.models import Deep_Biaffine
from trankit.tpipeline import TPipeline, PosDepClassifier
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from ..utils.misc import TConfig

@dec.register_decoder(name='posdep')
class PosDepDecoder(dec.DecoderBase, TPipeline):
    def __init__(self, opt:dict,embeddings = None, attentional=True) -> None:
        dec.DecoderBase.__init__(self, attentional)
        self.copy_attn = opt.copy_attn
        # TODO
        # trankit's config for TPipeline, parameters are : self._param
        # can be optimized by init only posdep and useful elements
        TPipeline.__init__(self, training_config=TConfig(opt).get_default_tconfig())
        # tconfig = TConfig(opt)
        # self.posdep = PosDepClassifier(tconfig) # should optimize the init time
        # or could create a new class adapted from trankit's PosDepClassifier 
        self.embeddings = embeddings # see if needed or overwrite trankit's embeddings

    
    @classmethod
    def from_opt(cls, opt, embeddings, attentional=True):
        return cls(opt, embeddings, attentional)
    
    def forward(self, batch:torch.tensor, word_reprs:torch.tensor, cls_reprs:torch.tensor):
        """A function wich returns the pos and dep relations from the
        embeddings of xlmr (embedding could be used with only the 
        decoder)

        Args:
            batch (tensor): _description_
            word_reprs (tensor): _description_
            cls_reprs (tensor): _description_

        Returns:
            Tuple[list,list]: hte whole descriptions, with the output and raw output of the dependencies
            for loss calculation
        """
        predictions = self._tagger.predict(batch, word_reprs, cls_reprs)
        preds = self.preds_to_cpu(predictions)
        deps, deps_idxs = self.deprel_to_deps(predictions[3], batch)
        preds.append(deps)
        deps_idxs = self.padding_deps(deps_idxs)
        return preds, deps_idxs
    
    def preds_to_cpu(self, predictions):
        predicted_upos = predictions[0]
        predicted_xpos = predictions[1]
        predicted_feats = predictions[2]
        
        predicted_upos = predicted_upos.data.cpu().numpy().tolist()
        predicted_xpos = predicted_xpos.data.cpu().numpy().tolist()
        predicted_feats = predicted_feats.data.cpu().numpy().tolist()
        return [predicted_upos, predicted_xpos, predicted_feats]

    def deprel_to_deps(self, predicted_dep, batch):
        """Function wich transform the deprel into deps wich is an 
        enhanced deps with deprel and heads

        Args:
            predicted_dep (_type_): _description_
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size = len(batch.word_num)
        # head, deprel
        sentlens = [l + 1 for l in batch.word_num]
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] 
                     for adj, l in zip(predicted_dep[0], sentlens)]
        deprel_seqs = [[self._config.itos[DEPREL][predicted_dep[1][i][j + 1][h]] 
                        for j, h in enumerate(hs)] 
                        for i, hs in enumerate(head_seqs)]
        deprel_seqs_idxs = [[predicted_dep[1][i][j + 1][h] for j, h in enumerate(hs)] 
                            for i, hs in enumerate(head_seqs)]
        pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] 
                       for i in range(batch_size)]
        return pred_tokens, deprel_seqs_idxs
    
    def padding_deps(self, dep):
        max_len = max([len(sentences) for sentences in dep])
        for sent in dep:
            for _ in range(max_len - len(sent)):
                sent.append(0) # 0 is the padding value
        return dep
    
    def loss(self, batch, word_reprs, cls_reprs, predicted):
        """Loss computation adapted from trankit's training to be compatible
        with the OpenNMT-py training

        Args:
            batch (_type_): _description_
            word_reprs (_type_): _description_
            cls_reprs (_type_): _description_
            predicted (_type_): predicted is the output of the decoder
            (1-upos, 2-xpos, 3-features, 4-deps) / here forcusing on deprel

        Returns:
            _type_: _description_
        """
        batch_size = len(batch.word_num)
        loss = self._tagger(batch, word_reprs, cls_reprs)
        pred = predicted.contiguous().view(-1)
        tgt = batch.deprel_idxs.contiguous().view(-1)        
        stats = self.stats(batch_size, loss, pred, tgt)
        return loss, stats
    
    def stats(self, bsz, loss, pred, target):
        """ Builds the statistics of the model
        Args:
            loss (int): the loss computed by the loss criterion.
            pred (:obj:`FloatTensor`): the prediction for deps, needed flatten
            target (:obj:`FloatTensor`): true targets, needed flatten

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        non_padding = target.ne(0) # 0 is the padding value
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        n_batchs = 1 if bsz else 0
        # in the case criterion reduction is None then we need
        # to sum the loss of each sentence in the batch
        return onmt.utils.Statistics(
            loss=loss,
            n_batchs=n_batchs,
            n_sents=bsz,
            n_words=num_non_padding,
            n_correct=num_correct,
        )
    
    def to_onmt_batch(self, batch, full=False):
        """A function which turns batches from the trankit library 
        to batches inspired from the onmt library

        Args:
            batch (_type_): batch from the trankit iterator
            full (bool, optional) : Defaults to False. A value to copy
            the whole batch

        Returns:
            dict: batch inspired from the onmt batch that should
            be used in the satistics functions
        """
        onmt_batch = {}
        # elts from the onmt's batch
        onmt_batch['src'] = batch.words
        onmt_batch['indices'] = batch.piece_idxs
        onmt_batch['srclen'] = batch.word_num
        onmt_batch['tgt'] = batch.deprel_idxs
        onmt_batch['tgtlen'] = batch.word_num
        if full :
            # elts from the trankit's batch
            onmt_batch['sent_index'] = batch.sent_index
            onmt_batch['word_ids'] = batch.word_ids
            onmt_batch['word_span_idxs'] = batch.word_span_idxs
            onmt_batch['attention_mask'] = batch.attention_mask
            onmt_batch['word_lens'] = batch.word_lens
            onmt_batch['edit_type_idxs'] = batch.edit_type_idxs
            onmt_batch['upos_type_idxs'] = batch.upos_type_idxs
            onmt_batch['xpos_type_idxs'] = batch.xpos_type_idxs
            onmt_batch['feats_type_idxs'] = batch.feats_type_idxs
            onmt_batch['upos_ids'] = batch.upos_ids
            onmt_batch['xpos_ids'] = batch.xpos_ids
            onmt_batch['feats_ids'] = batch.feats_ids
            onmt_batch['head_idxs'] = batch.head_idxs
            onmt_batch['deprel_idxs'] = batch.deprel_idxs
            onmt_batch['word_mask'] = batch.word_mask
        return onmt_batch

    def validation_metrics():
        return NotImplemented