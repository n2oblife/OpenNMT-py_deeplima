import os
import torch
import torch.nn as nn
import onmt.encoders as enc

from transformers.adapters import AdapterConfig, AdapterType, XLMRobertaAdapterModel
from trankit.utils.base_utils import word_lens_to_idxs_fast
from trankit.iterators.tagger_iterators import Batch
from onmt.utils.misc import use_gpu
from onmt.utils.loss import LossCompute
from torch import Tensor


@enc.register_encoder(name='xlmr')
class XLMREncoder(enc.EncoderBase) :
    # TODO add mother class XLMRobertaAdapterModel to avoid having xlmr attribute
    def __init__(self, opt:dict, embeddings = None) -> None:
        super(XLMREncoder, self).__init__()
        # self._adapter_name = opt.task+'_'+opt.treebank_name
        self._adapter_name = opt.task+opt.treebank_name # TODO choose if _ or not
        # xlmr encoder
        self.xlmr_dim = 768 if opt.embedding_name == 'xlm-roberta-base' else 1024
        self.xlmr = XLMRobertaAdapterModel.from_pretrained(opt.embedding_name,
                                                    output_hidden_states=True)
        self.xlmr_dropout = nn.Dropout(p=opt.dropout[0])
        # add task adapters
        task_config = AdapterConfig.load("pfeiffer",
                                         reduction_factor=6 if opt.embedding_name == 'xlm-roberta-base' else 4)
        self.xlmr.add_adapter(adapter_name=self._adapter_name, config=task_config)
        self.xlmr.train_adapter(self._adapter_name)
        self.xlmr.set_active_adapters(self._adapter_name)
        if use_gpu(opt) :
            self.xlmr.to("cuda")
        self.embeddings = embeddings  # TODO : see if needed for the model type seq2seq
    
    @classmethod
    def from_opt(cls, opt:dict, embeddings=None):
        """Alternate constructor from the option file"""
        return cls(opt, embeddings)
    
    def update_dropout(self, dropout:float, attention_dropout:int):
        self.xlmr_dropout = nn.Dropout(p = dropout)
        for layer in self.xlmr :
            layer.update_dropout(dropout, attention_dropout)

    def forward(self, src:Batch, src_len:list[list[int]] = None):
        # encoding by embedding
        if src_len is None :
            src_len = src.word_lens
        word_reprs, cls_reprs = self.encode_words(
            piece_idxs= src.piece_idxs,
            attention_masks= src.attention_masks,
            word_lens= src_len
            )
        return word_reprs, cls_reprs

    def encode_words(self, piece_idxs:Tensor, attention_masks:Tensor, word_lens:Tensor):
        if not isinstance(piece_idxs, Tensor):
            piece_idxs = torch.tensor(piece_idxs)
        if not isinstance(attention_masks, Tensor):
            attention_masks = torch.tensor(attention_masks)
        # if not isinstance(word_lens, Tensor):
        #     word_lens = torch.tensor(word_lens)
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0] # [batch_size, word_lens, xlmr_dim]
        cls_reprs = xlmr_outputs[:, 0, :].unsqueeze(1)  # [batch size, 1, xlmr dim]

        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1,
                                    idxs) * masks
        xlmr_outputs = xlmr_outputs.view(batch_size, token_num, token_len, self.xlmr_dim)
        xlmr_outputs = xlmr_outputs.sum(2)
        return xlmr_outputs, cls_reprs

    def get_tokenizer_inputs(self, batch):
        wordpiece_reprs = self.encode(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks
        )
        return wordpiece_reprs
    
    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]

        wordpiece_reprs = xlmr_outputs[:, 1:-1, :]  # [batch size, max input length - 2, xlmr dim]
        return wordpiece_reprs
        
    def batch_iterator(train_set):
        """_summary_

        Args:
            train_set (DataLoader): training set from the trankit's script after
            passing through a dataloader

        Yields:
            iterator: an iterator adapted for the normalization due to 
            the graddient accumulation
        """
        for batch in train_set:
            yield batch, 1