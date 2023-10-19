from typing import Any
import torch
from onmt.decoders.decoder import DecoderBase
from transformers.adapters import XLMRobertaAdapterModel, AdapterConfig

class XlmrAdaptedDecoder(DecoderBase):
    def __init__(self, attentional=True):
        super(XlmrAdaptedDecoder,self).__init__(attentional)
        self._decoder = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
        self._adapter_config = AdapterConfig.load("pfeiffer")
        # TODO change adapter initialization
        self._decoder.add_adapter("posdep_english_adapter", self._adapter_config)


    def forward(self, inputs : str | list[str]) -> torch.tensor:
        return self._decoder(**inputs, return_tensor = 'pt')

    def update_dropout(self, dropout):
        self._decoder = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base",
                                                               dropout = dropout)
        self._decoder.add_adapter("posdep_english_adapter", self._adapter_config,
                                  dropout=dropout)
    
    def add_adapter(self, task : str = 'Dep', lgge : str = 'English') -> None:
        self._adapter_config = AdapterConfig.load("pfeiffer") 
        self._task = task
        self._lgge = lgge

        self._adapter_name = 'deeplima_adapters_'+task+'_'+lgge
        self._decoder.add_adapter(self._adapter_name)

    def add_classification_head(self, adapter_name : str, num_labels : int, id2label : dict) -> None :
        self._decoder.add_classification_head(adapter_name,num_labels, id2label)
    
    def set_active_adapters(self, adapter_name : str) -> None:
        self._decoder.train()
        self._decoder.set_active_adapters(adapter_name)
    
    def has_adapters(self) -> bool:
        return self._decoder.has_adapters()
    
    def save_pretrained(self, path : str) -> None:
        self._decoder.save_pretrained(path)
    
    def save_adapter(self, path : str) -> None :
        self._decoder.save_adapter(path)

 # ------------------------------------------------------------------------   

    
    