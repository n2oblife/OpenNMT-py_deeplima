""" 
Implementation of the NllbEncoder to be used 
as interface by onmt trainer
"""

from transformers import NllbTokenizer
from onmt.encoders.encoder import EncoderBase

class NllbEncoder(EncoderBase):
    def __init__(self,embeding) -> None:
        """_summary_

        Args:
            embeding (_type_): _description_
        """
        super(NllbEncoder, self).__init__()
        self._embeding = embeding
        self._tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    def forward(self, src: str | list[str], batch_size : int | None):
        return self._tokenizer(src), None, batch_size   

    def update_dropout(self, dropout, attention_dropout):
        self._tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", 
                                                        dropout = dropout, 
                                                        attention_dropout = attention_dropout)
            