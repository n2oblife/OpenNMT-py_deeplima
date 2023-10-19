from .scorer import Scorer
from onmt.scorers import register_scorer
import torch.nn as nn


@register_scorer(metric="L2")
class L2LossScorer(Scorer):
    """L2 Loss scorer class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)
    
    def compute_loss(preds, text_refs):
        cl_loss = nn.MSELoss()
        return cl_loss(preds, text_refs)

    def compute_score(self, preds, texts_refs):
        if len(preds) > 0:
            score = self.compute_loss(preds, texts_refs)
        else:
            score = 0
        return score
