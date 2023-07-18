from .scorer import Scorer
from onmt.scorers import register_scorer


@register_scorer(metric="ACC")
class AccScorer(Scorer):
    """Accuracy scorer class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)
    
    def compute_accuracy(preds, text_refs):
        assert( len(preds) == len(text_refs), 
               "Preds en Refs don't have same size")
        n_words = len(preds)
        n_correct = 0
        for pred,ref in preds, text_refs:
            if pred == ref:
                n_correct += 1
        return 100*(n_correct / n_words)

    def compute_score(self, preds, texts_refs):
        if len(preds) > 0:
            score = self.compute_accuracy(preds, texts_refs)
        else:
            score = 0
        return score
