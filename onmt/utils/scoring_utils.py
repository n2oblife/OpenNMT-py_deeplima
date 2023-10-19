import codecs
import os
from onmt.utils.parse import ArgumentParser
from onmt.translate import GNMTGlobalScorer, Translator
from onmt.opts import translate_opts
from onmt.constants import CorpusTask
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice
from onmt.transforms import get_transforms_cls, make_transforms, TransformPipe


class ScoringPreparator:
    """Allow the calculation of metrics via the Trainer's
    training_eval_handler method.
    """

    def __init__(self, vocabs, opt):
        self.vocabs = vocabs
        self.opt = opt
        if self.opt.dump_preds is not None:
            if not os.path.exists(self.opt.dump_preds):
                os.makedirs(self.opt.dump_preds)
        self.transforms = opt.transforms
        transforms_cls = get_transforms_cls(self.transforms)
        transforms = make_transforms(self.opt, transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def warm_up(self, transforms):
        self.transforms = transforms
        transforms_cls = get_transforms_cls(self.transforms)
        transforms = make_transforms(self.opt, transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def ids_to_tokens_batch_side(self, batch, side):
        """Convert a batch into a list of transformed sentences
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
            side (string): 'src' or 'tgt'.
        Returns
            transformed_sentences (list): List of lists of tokens.
                Each list is a transformed sentence.
        """
        vocab = self.vocabs[side]
        batch_side = batch[side]
        nb_sentences = batch_side.shape[0]
        nb_tokens_per_sentence = batch_side.shape[1]
        nb_feats = batch_side.shape[2] - 1

        indices_to_remove = [
            vocab.lookup_token(token)
            for token in [DefaultTokens.PAD, DefaultTokens.EOS, DefaultTokens.BOS]
        ]
        transformed_sentences = []
        for i in range(nb_sentences):
            tokens = [
                vocab.lookup_index(batch_side[i, t, 0])
                for t in range(nb_tokens_per_sentence)
                if batch_side[i, t, 0] not in indices_to_remove
            ]
            transformed_sentences.append(tokens)

        if nb_feats > 0:
            transformed_feats = []
            for i_feat in range(nb_feats):
                fv = self.vocabs["src_feats"][i_feat]
                indices_to_remove = [
                    fv.lookup_token(token)
                    for token in [
                        DefaultTokens.PAD,
                        DefaultTokens.EOS,
                        DefaultTokens.BOS,
                    ]
                ]
                transformed_feat = []
                for i in range(nb_sentences):
                    tokens = [
                        fv.lookup_index(batch_side[i, t, i_feat + 1])
                        for t in range(nb_tokens_per_sentence)
                        if batch_side[i, t, i_feat + 1] not in indices_to_remove
                    ]
                    transformed_feat.append(tokens)
                transformed_feats.append(transformed_feat)
        else:
            transformed_feats = [repeat(None)]

        return transformed_sentences, transformed_feats

    def ids_to_tokens_batch(self, batch):
        """Reconstruct transformed source and reference
        sentences from a batch.
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
        Returns:
            transformed_batch(list): A list of examples
        with the fields "src" and "tgt"
        """
        transformed_srcs, transformed_src_feats = self.ids_to_tokens_batch_side(
            batch, "src"
        )
        transformed_tgts, _ = self.ids_to_tokens_batch_side(batch, "tgt")
        transformed_batch = []
        for src, tgt, *src_feats in zip(
            transformed_srcs, transformed_tgts, *transformed_src_feats
        ):
            ex = {"src": src, "tgt": tgt}
            if src_feats[0] is not None:
                ex["src_feats"] = src_feats
            transformed_batch.append(ex)

        return transformed_batch

    def translate(self, model, transformed_batches, gpu_rank, step, mode):
        """Compute and save the sentences predicted by the
        current model's state related to a batch.

        Args:
            model (:obj:`onmt.models.NMTModel`): The current model's state.
            transformed_batches(list of lists): A list of transformed batches.
            gpu_rank (int): Ordinal rank of the gpu where the
                translation is to be done.
            step: The current training step.
            mode: (string): 'train' or 'valid'.
        Returns:
            preds (list): Detokenized predictions
            texts_ref (list): Detokenized target sentences
        """
        # ########## #
        # Translator #
        # ########## #

        # Set translation options
        parser = ArgumentParser()
        translate_opts(parser)
        base_args = ["-model", "dummy"] + ["-src", "dummy"]
        opt = parser.parse_args(base_args)
        opt.gpu = gpu_rank
        ArgumentParser.validate_translate_opts(opt)

        # Build translator from options
        scorer = GNMTGlobalScorer.from_opt(opt)
        out_file = codecs.open(os.devnull, "w", "utf-8")
        model_opt = self.opt
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        translator = Translator.from_opt(
            model,
            self.vocabs,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=False,
            logger=None,
        )

        # ################### #
        # Validation iterator #
        # ################### #

        # Reinstantiate the validation iterator

        transforms_cls = get_transforms_cls(model_opt._all_transform)
        model_opt.num_workers = 0
        model_opt.tgt = None

        valid_iter = build_dynamic_dataset_iter(
            model_opt,
            transforms_cls,
            translator.vocabs,
            task=CorpusTask.VALID,
            copy=model_opt.copy_attn,
        )

        # Retrieve raw references and sources
        with codecs.open(
            valid_iter.corpora_info["valid"]["path_tgt"], "r", encoding="utf-8"
        ) as f:
            raw_refs = [line.strip("\n") for line in f if line.strip("\n")]
        with codecs.open(
            valid_iter.corpora_info["valid"]["path_tgt"], "r", encoding="utf-8"
        ) as f:
            raw_srcs = [line.strip("\n") for line in f if line.strip("\n")]

        valid_iter = IterOnDevice(valid_iter, opt.gpu)

        # ########### #
        # Predictions #
        # ########### #

        _, preds = translator._translate(
            valid_iter,
            transform=valid_iter.transform,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug,
        )

        # ####### #
        # Outputs #
        # ####### #

        # Flatten predictions
        preds = [x.lstrip() for sublist in preds for x in sublist]

        # Save results
        if len(preds) > 0 and self.opt.scoring_debug:
            path = os.path.join(self.opt.dump_preds, f"preds.valid_step_{step}.txt")
            with open(path, "a") as file:
                for i in range(len(preds)):
                    file.write("SOURCE: {}\n".format(raw_srcs[i]))
                    file.write("REF: {}\n".format(raw_refs[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        return preds, raw_refs
