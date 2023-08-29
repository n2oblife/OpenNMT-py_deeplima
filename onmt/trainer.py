"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import time
import torch
import traceback
import onmt.utils
from torch.utils.data import DataLoader
from onmt.utils.loss import LossCompute
from onmt.utils.logging import logger
from onmt.utils.scoring_utils import ScoringPreparator
from onmt.scorers import get_scorers_cls, build_scorers
from tqdm import tqdm

def build_trainer(opt, device_id, model, vocabs, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    train_loss = LossCompute.from_opts(opt, model, vocabs["tgt"])
    valid_loss = LossCompute.from_opts(opt, model, vocabs["tgt"], train=False)

    scoring_preparator = ScoringPreparator(vocabs=vocabs, opt=opt)
    validset_transforms = opt.data.get("valid", {}).get("transforms", None)
    if validset_transforms:
        scoring_preparator.warm_up(validset_transforms)
    scorers_cls = get_scorers_cls(opt.train_metrics)
    train_scorers = build_scorers(opt, scorers_cls)
    scorers_cls = get_scorers_cls(opt.valid_metrics)
    valid_scorers = build_scorers(opt, scorers_cls)

    trunc_size = opt.truncated_decoder  # Badly named...
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    attention_dropout = opt.attention_dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = -1
        n_gpu = 0

    earlystopper = (
        onmt.utils.EarlyStopping(
            opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)
        )
        if opt.early_stopping > 0
        else None
    )

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(
        model,
        train_loss,
        valid_loss,
        scoring_preparator,
        train_scorers,
        valid_scorers,
        optim,
        trunc_size,
        norm_method,
        accum_count,
        accum_steps,
        n_gpu,
        gpu_rank,
        opt.train_eval_steps,
        report_manager,
        with_align=True if opt.lambda_align > 0 else False,
        model_saver=model_saver if gpu_rank <= 0 else None,
        average_decay=average_decay,
        average_every=average_every,
        model_dtype=opt.model_dtype,
        earlystopper=earlystopper,
        dropout=dropout,
        attention_dropout=attention_dropout,
        dropout_steps=dropout_steps,
    )
    return trainer


class Trainer(object):
    """Class that controls the training process.

    Args:
        model(:py:class:`onmt.models.model.NMTModel`): model to train
        train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
          training loss computation
        valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
          training loss computation
        scoring_preparator(:obj:`onmt.translate.utils.ScoringPreparator`):
          preparator for the calculation of metrics via the
          training_eval_handler method
        train_scorers (dict): keeps in memory the current values
          of the training metrics
        valid_scorers (dict): keeps in memory the current values
          of the validation metrics
        optim(:obj:`onmt.utils.optimizers.Optimizer`):
          the optimizer responsible for update
        trunc_size(int): length of truncated back propagation
          through time
        accum_count(list): accumulate gradients this many times.
        accum_steps(list): steps for accum gradients changes.
        n_gpu (int): number of gpu.
        gpu_rank (int): ordinal rank of the gpu in the list.
        train_eval_steps (int): process a validation every x steps.
        report_manager(:obj:`onmt.utils.ReportMgrBase`):
          the object that creates reports, or None
        with_align (bool): whether to jointly lear alignment
          (Transformer)
        model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
          used to save a checkpoint.
          Thus nothing will be saved if this parameter is None.
        average_decay (float): cf opt.average_decay
        average_every (int): average model every x steps.
        model_dtype (str): fp32 or fp16.
        earlystopper (:obj:`onmt.utils.EarlyStopping`): add early
          stopping mecanism
        dropout (float): dropout value in RNN or FF layers.
        attention_dropout (float): dropaout in attention layers.
        dropout_steps (list): dropout values scheduling in steps."""

    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        scoring_preparator,
        train_scorers,
        valid_scorers,
        optim,
        trunc_size=0,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_rank=1,
        train_eval_steps=200,
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype="fp32",
        earlystopper=None,
        dropout=[0.3],
        attention_dropout=[0.1],
        dropout_steps=[0],
    ):
        # Basic attributes.

        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss

        self.scoring_preparator = scoring_preparator
        self.train_scorers = train_scorers
        self.valid_scorers = valid_scorers
        self.optim = optim
        self.trunc_size = trunc_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.train_eval_steps = train_eval_steps
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.dropout_steps = dropout_steps

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0

        # Set model in training mode.
        self.model.train()

    def _training_eval_handler(self, scorer, preds, texts_ref):
        """Trigger metrics calculations

        Args:
            scorer (:obj:``onmt.scorer.Scorer``): scorer.
            preds, texts_ref: outputs of the scorer's `translate` method.

        Returns:
            The metric calculated by the scorer."""

        return scorer.compute_score(preds, texts_ref)

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i], self.attention_dropout[i])
                logger.info(
                    "Updated dropout/attn dropout to %f %f at step %d"
                    % (self.dropout[i], self.attention_dropout[i], step)
                )

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                # adapted for trankit's training
                # TODO : see the normalization role in the loss ?
                try :
                    num_tokens = (
                        batch["tgt"][:, 1:, 0].ne(self.train_loss.padding_idx).sum()
                    )
                    normalization += num_tokens.item()
                    normalization -= len(batch["tgt"])  # don't count for EOS
                except :
                    normalization += sum(batch.word_num)
            else:
                # adapted for trankit's training
                try :
                    normalization += len(batch["tgt"])
                except :
                    normalization += sum(batch.word_num)
            if len(batches) == self.accum_count:
                yield (batches, normalization)
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield (batches, normalization)

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [
                params.detach().float() for params in self.model.parameters()
            ]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(
                enumerate(self.moving_average), self.model.parameters()
            ):
                self.moving_average[i] = (
                    1 - average_decay
                ) * avg + cpt.detach().float() * average_decay

    def train(
        self,
        train_iter,
        train_steps,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
        trankit = False,
        epoch = 1
    ):
        """The main training loop by iterating over ``train_iter`` and possibly
        running validation on ``valid_iter``.

        Args:
            train_iter: An iterator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            :obj:``nmt.Statistics``: training loss statistics"""

        if valid_iter is None:
            logger.info("Start training loop without validation...")
            valid_stats = None
        else:
            logger.info(
                "Start training loop and validate every %d steps...", valid_steps
            )
        logger.info("Scoring with: {}".format(self.scoring_preparator.transform))

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        # Let's clean the GPUs before training loop
        torch.cuda.empty_cache()

        # adapted for trankit
        if trankit :
            train_dataloader = DataLoader(train_iter, batch_size=self.model.decoder._config.batch_size,
                                        shuffle=True, collate_fn=train_iter.collate_fn)
            valid_iterator = DataLoader(valid_iter, batch_size=self.model.decoder._config.batch_size,
                                            shuffle=False, collate_fn=valid_iter.collate_fn)
        else:
            train_iterator = self._accum_batches(train_iter)
            epoch = 1

        
        for ep in range(epoch):
            if trankit :
                train_iterator = ((batch,1) for batch in train_dataloader) # generator in comprehesion
            progress = tqdm(total=int(len(train_iter)/self.model.decoder._config.batch_size)+1, 
                            ncols=75, desc=f'Train epoch {ep+1}')
            for i,(batches, normalization) in enumerate(train_iterator):
                step = self.optim.training_step 
                progress.update(1)
                # UPDATE DROPOUT
                self._maybe_update_dropout(step)

                if self.n_gpu > 1:
                    normalization = sum(
                        onmt.utils.distributed.all_gather_list(normalization)
                    )
                # trankit arg added for the adaptation
                self._gradient_accumulation(
                    batches, normalization, total_stats, report_stats, trankit
                )

                #testing step
                self.optim.step()

                if self.average_decay > 0 and i % self.average_every == 0:
                    self._update_average(step)

                # TODO manage the div by 0
                # report_stats = self._maybe_report_training(
                #     step, train_steps, self.optim.learning_rate(), report_stats
                # )

                if (
                    valid_iterator is not None
                    and step % valid_steps == 0
                    and self.gpu_rank <= 0
                ):
                    valid_stats = self.validate(
                        valid_iterator, moving_average=self.moving_average, 
                        trankit=trankit
                    )

                # TODO manage the div by 0
                # if step % valid_steps == 0 and self.gpu_rank <= 0:
                #     self._report_step(
                #         self.optim.learning_rate(),
                #         step,
                #         valid_stats=valid_stats,
                #         train_stats=total_stats,
                #     )

                #     # Run patience mechanism
                #     if self.earlystopper is not None:
                #         self.earlystopper(valid_stats, step)
                #         # If the patience has reached the limit, stop training
                #         if self.earlystopper.has_stopped():
                #             logger.info("earlystopper has_stopped!")
                #             break
            
                if not(trankit) :
                    if self.model_saver is not None and (
                        save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0
                    ):
                        self.model_saver.save(step, moving_average=self.moving_average)

                    if train_steps > 0 and step >= train_steps:
                        break
    
                break
            progress.close()
            print(f"step : {step}")
        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average,
                                  epoch=epoch)
        return total_stats

    def validate(self, valid_iter, moving_average=None, trankit=False):
        """Validate model.

        Args:
            valid_iter: validate data iterator

        Returns:
            :obj:``nmt.Statistics``: validation loss statistics"""

        if not(trankit) :
            valid_model = self.model
            if moving_average:
                # swap model params w/ moving average
                # (and keep the original parameters)
                model_params_data = []
                for avg, param in zip(self.moving_average, valid_model.parameters()):
                    model_params_data.append(param.data)
                    param.data = (
                        avg.data.half() if param.dtype == torch.float16 else avg.data
                    )

            # Set model in validating mode.
            valid_model.eval()

            transformed_batches = []
            with torch.no_grad():
                stats = onmt.utils.Statistics()
                start = time.time()
                for batch in valid_iter:
                    src = batch["src"]
                    src_len = batch["srclen"]
                    tgt = batch["tgt"]
                    if self.valid_scorers:
                        transformed_batch = self.scoring_preparator.ids_to_tokens_batch(
                            batch
                        )
                        transformed_batches.append(transformed_batch)
                    with torch.cuda.amp.autocast(enabled=self.optim.amp):
                        # F-prop through the model.
                        model_out, attns = valid_model(
                            src, tgt, src_len, with_align=self.with_align
                        )

                        # Compute loss.
                        _, batch_stats = self.valid_loss(batch, model_out, attns)

                        stats.update(batch_stats)
                logger.info(
                    """valid stats calculation and sentences rebuilding
                            took: {} s.""".format(
                        time.time() - start
                    )
                )

                # Compute validation metrics (at batch.dataset level)
                if len(self.valid_scorers) > 0:
                    computed_metrics = {}
                    start = time.time()
                    preds, texts_ref = self.scoring_preparator.translate(
                        model=self.model,
                        transformed_batches=transformed_batches,
                        gpu_rank=self.gpu_rank,
                        step=self.optim.training_step,
                        mode="valid",
                    )
                    logger.info(
                        """The translation of the valid dataset
                                    took : {} s.""".format(
                            time.time() - start
                        )
                    )
                for i, metric in enumerate(self.valid_scorers):
                    logger.info("UPDATING VALIDATION {}".format(metric))
                    self.valid_scorers[metric]["value"] = self._training_eval_handler(
                        scorer=self.valid_scorers[metric]["scorer"],
                        preds=preds,
                        texts_ref=texts_ref,
                    )
                    computed_metrics[metric] = self.valid_scorers[metric]["value"]
                    logger.info(
                        "validation {}: {}".format(
                            metric, self.valid_scorers[metric]["value"]
                        )
                    )
                    # Compute stats
                    metric_stats = onmt.utils.Statistics(0, 0, 0, 0, 0, computed_metrics)

                    # Update statistics.
                    stats.update(metric_stats)

                if moving_average:
                    for param_data, param in zip(model_params_data, self.model.parameters()):
                        param.data = param_data

                # Set model back to training mode.
                valid_model.train()
        else :
            # TODO adapter l'evaluation jusqu'au bout et revoir la fonction qui transforme les batchs
            valid_model = self.model
            if moving_average:
                # swap model params w/ moving average
                # (and keep the original parameters)
                model_params_data = []
                for avg, param in zip(self.moving_average, valid_model.parameters()):
                    model_params_data.append(param.data)
                    param.data = (
                        avg.data.half() if param.dtype == torch.float16 else avg.data
                    )
            
            # Set model in validation mode.
            valid_model.eval()

            with torch.no_grad():
                stats = onmt.utils.Statistics()
                start = time.time()
                self.model.decoder._prepare_data_and_vocabs()
                logger.info(
                    """valid stats calculation and sentences rebuilding
                            took: {} s.""".format(
                        time.time() - start
                    )
                )

                # Compute validation metrics (at batch.dataset level)
                if len(self.valid_scorers) > 0:
                    computed_metrics = {}
                    preds, texts_ref = [], []
                    start = time.time()
                    for batch in valid_iter:
                        with torch.cuda.amp.autocast(enabled=self.optim.amp):
                            # trankit's processsing, see trankit's forward code
                            word_reprs, cls_reprs = self.model.encoder(batch)
                            # the goal is to train the deps
                            # TODO encapsulate these steps to enable other tasks
                            _, deps_pred_idxs = self.model.decoder(batch, word_reprs, cls_reprs)
                            _, batch_stats = self.model.decoder.loss(batch, 
                                                                    word_reprs, 
                                                                    cls_reprs,
                                                                    torch.tensor(deps_pred_idxs)
                                                                    )
                            # Here the goal is to study the deprel
                            # text_ref = [" ".join(w) for w in [words for words in batch.words]]
                            preds.append(deps_pred_idxs)
                            texts_ref.append(batch.deprel_idxs)
                            stats.update(batch_stats)

                    logger.info(
                        """The translation of the valid dataset
                                    took : {} s.""".format(
                            time.time() - start
                        )
                    )

                for i, metric in enumerate(self.valid_scorers):
                    logger.info("UPDATING VALIDATION {}".format(metric))
                    self.valid_scorers[metric]["value"] = self._training_eval_handler(
                        scorer=self.valid_scorers[metric]["scorer"],
                        preds=preds,
                        texts_ref=texts_ref,
                    )
                    computed_metrics[metric] = self.valid_scorers[metric]["value"]
                    logger.info(
                        "validation {}: {}".format(
                            metric, self.valid_scorers[metric]["value"]
                        )
                    )
                    # Compute stats
                    metric_stats = onmt.utils.Statistics(0, 0, 0, 0, 0, computed_metrics)

                    # Update statistics.
                    stats.update(metric_stats)

                if moving_average:
                    for param_data, param in zip(model_params_data, self.model.parameters()):
                        param.data = param_data

                # Set model back to training mode.
                valid_model.train()

        return stats

    def _gradient_accumulation(
        self, true_batches, normalization, total_stats, report_stats, 
        trankit=False
    ):
        """Function that iterates over big batches = ``true_batches``

        Perform a backward on the loss of each sub_batch and
        finally update the params at the end of the big batch."""

        if not(trankit) : 
            # original processing
            if self.accum_count > 1:
                self.optim.zero_grad(set_to_none=True)

            for k, batch in enumerate(true_batches):
                target_size = batch["tgt"].size(1)
                # Truncated BPTT: reminder not compatible with accum > 1
                if self.trunc_size:
                    trunc_size = self.trunc_size
                else:
                    trunc_size = target_size

                src = batch["src"]
                src_len = batch["srclen"]
                if src_len is not None:
                    report_stats.n_src_words += src_len.sum().item()
                    total_stats.n_src_words += src_len.sum().item()

                tgt_outer = batch["tgt"]

                bptt = False
                for j in range(0, target_size - 1, trunc_size):
                    # 1. Create truncated target.

                    tgt = tgt_outer[:, j : j + trunc_size, :]

                    # 2. F-prop all but generator.
                    if self.accum_count == 1:
                        self.optim.zero_grad(set_to_none=True)

                    try:
                        with torch.cuda.amp.autocast(enabled=self.optim.amp):
                            model_out, attns = self.model(
                                src, tgt, src_len, bptt=bptt, with_align=self.with_align
                            )
                            bptt = True

                            # 3. Compute loss.
                            loss, batch_stats = self.train_loss(
                                batch,
                                model_out,
                                attns,
                                trunc_start=j,
                                trunc_size=trunc_size,
                            )

                        step = self.optim.training_step
                        if self.train_scorers != {} and step % self.train_eval_steps == 0:
                            # Compute and save stats
                            computed_metrics = {}
                            transformed_batch = self.scoring_preparator.ids_to_tokens_batch(
                                batch
                            )
                            preds, texts_ref = self.scoring_preparator.translate(
                                model=self.model,
                                transformed_batches=[transformed_batch],
                                gpu_rank=self.gpu_rank,
                                step=self.optim.training_step,
                                mode="train",
                            )
                            for i, metric in enumerate(self.train_scorers):
                                logger.info("UPDATING TRAINING {}".format(metric))
                                self.train_scorers[metric][
                                    "value"
                                ] = self._training_eval_handler(
                                    scorer=self.train_scorers[metric]["scorer"],
                                    preds=preds,
                                    texts_ref=texts_ref,
                                )
                                logger.info(
                                    "training {}: {}".format(
                                        metric, self.train_scorers[metric]["value"]
                                    )
                                )
                                computed_metrics[metric] = self.train_scorers[metric][
                                    "value"
                                ]
                            batch_stats.computed_metrics = computed_metrics

                        if loss is not None:
                            loss /= normalization
                            self.optim.backward(loss)

                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)

                    except Exception as exc:
                        trace_content = traceback.format_exc()
                        if "CUDA out of memory" in trace_content:
                            logger.info(
                                "Step %d, cuda OOM - batch removed",
                                self.optim.training_step,
                            )
                            torch.cuda.empty_cache()
                        else:
                            traceback.print_exc()
                            raise exc

                    # If truncated, don't backprop fully.
                    if self.model.decoder.state != {}:
                        self.model.decoder.detach_state()

            # in case of multi step gradient accumulation,
            # update only after accum batches
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(self.n_gpu)
                )
            self.optim.step()
        # adapted for trankit, will need optimization later
        else :
            report_stats.n_src_words += sum([len(word_ids) for word_ids in true_batches.word_ids])
            total_stats.n_src_words += sum([len(word_ids) for word_ids in true_batches.word_ids])

            self.optim.zero_grad()
            try :
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # trankit's processing; see the training code of posdep in tpipeline class
                    word_reprs, cls_reprs = self.model.encoder(true_batches)
                    preds, deps_pred_idxs = self.model.decoder(true_batches, word_reprs, cls_reprs)
                    loss, batch_stats = self.model.decoder.loss(true_batches, 
                                                                word_reprs, 
                                                                cls_reprs,
                                                                torch.tensor(deps_pred_idxs)
                                                                )
                    bptt = True # used for onmt

                step = self.optim.training_step
                if self.train_scorers != {} and step % self.train_eval_steps ==0 :
                    # Compute and save stats
                    computed_metrics = {}

                    # the goal here is not to translate but to parse dependences
                    # texts_ref = [" ".join(w) for w in [words for words in true_batches.words]]
                    dep_refs = true_batches.deprel_idxs


                    for metric in self.train_scorers:
                        logger.info("UPDATING TRAINING {}".format(metric))
                        self.train_scorers[metric]["value"] = self._training_eval_handler(
                            scorer=self.train_scorers[metric]["scorer"],
                            preds=deps_pred_idxs,
                            texts_ref=dep_refs,
                        )
                        logger.info(
                            "training {}: {}".format(
                            metric, self.train_scorers[metric]["value"]
                            )
                        )
                        computed_metrics[metric] = self.train_scorers[metric]["value"]
                        batch_stats.computed_metrics = computed_metrics

                    self.optim.backward(loss) 
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)     

            except Exception as exc:
                trace_content = traceback.format_exc()
                if "CUDA out of memory" in trace_content:
                    logger.info(
                        "Step %d, cuda OOM - batch removed",
                        self.optim.training_step,
                    )
                    torch.cuda.empty_cache()
                else:
                    traceback.print_exc()
                    raise exc

            # should test decoder.state but no attribute
            
            # in case of multi step gradient accumulation,
            # updtae only after accum batches
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(self.n_gpu)
                )
            self.optim.step()                 

    def _start_report_manager(self, start_time=None):
        """Simple function to start report manager (if any)"""

        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """Simple function to report training stats (if report_manager is set)
        see ``onmt.utils.ReportManagerBase.report_training`` for doc"""

        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None
                if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                report_stats,
                multigpu=self.n_gpu > 1,
            )

    def _report_step(self, learning_rate, step, valid_stats=None, train_stats=None):
        """Simple function to report stats (if report_manager is set)
        see ``onmt.utils.ReportManagerBase.report_step`` for doc"""

        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None
                if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                step,
                valid_stats=valid_stats,
                train_stats=train_stats,
            )
