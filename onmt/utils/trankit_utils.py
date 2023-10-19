import torch
from onmt.utils.misc import use_gpu
from copy import deepcopy
from trankit.utils.tbinfo import *
from trankit.utils.tokenizer_utils import *
from collections import defaultdict
from trankit.iterators.tagger_iterators import TaggerDatasetLive, DataLoader, Instance
from trankit.iterators.tokenizer_iterators import TokenizeDatasetLive
from trankit.adapter_transformers import XLMRobertaTokenizer
import onmt
from trankit import Pipeline
from trankit.models.classifiers import TokenizerClassifier
from collections import namedtuple

FIELD_NUM = 10

ID = 'id'
TEXT = 'text'
LEMMA = 'lemma'
UPOS = 'upos'
XPOS = 'xpos'
FEATS = 'feats'
HEAD = 'head'
DEPREL = 'deprel'
DEPS = 'deps'
MISC = 'misc'
SSPAN = 'span'
DSPAN = 'dspan'
EXPANDED = 'expanded'
SENTENCES = 'sentences'
TOKENS = 'tokens'
NER = 'ner'
LANG = 'lang'
SCORES = 'score'

FIELD_TO_IDX = {ID: 0, TEXT: 1, LEMMA: 2, UPOS: 3, XPOS: 4, FEATS: 5, HEAD: 6, DEPREL: 7, DEPS: 8, MISC: 9, SCORES: 10}

ALREADY_CONFIGURED_TRAIN = ['save_data', 
                      'save_data', 
                      'data',
                      'gpu_ranks',
                      ]

# for sents
instance_fields_tok = [
    'paragraph_index',
    'wordpieces', 'wordpiece_labels', 'wordpiece_ends',
    'piece_idxs', 'attention_masks', 'token_type_idxs',
    'wordpiece_num'
]

Instance_tok = namedtuple('Instance_tok', field_names=instance_fields_tok)

class TConfig():
    """Training Config adapted from trankit with elements needed for building model"""
    def __init__(self, opt) -> None:
            self.category =  'customized'
            self.gpu =   use_gpu(opt)
            self.save_dir =  opt.save_data
            self._save_dir = opt.save_data # considered cache
            self.train_conllu_fpath =  opt.data['corpus_1']['path_tgt']
            self.dev_conllu_fpath =  opt.data['valid']['path_tgt']
            self.adapter_name = opt.task + opt.treebank_name
            self.load_opt(opt)
    
    def load_opt(self, opt)-> None:
        for key, value in opt.__dict__.items():
            if not (key in ALREADY_CONFIGURED_TRAIN):
                setattr(self, key, value)
    
    def get_default_tconfig(self):
        return {'category': self.category, # pipeline category
                'gpu' : self.gpu,
                'save_dir': self.save_dir, # directory for saving trained model
                'train_conllu_fpath': self.train_conllu_fpath, # annotations file in CONLLU format  for training
                'dev_conllu_fpath': self.dev_conllu_fpath, # annotations file in CONLLU format for development
                }
    
    def build_dict_from_attr(self):
        dico = self.get_default_tconfig()
        for key, value in self.__dict__.items():
            dico[key] = value
        return dico

    def get_config(self):
        return self.build_dict_from_attr()

class Config():
    """Test Config adapted from trankit with elements needed for building model"""
    def __init__(self, opt) -> None:
            self.load_opt(opt)
    
    def load_opt(self, opt)-> None:
        for key, value in opt.__dict__.items():
            setattr(self, key, value)
    
    def get_default_tconfig(self):
        return {'category': self.category, # pipeline category
                'gpu' : self.gpu,
                'save_dir': self.save_dir, # directory for saving trained model
                'train_conllu_fpath': self.train_conllu_fpath, # annotations file in CONLLU format  for training
                'dev_conllu_fpath': self.dev_conllu_fpath, # annotations file in CONLLU format for development
                }
    
    def build_dict_from_attr(self):
        dico = self.get_default_tconfig()
        for key, value in self.__dict__.items():
            dico[key] = value
        return dico

    def get_config(self):
        return self.build_dict_from_attr()

# Definition of inferences functions
  
def infer_trankit_model(opt, state_dic, t_opt): 
    if t_opt.task=='posdep':
        posdep(opt, state_dic, t_opt)
    else:
        # TODO
        raise NotImplementedError

def posdep(opt, state_dic, t_opt): 
    # _config = TConfig(t_opt)
    _config = state_dic['trankit_config']
    word_splitter = XLMRobertaTokenizer.from_pretrained(t_opt.embedding_name)
    # text_gen = (row for row in open(opt.src, "r"))
    dict_output = posdep_doc(in_doc=opt.src, opt=opt, _config=_config)
    write_output(opt, dict_output)
    # dict_output = {TEXT: text, SENTENCES: sents, SCORES : score, LANG: t_opt.treebank_name}

def write_output(opt, dict_output): 
    with open(opt.output, "w") as f:
        json.dump(dict_output, f)

def load_model(opt):
    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    _, model, _ = load_test_model(opt) # = vocabs, model, model_opt 
    return model

def download_vocab(config) :
    """_summary_

    Args:
        config (_type_): _description_
    """
    link = "https://github.com/n2oblife/OpenNMT-py_deeplima/tree/for_deeplima/bin/download_trankit_vocab.sh"
    os.system(f"wget -q {link}")
    lang = treebank2lang[config.treebank_name]
    os.system(f"bash download_trankit_vocab.sh -l {lang} -d {config._save_dir}/")
    os.system("rm download_trankit_vocab.sh")

def adapt_config(config):
    """Adapt the config for inference"""
    # TODO save the vocab in .json format, check result with cache for training
    # save_vocab_json(config) 
    # config.treebank_name = lang2treebank[config.treebank_name]
    config.training = False
    config._ud_eval = False
    return config

def numberize_tok(test_set, wordpiece_splitter):  # wordpiece tokenizer
    data = []
    for inst in test_set.data:
        wordpieces = inst['wordpieces']
        wordpiece_labels = inst['wordpiece_labels']
        wordpiece_ends = inst['wordpiece_ends']
        paragraph_index = inst['paragraph_index']
        # Pad word pieces with special tokens
        piece_idxs = wordpiece_splitter.encode(
            wordpieces,
            add_special_tokens=True,
            max_length=test_set.max_input_length,
            truncation=True
        )
        assert len(piece_idxs) <= test_set.max_input_length

        pad_num = test_set.max_input_length - len(piece_idxs)
        attn_masks = [1] * len(piece_idxs) + [0] * pad_num
        piece_idxs = piece_idxs + [0] * pad_num

        # token type idxs
        token_type_idxs = [-100 if piece_id >= len(wordpieces) else wordpiece_labels[piece_id] for piece_id in
                            range(len(piece_idxs) - 2)]

        instance = Instance_tok(
            paragraph_index=paragraph_index,
            wordpieces=wordpieces,
            wordpiece_labels=wordpiece_labels,
            wordpiece_ends=wordpiece_ends,
            piece_idxs=piece_idxs,
            attention_masks=attn_masks,
            token_type_idxs=token_type_idxs,
            wordpiece_num=len(wordpieces)
        )
        data.append(instance)
    test_set.data = data

def tokenize_doc(in_doc, model, opt, _config):  # assuming input is a document
    eval_batch_size = tbname2tokbatchsize.get(lang2treebank[_config.lang], _config.batch_size)
    if _config.embedding_name == 'xlm-roberta-large':
        eval_batch_size = int(eval_batch_size / 2)
    # load input text
    config = _config
    raw_text = open(in_doc, "r").read() # could be optimized as a generator
    # TODO download the lang.tokenizer.mdl file from LINK in download_vocabtrankit.sh 
    test_set = TokenizeDatasetLive(config=config, raw_text=raw_text, 
                                   max_input_length=tbname2max_input_length.get(lang2treebank[config.lang], 400))
    numberize_tok(test_set, config.wordpiece_splitter)

    # # load weights of tokenizer into the combined model
    # self._load_adapter_weights(model_name='tokenizer')

    # make predictions
    wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
    tokenizer_classifier = TokenizerClassifier(_config, lang2treebank[_config.lang])
    for batch in DataLoader(test_set, batch_size=eval_batch_size,
                            shuffle=False, collate_fn=test_set.collate_fn):
        wordpiece_reprs = model.encoder.get_tokenizer_inputs(batch)
        predictions = tokenizer_classifier.predict(batch, wordpiece_reprs)
        wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
        wp_pred_labels = wp_pred_labels.data.cpu().numpy().tolist()

        for i in range(len(wp_pred_labels)):
            wordpiece_pred_labels.append(wp_pred_labels[i][: len(wp_ends[i])])

        wordpiece_ends.extend(wp_ends)
        paragraph_indexes.extend(para_ids)
    # mapping
    para_id_to_wp_pred_labels = defaultdict(list)

    for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                            paragraph_indexes):
        para_id_to_wp_pred_labels[p_index].extend([(pred, char_position) for pred, char_position in
                                                    zip(wp_pred_ls, wp_es)])
    # get predictions
    corpus_text = raw_text

    paragraphs = [pt.rstrip() for pt in
                    NEWLINE_WHITESPACE_RE.split(corpus_text) if
                    len(pt.rstrip()) > 0]
    all_wp_preds = []
    all_para_texts = []
    all_para_starts = []
    ##############
    cloned_raw_text = deepcopy(raw_text)
    global_offset = 0
    for para_index, para_text in enumerate(paragraphs):
        cloned_raw_text, start_char_idx = get_start_char_idx(para_text, cloned_raw_text)
        start_char_idx += global_offset
        global_offset = start_char_idx + len(para_text)
        all_para_starts.append(start_char_idx)

        para_wp_preds = [0 for _ in para_text]
        for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
            # TODO find why wp_l is always 0
            # para_wp_preds[end_position] = wp_l
            para_wp_preds[end_position] = 1

        all_wp_preds.append(para_wp_preds)
        all_para_texts.append(para_text)
    ###########################
    doc = []
    for j in range(len(paragraphs)):
        para_text = all_para_texts[j]
        wp_pred = all_wp_preds[j]
        para_start = all_para_starts[j]

        current_tok = ''
        current_sent = []
        local_position = 0
        for t, wp_p in zip(para_text, wp_pred):
            local_position += 1
            current_tok += t
            if wp_p >= 1:
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=_config._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) <= 0:
                    current_tok = ''
                    continue
                additional_info = {DSPAN: (para_start + local_position - len(tok),
                                            para_start + local_position)}
                current_sent += [(tok, wp_p, additional_info)]
                current_tok = ''
                if (wp_p == 2 or wp_p == 4):
                    processed_sent = get_output_sentence(current_sent)
                    doc.append({
                        ID: len(doc) + 1,
                        TEXT: raw_text[processed_sent[0][DSPAN][0]: processed_sent[-1][DSPAN][
                            1]],
                        TOKENS: processed_sent,
                        DSPAN: (processed_sent[0][DSPAN][0], processed_sent[-1][DSPAN][1])
                    })
                    current_sent = []

        if len(current_tok):
            tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=_config._ud_eval)
            assert '\t' not in tok, tok
            if len(tok) > 0:
                additional_info = {DSPAN: (para_start + local_position - len(tok),
                                            para_start + local_position)}
                current_sent += [(tok, 2, additional_info)]

        if len(current_sent):
            processed_sent = get_output_sentence(current_sent)
            doc.append({
                ID: len(doc) + 1,
                TEXT: raw_text[
                        processed_sent[0][DSPAN][0]: processed_sent[-1][DSPAN][1]],
                TOKENS: processed_sent,
                DSPAN: (processed_sent[0][DSPAN][0], processed_sent[-1][DSPAN][1])
            })

    # multi-word expansion if required
    # if tbname2training_id[_config.treebank_name] % 2 == 1:
    #     doc = self._mwt_expand(doc)
    torch.cuda.empty_cache()
    return doc

def posdep_doc(in_doc, opt, _config):  # assuming input is a document    
    model = load_model(opt)

    # load wordpiece_splitter because not trained
    wordpiece_splitter = XLMRobertaTokenizer.from_pretrained(_config.embedding_name)
    config = adapt_config(_config)

    tokenized_doc = tokenize_doc(in_doc, model, opt, config)
    dposdep_doc = deepcopy(tokenized_doc)
    # tokenized_doc = tokenize_doc_cheat(in_doc, model, opt, _config)
    
    data_config = deepcopy(config)
    data_config._cache_dir = data_config._cache_dir+'/../../' 
    download_vocab(config)

    test_set = TaggerDatasetLive(
        tokenized_doc=dposdep_doc,
        wordpiece_splitter=wordpiece_splitter,
        config=data_config
    )

    numberize_(test_set, wordpiece_splitter)

    # make predictions
    tagged_doc = posdep_prediction(model,opt, config, test_set, dposdep_doc)
    torch.cuda.empty_cache()
    return tagged_doc

def posdep_prediction(model, opt, config, test_set, dposdep_doc):
    # set batch size
    eval_batch_size = tbname2tagbatchsize.get(config.treebank_name, config.batch_size)
    if config.embedding_name == 'xlm-roberta-large':
        eval_batch_size = int(eval_batch_size / 3)
    
    # make prediction
    for batch in DataLoader(test_set,
                            batch_size=eval_batch_size,
                            shuffle=False, collate_fn=test_set.collate_fn):
        batch_size = len(batch.word_num)

        word_reprs, cls_reprs = model.encoder(batch)
        predictions, _, scores = model.decoder(batch, word_reprs, cls_reprs)
        predicted_upos = predictions[0]
        predicted_xpos = predictions[1]
        predicted_feats = predictions[2]
        pred_tokens = predictions[3]

        f = open(opt.src, 'r')
        for bid in range(batch_size):
            sentid = batch.sent_index[bid]
            for i in range(batch.word_num[bid]):
                wordid = batch.word_ids[bid][i]

                # upos
                pred_upos_id = predicted_upos[bid][i]
                upos_name = config.itos[UPOS][pred_upos_id]
                test_set.conllu_doc[sentid][wordid][UPOS] = upos_name
                # xpos
                pred_xpos_id = predicted_xpos[bid][i]
                xpos_name = config.itos[XPOS][pred_xpos_id]
                test_set.conllu_doc[sentid][wordid][XPOS] = xpos_name
                # feats
                pred_feats_id = predicted_feats[bid][i]
                feats_name = config.itos[FEATS][pred_feats_id]
                test_set.conllu_doc[sentid][wordid][FEATS] = feats_name

                # head
                test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                # deprel
                test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]

                #scores
                # test_set.conllu_doc[sentid][wordid][SCORES] = scores[-1][i]
        f.close()
        # TODO check if need to be written
        # temp = open(opt.output+'temp')
    return get_output_doc(dposdep_doc, test_set.conllu_doc)

def numberize_(test_set, wordpiece_splitter):
    data = []
    for inst in test_set.data:
        words = inst['words']
        pieces = [[p for p in wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
        for ps in pieces:
            if len(ps) == 0:
                ps += ['-']
        word_lens = [len(x) for x in pieces]
        assert 0 not in word_lens
        flat_pieces = [p for ps in pieces for p in ps]
        assert len(flat_pieces) > 0

        word_span_idxs = []
        start = 1
        for l in word_lens:
            word_span_idxs.append([start, start + l])
            start += l

        # Pad word pieces with special tokens
        piece_idxs = wordpiece_splitter.encode(
            flat_pieces,
            add_special_tokens=True,
            max_length=test_set.max_input_length,
            truncation=True
        )
        assert len(piece_idxs) <= test_set.max_input_length

        attn_masks = [1] * len(piece_idxs)
        piece_idxs = piece_idxs
        assert len(piece_idxs) > 0

        edit_type_idxs = [test_set.vocabs[LEMMA][edit] for edit in inst[LEMMA]]
        upos_type_idxs = [test_set.vocabs[UPOS][upos] for upos in inst[UPOS]]
        xpos_type_idxs = [test_set.vocabs[XPOS][xpos] for xpos in inst[XPOS]]
        feats_type_idxs = [test_set.vocabs[FEATS][feats] for feats in inst[FEATS]]

        assert len(edit_type_idxs) == len(inst['words'])

        # head, deprel, word_mask
        head_idxs = [head for head in inst[HEAD]]
        deprel_idxs = [test_set.vocabs[DEPREL][deprel] for deprel in inst[DEPREL]]
        word_mask = [0] * (len(inst['words']) + 1)

        instance = Instance(
            sent_index=inst['sent_index'],
            word_ids=inst['word_ids'],
            words=inst['words'],
            word_num=len(inst['words']),
            piece_idxs=piece_idxs,
            attention_masks=attn_masks,
            word_span_idxs=word_span_idxs,
            word_lens=word_lens,
            edit_type_idxs=edit_type_idxs,
            upos_type_idxs=upos_type_idxs,
            xpos_type_idxs=xpos_type_idxs,
            feats_type_idxs=feats_type_idxs,
            head_idxs=head_idxs,
            deprel_idxs=deprel_idxs,
            word_mask=word_mask
        )
        data.append(instance)
    test_set.data = data