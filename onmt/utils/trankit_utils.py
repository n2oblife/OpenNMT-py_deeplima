import torch
from onmt.utils.misc import use_gpu
from copy import deepcopy
from trankit.utils.tbinfo import *
from trankit.utils.tokenizer_utils import *
from collections import defaultdict
from trankit.iterators.tagger_iterators import TaggerDatasetLive, DataLoader
from trankit.iterators.tokenizer_iterators import TokenizeDatasetLive
from trankit.adapter_transformers import XLMRobertaTokenizer
import onmt

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
                      'task',
                      'treebank'
                      ]

class TConfig():
    """Training Config adapted from trankit with elements needed for building model"""
    def __init__(self, opt) -> None:
            self.category =  'customized'
            self.gpu =   use_gpu(opt)
            self.save_dir =  opt.save_data
            self._save_dir = opt.save_data
            self.train_conllu_fpath =  opt.data['corpus_1']['path_tgt']
            self.dev_conllu_fpath =  opt.data['valid']['path_tgt']
            self.adapter_,name = opt.task + opt.treebank
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
    _config = TConfig(t_opt)
    # _config = state_dic['config']
    word_splitter = XLMRobertaTokenizer.from_pretrained(t_opt.embedding_name)
    text_gen = (row for row in open(opt.src, "r"))
    sents = []

    dict_output = posdep_doc(in_doc=opt.src, opt=opt, _config=_config)
    write_output(opt, dict_output)
    # dict_output = {TEXT: text, SENTENCES: sents, SCORES : score, LANG: t_opt.treebank}

def file_loader(text_path):
    for line in open(text_path, "r"):
        yield line

def write_output(opt, dict_output, file): 
    with open(opt.output, "w") as f:
        file.write(dict_output)

def posdep_doc(in_doc, opt, _config):  # assuming input is a document 
    dposdep_doc = deepcopy(in_doc)
    # load outputs of tokenizer
    config = _config
    wordpiece_splitter = XLMRobertaTokenizer.from_pretrained(config.embedding_name)
    test_set = TaggerDatasetLive(
        tokenized_doc=dposdep_doc,
        wordpiece_splitter=wordpiece_splitter,
        config=config
    )
    test_set.numberize()

    # load weights of tagger into the combined model
    # self._load_adapter_weights(model_name='tagger')

    # make predictions
    eval_batch_size = tbname2tagbatchsize.get(_config.treebank_name, config.batch_size)
    if config.embedding_name == 'xlm-roberta-large':
        eval_batch_size = int(eval_batch_size / 3)

    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    _, model, _ = load_test_model(opt)

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

        f = open(opt.src, 'w')
        for bid in range(batch_size):
            sentid = batch.sent_index[bid]
            for i in range(batch.word_num[bid]):
                wordid = batch.word_ids[bid][i]

                # upos
                pred_upos_id = predicted_upos[bid][i]
                upos_name = config.itos[config.active_lang][UPOS][pred_upos_id]
                test_set.conllu_doc[sentid][wordid][UPOS] = upos_name
                # xpos
                pred_xpos_id = predicted_xpos[bid][i]
                xpos_name = config.itos[config.active_lang][XPOS][pred_xpos_id]
                test_set.conllu_doc[sentid][wordid][XPOS] = xpos_name
                # feats
                pred_feats_id = predicted_feats[bid][i]
                feats_name = config.itos[config.active_lang][FEATS][pred_feats_id]
                test_set.conllu_doc[sentid][wordid][FEATS] = feats_name

                # head
                test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                # deprel
                test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]

                #scores
                test_set.conllu_doc[sentid][wordid][SCORES] = scores[bid][i]
        f.write(test_set.conllu_doc)
    f.close()
    tagged_doc = get_output_doc(dposdep_doc, test_set.conllu_doc)
    torch.cuda.empty_cache()
    return tagged_doc
