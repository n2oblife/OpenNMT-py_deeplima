import sys
import os
import pyconll
import logging
import ruamel.yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s')
warnings.simplefilter("ignore")

# keeps the comments of the yaml file
yaml = ruamel.yaml.YAML()

def is_multi_token(token, input_field)-> bool:
    """Checks if the token is a multi token one with empty fields to avoid counting it twice

    Args:
        token (_type_): _description_
        input_field (_type_): _description_

    Returns:
        bool: _description_
    """
    n = len(input_field)
    k = 0
    for field in input_field:
        if getattr(token, field) is None:
            k+=1
    if n==k:
        return True
    return False

def write_onmt(path:str, input_field:list):
    file = pyconll.load_from_file(path)
    
    src_txt_path = path[:-7] +'_src_onmt.txt'
    tgt_txt_path = path[:-7] +'_tgt_onmt.txt' 
    src_file = open(src_txt_path,'a')
    tgt_file = open(tgt_txt_path, 'a')

    line_tokens, line_field = '', '' 
    field_sep = '' if len(input_field)==1 else '-'

    for sentence in file:
        for token in sentence:
            if not is_multi_token(token, input_field):
                line_tokens += str(token.form) + ' '
                for field in input_field:
                    line_field += str(getattr(token, field))  + field_sep
                line_field = line_field[:-1] + ' '

        src_file.write(line_tokens + '\n')
        tgt_file.write(line_field + '\n')
        line_tokens, line_field = '', ''
    
    src_file.close()
    tgt_file.close()
    return src_txt_path, tgt_txt_path


def parse_input():
    """parse the command line for building
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--train', default='.', help='path of the conllu file for training')
    parser.add_argument('-d', '--dev', default='.', help='path of the conllu file for validation')
    parser.add_argument('-c', '--config', default='.', help="path to config file .yaml format used for the next part of the training")
    parser.add_argument('-f', '--fields', default='deprel', help="fields of the conllu file to be used for validation, can use multiple ones separated by '-' -> upos /xpos / feats / head / deprel / deps")
    parser.add_argument('-w', '--write', default='false', help="write in the config file if it is given -> true / false")
    parser.add_argument('-u', '--use', default='cli', help="choose to build the dataset according to the cli or the config file -> cli / config")
    args = vars(parser.parse_args())

    WRITE = ['true', 'false']
    if args['write'].lower() in WRITE:
        write_config = False if args['write'].lower() == 'false' else True
    else:
        raise ValueError(f"the argument to write in the config file must be {WRITE}. False by default")

    USE = ['cli', 'config']
    if args['use'].lower() in USE:
        use_cli = True if args['use'].lower() == 'cli' else False
    else:
        raise ValueError(f"the argument to use the cli or config file must be {USE}. cli by default")
    
    config_path = args['config']
    config_exist = os.path.isfile(config_path) and config_path.endswith('.yaml')
    need_config = write_config or (not use_cli)

    FIELDS = ['upos', 'xpos', 'feats', 'head', 'deprel', 'deps']
    input_field = args['fields'].split('-')
    for field in input_field:
        if field not in FIELDS:
            raise ValueError(f"the fields you entered are not part of the connlu format which contains {FIELDS}")


    if need_config and (not config_exist):
        raise ValueError(f"The config file {config_path} doesn't exist or is not a .yaml file but you need it")
    else:
        if use_cli:
            train_path = args['train']
            dev_path = args['dev']
            with open(config_path, 'r') as f:
                config_str = f.read()
                config_file = yaml.load(config_str)
        else:
            with open(config_path, 'r') as f:
                config_str = f.read()
                config_file = yaml.load(config_str)
                train_path = config_file["data"]["corpus_1"]["path_tgt"]
                dev_path = config_file["data"]["valid"]["path_tgt"]
    
    return train_path, dev_path, input_field, write_config, config_file, config_path


if __name__=='__main__':
    # build dataset for onmt (v1 - v2 with spaces between tokens)
    train_path, dev_path, input_field, write_config, config_file, config_path = parse_input()
    for_train_vocab_src, for_train_vocab_tgt = '',''
    for_dev_vocab_src, for_dev_vocab_tgt = '',''

    if train_path!='.':
        if os.path.isfile(train_path) and train_path.endswith('.conllu') and train_path!='.':
            logging.info("BUILDING TRAIN DATASET")
            write_onmt(train_path, input_field)
        else:
            raise FileExistsError(f"The file {train_path} doesn't exist or is not a .conllu file")
    
    if dev_path!='.':
        if os.path.isfile(dev_path) and dev_path.endswith('.conllu'):
            logging.info("BUILDING VALIDATION DATASET")
            write_onmt(dev_path, input_field)
        else:
            raise FileExistsError(f"The file {dev_path} doesn't exist or is not a .conllu file")
    
    if (train_path=='.') and (dev_path=='.') and (config_path=='.') :
        logging.warning("NOTHING HAS BEEN DONE")
        sys.exit(1)
    else:
        if write_config:
            if train_path!='.':
                config_file["data"]["corpus_1"]["path_src"] = train_path[:-7] +'_src_onmt.txt'
                config_file["data"]["corpus_1"]["path_tgt"] = train_path
                config_file["src_vocab"] = train_path[:-4] +'.vocab.txt'
            if dev_path!='.':
                config_file["data"]["valid"]["path_src"] = dev_path[:-7] +'_src_onmt.txt'
                config_file["data"]["valid"]["path_tgt"] = dev_path
                config_file["tgt_vocab"] = dev_path[:-4] +'.vocab.txt'
            with open(config_path, 'w') as f:
                logging.info("CONFIG UPDATED")
                yaml.dump(config_file, f)
                f.close()
        logging.info("DATASETS BUILT")
        sys.exit(0) 
