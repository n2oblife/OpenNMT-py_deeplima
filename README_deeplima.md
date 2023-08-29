
# OpenNMT-py: Open-Source Neural Machine Translation adapted for deeplima use

For the deeplima adaptation using the trankit training method :

## Installation

Install the project from the repository
```bash
pip install -e .
pip install -r requirements.opt.txt
```

## Prepare dataset

You have to prepare the dataset for the onmt library to work, especially to use the validation pipeline. There is a bash file to launch the training. It needs a config file with a .yaml format and data with .conllu format. There ar

```bash
bash onmt_training.sh --train path/to/training/set.conllu --dev path/to/validation/set.conllu --fields head-deprel --config path/to/config/file.yaml  
```

This script will both prepare the dataset and launch the training according to the config file

### Training

To launch a training once the dataset is built :

```bash
onmt_train -config /path/to/your/config
```