# OpenNMT-py_deeplima: Open-Source Neural Machine Translation adapted for deeplima use

This repository is an adaptatin from the official [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) library. It aims to train models according to the OpenNMT processes in order to convert and use them in C++ instead of python.

To reach the state of the art, the [Trankit](https://github.com/nlp-uoregon/trankit)'s models have been implemented. This models are close to the state of the art and use adapters. The adapters boil down to a lightweight way of finnetunning a transformer based model.

The OpenNMT library can convert the models trained in order to be used by [Ctranslate2](https://github.com/OpenNMT/CTranslate2) in C++, which is an inference engine for transformer based models.

So far, the Piece of Sentences and the Dependency parsing have been added to the OpenNMT-py_deeplima library from training to inference in Python. The conversion is available but the inference in C++ is still in developpement.

This project aims to train models in order to push them into Deeplima, the Deep Learning adaptation of [Lima](https://github.com/aymara/lima), a multilingual linguistic analyzer developed by the [CEA LIST](http://www-list.cea.fr/en), [LASTI laboratory](http://www.kalisteo.fr/en/index.htm).

## Installation

Clone the project and install the project from the sources :

```bash
git clone https://github.com/n2oblife/OpenNMT-py_deeplima.git
pip install -e OpenNMT-py_deeplima/.
pip install -r OpenNMT-py_deeplima/requirements.opt.txt
```

## Prepare dataset

You have to prepare the dataset for the onmt library to work, especially to use the validation pipeline. There is a bash file to launch the training. It needs a config file with a .yaml format and data with .conllu format :

```bash
bash onmt_training.sh --train <path/to/training/set.conllu> --dev <path/to/validation/set.conllu> --fields <task> --config <path/to/config/file.yaml>  
```

To disable the training in this script :

```bash
bash onmt_training.sh --train <path/to/training/set.conllu> --dev <path/to/validation/set.conllu> --fields <task> --config <path/to/config/file.yaml> --build <bool>
```

This script will both prepare the dataset and launch the training according to the config file. It will also enable to launch the script to download the vocab for trankit. There is an example of a config file with comments to help build it from scratch.

## Training

Models can be trained with different config files. To launch a training without building the datasets again :

```bash
onmt_train -config </path/to/your/config>
```

## Inference

Once the model has been trained, it is saved on the folder you mentioned in the config file. You can than use the inference to predict the task you trained :

```bash
onmt_translate -model <path/to/your/model.pt> -src <text/to/predict.txt> -output <path/to/result.txt>
```

##  C++ conversion

### Install CTranslate2

You have to install Ctranslate2 to enable conversion.
Clone the [adaptation of CTranslate2](https://github.com/n2oblife/CTranslate2_deeplima.git) for this project. Then build the project as followed. First compile the C++ library, requires a compiler supporting C++17 and CMake 3.15 or greater : 

```bash
git clone --recursive https://github.com/n2oblife/CTranslate2_deeplima.git
mkdir build && cd build
cmake ..
make -j4
make install
```

Then compile the python wrapper :

```bash
cd ../python
pip install -r install_requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl
```

### Set the env variables

If you installed the C++ library in a custom directory, you should configure additional environment variables:

```bash
export CTRANSLATE2_ROOT=<path/to/Ctranslate2/installation/directory>
export LD_LIBRARY_PATH=$CTRANSLATE2_ROOT/lib:$LD_LIBRARY_PATH
```

### Convert

To convert the model into a binary readalbe by CTranslate2, you enter the next command :

```bash
onmt_release_model --model <path/to/model.pt> --output <path/to/folder> --format  ctranslate2
```
