#!/usr/bin/env bash

# Default values
LINK=http://nlp.uoregon.edu/download/trankit/ 

PATH_TO_DIR="./data/trankit_voc/"
LANG="english"

# handling of the inputs for the bash file
help(){
    echo "Usage onmt_training - it downloads the voc of languages from trankit to do the inference. 
    Check the languages availabe at : $LINK
        [-l | --langue] : language to download, by default=$LANG -> all
        [-d | --directory] : change the default path where to download the vocab, by default=$PATH_TO_DIR
        "
    exit 2
}

handle_error(){
    if [ $? -ne 0 ]; then
        echo $1 # prompt the error message
        exit 1
    fi
}

# Options
SHORT=l:,h
LONG=langue:,help
OPTS=$(getopt -a -n onmt_training --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

eval set -- "$OPTS"

while :
do
  case "$1" in
    -l | --langue )
      LANG="$2"
      shift 2
      ;;
    -d | --directory )
      PATH_TO_DIR="$2"
      shift 2
      ;;
    -h | --help)
      help
      exit 2
      ;;
    --)
      shift;
      break 
      ;;
    *)
      echo "INFO - Unexpected option : $1"
      help
      exit 2
      ;;
  esac
done

# define functions

get_vocab() {
    LG=$1
    DIR=$2
    if [ -f "$DIR$LG.vocabs.json" ]; then
        echo "INFO - $DIR$LG.vocabs.json already exists"
    else 
        echo "INFO - $DIR$LG doesn't exists, let's download it"
        wget -q "$LINK/$LG.zip" -O temp.zip
        handle_error "INFO - Language not supported, check the languages availabe at : $LINK"
        unzip -q temp.zip
        rm temp.zip
        mv "$LG.vocabs.json" $DIR
        rm $LG*
    fi
}

# --main--

# check directory
if [ -d "$PATH_TO_DIR" ]; then
    echo "INFO - $PATH_TO_DIR exists"
else 
    echo "INFO - $PATH_TO_DIR doesn't exists, let's make one"
    mkdir $PATH_TO_DIR
fi

# check langu   ge to download
if [ "$LANG" = "all" ]; then
    echo "INFO - can't download all vocabs now"
else
    if [ -f "$PATH_TO_DIR$LG.vocabs.json" ]; then
        echo "INFO - $PATH_TO_DIR/$LG.vocabs.json exists"
    else
        echo "INFO - downloading vocab"
        get_vocab $LANG $PATH_TO_DIR
    fi
fi

exit 0