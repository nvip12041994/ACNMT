#!/bin/bash

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py
DATA_OUT=$ROOT/discriminator_train_data/iwslt15.en-vi.bpe32k/

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens
ORIG=$ROOT/clean_iwslt15/
# encode train/valid
echo "encoding train with learned BPE..."
python "$SPM_ENCODE" \
    --model "./PhoMT.en-vi.bpe32k/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $ORIG/train.en \
    --outputs $DATA_OUT/train.bpe.en \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN


# genrate N hypotheses with the base MT model (fw score)
TEST_SOURCE_FILE="discriminator_train_data/iwslt15.en-vi.bpe32k/train.bpe.en" # one sentence per line, converted to the sentencepiece used by the base MT model
MT_DATA_PATH="data-bin/PhoMT_sentencepiece.tokenized.en-vi/"
MT_MODEL="checkpoints/transformer_PhoMT_en_vi/checkpoint30.pt"
N=5

cat ${TEST_SOURCE_FILE} | \
    fairseq-interactive ${MT_DATA_PATH} \
    --max-tokens 4000 --buffer-size 16 \
    --num-workers 32 --path ${MT_MODEL} \
    --beam $N --nbest $N \
    --post-process sentencepiece | tee -a  discriminator_train_data/train.vi 