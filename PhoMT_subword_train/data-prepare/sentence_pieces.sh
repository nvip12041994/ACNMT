# SRCS=(
#     "en"
# )
# TGT=vi

SRCS=(
    "vi"
)
TGT=en

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts_python
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=32000
ORIG=$ROOT/PhoMT_sentencepiece
#DATA_OUT=$ROOT/PhoMT.en-vi.bpe32k
DATA_OUT=$ROOT/PhoMT.vi-en.bpe32k

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $ORIG/train.${SRC}; echo $ORIG/train.${TGT}; done | tr "\n" ",")

echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA_OUT/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train/valid
echo "encoding train with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATA_OUT/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $ORIG/train.${SRCS[0]} $ORIG/train.${TGT} \
    --outputs $DATA_OUT/train.bpe.${SRCS[0]} $DATA_OUT/train.bpe.${TGT} \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN

echo "encoding valid with learned BPE..."

python "$SPM_ENCODE" \
    --model "$DATA_OUT/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $ORIG/valid.${SRCS[0]} $ORIG/valid.${TGT} \
    --outputs $DATA_OUT/valid.bpe.${SRCS[0]} $DATA_OUT/valid.bpe.${TGT} \

echo "encoding test with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATA_OUT/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $ORIG/test.${SRCS[0]} $ORIG/test.${TGT} \
    --outputs $DATA_OUT/test.bpe.${SRCS[0]} $DATA_OUT/test.bpe.${TGT} \