#!/bin/bash

# genrate N hypotheses with the base MT model (fw score)
# TEST_SOURCE_FILE="PhoMT.vi-en.bpe32k/test.bpe.vi" # one sentence per line, converted to the sentencepiece used by the base MT model
# MT_DATA_PATH="data-bin/PhoMT_sentencepiece.tokenized.vi-en/"
# MT_MODEL="checkpoints/transformer_PhoMT_vi_en/checkpoint.best_loss_3.6560.pt"
# N=50

# cat ${TEST_SOURCE_FILE} | \
#     fairseq-interactive ${MT_DATA_PATH} \
#     --max-tokens 4000 --buffer-size 16 \
#     --num-workers 32 --path ${MT_MODEL} \
#     --beam $N --nbest $N \
#     --post-process sentencepiece | tee -a  test-hypo_vi_en.out


TEST_SOURCE_FILE="PhoMT.vi-en.bpe32k/test.bpe.vi" # one sentence per line, converted to the sentencepiece used by the base MT model
MT_DATA_PATH="data-bin/PhoMT_sentencepiece.tokenized.vi-en/"
N=5

# use nullglob in case there are no matching files
shopt -s nullglob

# create an array with all the filer/dir inside ~/myDir
arr=(./checkpoints/transformer_PhoMT_vi_en/*)

# iterate through array using a counter
for ((i=0; i<${#arr[@]}; i++)); do
    #do something to each element of array
    echo "Translate use ${arr[$i]}" 
    name=`echo "${arr[$i]}" | awk -F/ '{print $NF}' | sed -r 's/^checkpoint_//g' | sed -r 's/^checkpoint//g' | sed -r 's/^checkpoint.//g' | sed -r 's/\.pt//g'`
    cat ${TEST_SOURCE_FILE} | \
        fairseq-interactive ${MT_DATA_PATH} \
        --max-tokens 4000 --buffer-size 16 \
        --num-workers 32 --path ${arr[$i]} \
        --beam $N --nbest $N \
        --post-process sentencepiece  &> ./out/vi_en/${name}.out
done