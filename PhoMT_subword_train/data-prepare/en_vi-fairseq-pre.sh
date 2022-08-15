#!/bin/bash
TEXT=./en_vi_clean_data_bpe/
python ../ACNMT/fairseq_cli/preprocess.py  --source-lang en --target-lang vi \
                    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                    --destdir data-bin/iwslt15.tokenized.en-vi \
                    --bpe subword_nmt \
                    --workers 32
