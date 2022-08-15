#!/bin/bash
TEXT=./vi_en_clean_data_bpe/
python ../ACNMT/fairseq_cli/preprocess.py  --source-lang vi --target-lang en \
                    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                    --destdir data-bin/iwslt15.tokenized.vi-en \
                    --bpe subword_nmt \
                    --workers 32
