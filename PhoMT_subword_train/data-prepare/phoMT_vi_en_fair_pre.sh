#!/bin/bash
TEXT=./PhoMT_bpe/
fairseq-preprocess  --source-lang vi --target-lang en \
                    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                    --destdir data-bin/PhoMT.tokenized.vi-en \
                    --bpe subword_nmt \
                    --workers 32
cp $TEXT/bpe_code data-bin/PhoMT.tokenized.vi-en/