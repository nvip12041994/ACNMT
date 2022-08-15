#!/bin/bash
# TEXT=./PhoMT.en-vi.bpe32k/
# fairseq-preprocess  --source-lang en --target-lang vi \
#                     --trainpref $TEXT/train.bpe --validpref $TEXT/valid.bpe --testpref $TEXT/test.bpe \
#                     --destdir data-bin/PhoMT_sentencepiece.tokenized.en-vi \
#                     --workers 32


TEXT=./PhoMT.vi-en.bpe32k/
fairseq-preprocess  --source-lang vi --target-lang en \
                    --trainpref $TEXT/train.bpe --validpref $TEXT/valid.bpe --testpref $TEXT/test.bpe \
                    --destdir data-bin/PhoMT_sentencepiece.tokenized.vi-en \
                    --workers 32