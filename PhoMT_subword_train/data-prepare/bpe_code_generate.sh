#!/bin/bash

#pip install https://github.com/rsennrich/subword-nmt/archive/master.zip
#pip show subword-nmt
script_link="./subword-nmt/subword_nmt/"
apply_bpe=${script_link}"/apply_bpe.py "
get_vocab=${script_link}"/get_vocab.py "
learn_bpe=${script_link}"/learn_bpe.py "

train_file="./clean_iwslt15/train"
test_file="./clean_iwslt15/test"
valid_file="./clean_iwslt15/valid"
num_operations=47000
codes_file="./clean_data_bpe/bpe_code"
tgt_lan=".vi"
src_lan=".en"


mkdir -p clean_data_bpe
output_test="./clean_data_bpe/test"
output_valid="./clean_data_bpe/valid"
output_train="./clean_data_bpe/train"
vocab_file="./clean_data_bpe/vocab_file"

echo "generate bpe code file"
cat ${train_file}${tgt_lan} ${train_file}${src_lan} |  python ${learn_bpe} -s ${num_operations} -o ${codes_file}

echo " get resulting vocabulary"
python ${apply_bpe} -c ${codes_file} < ${train_file}${tgt_lan} | python ${get_vocab} > ${vocab_file}${tgt_lan}
python ${apply_bpe} -c ${codes_file} < ${train_file}${src_lan} | python ${get_vocab} > ${vocab_file}${src_lan}

echo "re-apply byte pair encoding with vocabulary filter"
for i in {1..5}
do
    python ${apply_bpe} --dropout 0.1 -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 < ${train_file}${tgt_lan} > ${output_train}${tgt_lan}_${i}
    python ${apply_bpe} --dropout 0.1 -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 < ${train_file}${src_lan} > ${output_train}${src_lan}_${i}
done

echo "apply to test data"

python ${apply_bpe} -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 < ${test_file}${src_lan} > ${output_test}${src_lan}
python ${apply_bpe} -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 < ${test_file}${tgt_lan} > ${output_test}${tgt_lan}

echo "apply to valid data"
python ${apply_bpe} -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 < ${valid_file}${src_lan} > ${output_valid}${src_lan}
python ${apply_bpe} -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 < ${valid_file}${tgt_lan} > ${output_valid}${tgt_lan}
