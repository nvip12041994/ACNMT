#!/bin/bash

#pip install https://github.com/rsennrich/subword-nmt/archive/master.zip
#pip show subword-nmt
script_link="./subword-nmt/subword_nmt/"
apply_bpe=${script_link}"/apply_bpe.py "
get_vocab=${script_link}"/get_vocab.py "
learn_bpe=${script_link}"/learn_bpe.py "
learn_joint_bpe_and_vocab=${script_link}"/learn_joint_bpe_and_vocab.py"

train_file="./PhoMT/tokenization/train/train"
test_file="./PhoMT/tokenization/test/test"
valid_file="./PhoMT/tokenization/valid/valid"
num_operations=32000
codes_file="./PhoMT_bpe/bpe_code"
tgt_lan=".en"
src_lan=".vi"


mkdir -p PhoMT_bpe
output_test="./PhoMT_bpe/test"
output_valid="./PhoMT_bpe/valid"
output_train="./PhoMT_bpe/train"
vocab_file="./PhoMT_bpe/vocab_file"

echo "learn joint bpe and vocab."
echo "python ${learn_joint_bpe_and_vocab} --input ${train_file}${src_lan} ${train_file}${tgt_lan} -s ${num_operations} -o ${codes_file} --write-vocabulary ${vocab_file}${src_lan} ${vocab_file}${tgt_lan}"
python ${learn_joint_bpe_and_vocab} --input ${train_file}${src_lan} ${train_file}${tgt_lan} -s ${num_operations} -o ${codes_file} --write-vocabulary ${vocab_file}${src_lan} ${vocab_file}${tgt_lan}


echo "re-apply byte pair encoding with vocabulary filter"
echo "python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 -i ${train_file}${tgt_lan} -o ${output_train}${tgt_lan}"
python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 -i ${train_file}${tgt_lan} -o ${output_train}${tgt_lan}
echo "python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 -i ${train_file}${src_lan} -o ${output_train}${src_lan}"
python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 -i ${train_file}${src_lan} -o ${output_train}${src_lan}

echo "apply to test data"
echo "python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 -i ${test_file}${src_lan} -o ${output_test}${src_lan}"
python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 -i ${test_file}${src_lan} -o ${output_test}${src_lan}
echo "python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 -i ${test_file}${tgt_lan} -o ${output_test}${tgt_lan}"
python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 -i ${test_file}${tgt_lan} -o ${output_test}${tgt_lan}

echo "apply to valid data"
echo "python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 -i ${valid_file}${src_lan} -o ${output_valid}${src_lan}"
python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${src_lan} --vocabulary-threshold 50 -i ${valid_file}${src_lan} -o ${output_valid}${src_lan}
echo "python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 -i ${valid_file}${tgt_lan} -o ${output_valid}${tgt_lan}"
python ${apply_bpe} --glossaries "[\u4e00-\u9fff]+" -c ${codes_file} --vocabulary ${vocab_file}${tgt_lan} --vocabulary-threshold 50 -i ${valid_file}${tgt_lan} -o ${output_valid}${tgt_lan}