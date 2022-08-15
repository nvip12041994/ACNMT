import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='vi')

def read_file(filename,detoken):
    output = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            line = detoken.detokenize(line)
            output.append(line)
    return output

def eval_metric(metric, hypos, ref):
    if metric == "bleu":
        score = sacrebleu.corpus_bleu(hypos, [ref])#.score
    else:
        score = sacrebleu.corpus_ter(hypos, [ref])#.score

    return score

# ref = read_file("./bleu_evaluate/test_PhoMT_vi_en/target.en",md)
# hyp = read_file("./bleu_evaluate/test_PhoMT_vi_en/hypo.en",md)

ref = read_file("./bleu_evaluate/test_PhoMT_en_vi/target.vi",md)
hyp = read_file("./bleu_evaluate/test_PhoMT_en_vi/hypo.vi",md)

# ref = read_file("./bleu_evaluate/test_iwslt15_vi_en/target.en",md)
# hyp = read_file("./bleu_evaluate/test_iwslt15_vi_en/hypo.en",md)

# ref = read_file("./bleu_evaluate/test_iwslt15_en_vi/target.vi",md)
# hyp = read_file("./bleu_evaluate/test_iwslt15_en_vi/hypo.vi",md)

bleu_score = eval_metric("bleu",hyp,ref)
print("Bleu score: {}".format(bleu_score))

ter_score = eval_metric("ter",hyp,ref)
print("TER score: {}".format(ter_score))

#perl mosesdecoder/scripts/generic/multi-bleu-detok.perl bleu_evaluate/test_iwslt15_vi_en/hypo.en < bleu_evaluate/test_iwslt15_vi_en/target.en