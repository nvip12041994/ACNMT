#!/usr/bin/env python3 -u
"""
Score raw text with a trained model.
"""

import sacrebleu
from sacremoses import MosesDetokenizer
import numpy as np
import json
from os import walk
import tqdm

import fairseq

def parse_fairseq_gen(filename):
    source = {}
    hypos = {}
    scores = {}
    #num_lines = sum(1 for line in open(filename,'r'))
    with open(filename, "r", encoding="utf-8") as f:
        #for line in tqdm.tqdm(f, total=num_lines):
        for line in f:
            line = line.strip()
            if line.startswith("S-"):  # source
                uid, text = line.split("\t", 1)
                uid = int(uid[2:])
                source[uid] = text
            elif line.startswith("D-"):  # hypo
                uid, score, text = line.split("\t", 2)
                uid = int(uid[2:])
                if uid not in hypos:
                    hypos[uid] = []
                    scores[uid] = []
                hypos[uid].append(text)
                scores[uid].append(float(score))
            else:
                continue

    source_out = [source[i] for i in range(len(hypos))]
    hypos_out = [h for i in range(len(hypos)) for h in hypos[i]]
    scores_out = [s for i in range(len(scores)) for s in scores[i]]

    return source_out, hypos_out, scores_out

def read_target(filename):
    with open(filename, "r", encoding="utf-8") as f:
        output = [line.strip() for line in f]
    return output

def read_target_detoken(filename,detoken):
    output = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            line = detoken.detokenize(line)
            output.append(line)
    return output

def get_score(mt_s, w1, lp, tgt_len):
    return mt_s / (tgt_len ** lp) * w1

def get_best_hyps(mt_scores, hypos, fw_weight, lenpen, beam):
    assert len(mt_scores) == len(hypos)
    hypo_scores = []
    best_hypos = []
    best_scores = []
    offset = 0
    #for i in tqdm.tqdm(range(len(hypos))):
    for i in range(len(hypos)):
        tgt_len = len(hypos[i].split())
        hypo_scores.append(
            get_score(mt_scores[i], fw_weight, lenpen, tgt_len)
        )

        if (i + 1) % beam == 0:
            max_i = np.argmax(hypo_scores)
            #best_hypos.append(hypos[offset + max_i] + ".")
            best_hypos.append(hypos[offset + max_i])
            best_scores.append(hypo_scores[max_i])
            hypo_scores = []
            offset += beam
    return best_hypos, best_scores


def eval_metric(metric, hypos, ref):
    if metric == "bleu":
        score = sacrebleu.corpus_bleu(hypos, [ref], force=True).score
    else:
        score = sacrebleu.corpus_ter(hypos, [ref], asian_support=True).score

    return score


if __name__ == "__main__":
    bleu = {}
    ter = {}
    fairseq_out = 'out/en_vi/'
    best_out = 'out/best_en_vi/'
    filenames = next(walk(fairseq_out), (None, None, []))[2]
    ref = read_target("PhoMT_data/test/test.vi")
    for i in tqdm.tqdm(range(len(filenames))):
        #print("================================================")
        #print(filenames[i])
        #print("start parsing from fairseq generate")
        source_out, hypos_out, scores_out = parse_fairseq_gen(fairseq_out + filenames[i])
        #print("getting best hypothesis")
        best_hypos, best_scores = get_best_hyps(scores_out, hypos_out, 1, 1, 5)
        f = open(best_out + filenames[i],'w')
        for j in best_hypos:
            f.write(j)
            f.write('\n')
        f.close()
        #print("calculate Bleu score")
        if filenames[i].split(".")[0].isnumeric():
            if len(best_hypos) == len(ref):
                bleu[filenames[i].split(".")[0]] = eval_metric("bleu", best_hypos, ref)
            else:
                bleu[filenames[i].split(".")[0]] = 0
                print("Checkpoints {} errors not generate enough hypo, please do it manual".format(filenames[i].split(".")[0]))
        #print("calculate Ter score")
        #ter[filenames[i].split()[0]] = eval_metric("ter", best_hypos, ref)
        #print("checkpoints {}: Bleu = {} TER = {}".format( filenames[i].split(".")[0],bleu[ filenames[i].split(".")[0]],ter[ filenames[i].split()[0]]))

    path_bleu = "./out/en_vi_bleu_PhoMT_sentencepiece.json"
    path_ter = "./out/en_vi_ter_PhoMT_sentencepiece.json"
    
    if bleu:
        # Serializing json 
        json_object = json.dumps(bleu, indent = 4)
        # Writing to sample.json
        with open(path_bleu, "w") as outfile:
            outfile.write(json_object)
    if ter:
        # Serializing json 
        json_object = json.dumps(ter, indent = 4)
        
        # Writing to sample.json
        with open(path_ter, "w") as outfile:
            outfile.write(json_object)

    