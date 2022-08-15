
import numpy as np
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

if __name__ == "__main__":
    fairseq_out = 'out/en_vi/'
    best_hypos_path = './discriminator_train_data/best_hyp.vi'
    all_hypos_path = './discriminator_train_data/all_hyp.vi'
    source_out, hypos_out, scores_out = parse_fairseq_gen(fairseq_out)
    with open(all_hypos_path, "w") as outfile1:
        for sent1 in hypos_out:
            outfile1.write(sent1)
            outfile1.write('\n')

    best_hypos, best_scores = get_best_hyps(scores_out, hypos_out, 1, 1, 5)
    with open(best_hypos_path, "w") as outfile2:
        for sent2 in best_hypos:
            outfile2.write(sent2)
            outfile2.write('\n')
    
        