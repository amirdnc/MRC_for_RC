import copy
import json
import os
import re
import string
from collections import defaultdict
import numpy as np
from transformers.data.metrics.squad_metrics import get_raw_scores, apply_no_ans_threshold, make_eval_dict, merge_eval, \
    find_all_best_thresh


class RelData:
    def __init__(self):
        self.true_number = 0
        self.algo_pos = 0
        self.algo_correct = 0

    def recall(self):
        if self.true_number == 0:
            return 1
        else:
            return self.algo_correct / self.true_number

    def precision(self):
        if self.algo_pos == 0:
            return 1
        else:
            return self.algo_correct / self.algo_pos

    def f1(self):
        p = self.precision()
        r = self.recall()
        if p == 0 and r == 0:
            return 0
        return 2 * ((p * r) / (p + r))


def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def validate_strings(s1, s2):  # see if one string is contained in the other
    if type(s2) != list:
        s2 = [s2]
    else:
        s2 = [x['text'] for x in s2]
    for cur_s in s2:
        s1 = normalize_answer(s1)
        cur_s = normalize_answer(cur_s)
        if s1 in cur_s or cur_s in s1:
            return True
    return False


def get_rel(cur_preds, q, subj, obj):
    p1 = cur_preds[str(q['id'])]
    if p1 and (validate_strings(p1, subj) or validate_strings(p1, obj)):
        return q['rel']
    else:
        return None


def eval_na(qas, subj, obj, na_probs):
    qas = sorted(qas, key=lambda x: x['id'])
    pred_rels = []
    for i in range(0, len(qas), 2):
        q1 = qas[i]
        q2 = qas[i + 1]
        # if get_rel(q1, subj, obj) and get_rel(q1, subj, obj) == get_rel(q2, subj, obj):
        #     pred_rels.append(get_rel(q1, subj, obj))
        pred_rels.append((get_rel(q1, subj, obj), na_probs[q1['id']]))
        pred_rels.append((get_rel(q2, subj, obj), na_probs[q2['id']]))
    l = list(dict.fromkeys([x for x in pred_rels if x[0]]))
    if not l:
        return []
    l = sorted(l, key=lambda x: x[1])
    return [l[0][0]]


def eval_group(cur_predictions, qas, subj, obj):
    qas = sorted(qas, key=lambda x: x['id'])
    pred_rels = []
    for i in range(0, len(qas), 2):
        q1 = qas[i]
        q2 = qas[i + 1]
        # if get_rel(q1, subj, obj) and get_rel(q1, subj, obj) == get_rel(q2, subj, obj):
        #     pred_rels.append(get_rel(q1, subj, obj))
        pred_rels.append(get_rel(cur_predictions, q1, subj, obj))
        pred_rels.append(get_rel(cur_predictions, q2, subj, obj))
    l = list(dict.fromkeys([x for x in pred_rels if x]))
    if l:
        return [l[0]]
    return []


def eval_single(qas, rels):
    for qa in qas:
        pred = preds[qa['id']]
        # if qa['is_impossible']:
        #     continue
        if pred == '' and qa['is_impossible']:
            continue
        if pred != '' and qa['is_impossible']:
            rels[qa['rel']].algo_pos += 1
            rels[all_rel].algo_pos += 1
            continue

        if pred == '' and not qa['is_impossible']:
            rels[qa['rel']].true_number += 1
            rels[all_rel].true_number += 1
            continue

        if validate_strings(pred, qa['answers'][0][
            'text']):  # pred.lower() == qa['answers'][0]['text'].lower():  # this is good only for the tacred version - we assume single question
            rels[qa['rel']].true_number += 1
            rels[qa['rel']].algo_pos += 1
            rels[qa['rel']].algo_correct += 1
            rels[all_rel].true_number += 1
            rels[all_rel].algo_pos += 1
            rels[all_rel].algo_correct += 1
            continue

        if not pred:
            rels[qa['rel']].true_number += 1
            rels[qa['rel']].algo_pos += 1
            rels[all_rel].true_number += 1
            rels[all_rel].algo_pos += 1
            continue

        else:
            rels[qa['rel']].true_number += 1
            rels[qa['rel']].algo_pos += 1
            rels[all_rel].true_number += 1
            rels[all_rel].algo_pos += 1
            continue


def trim_preds(preds, na_probs, thresh):
    new_preds = copy.deepcopy(preds)
    for k in new_preds:
        if na_probs[k] > thresh:
            new_preds[k] = ''
    return new_preds


def trim_preds_multi_thresh(preds, na_probs, threshs):
    new_preds = copy.deepcopy(preds)

    for r in gold:
        for q in r['paragraphs'][0]['qas']:
            id = str(q['id'])
            if na_probs[id] > threshs[q_rel(q)]:
                new_preds[id] = ''
    return new_preds


def eval_group_rels(cur_predictions, cur_rel):
    title = cur_rel['title']
    p = cur_rel['paragraphs'][0]
    if title != 'no_relation':
        rels[title].true_number += 1
        rels[all_rel].true_number += 1
    if 'subj' in p:
        subj, obj = p['subj'], p['obj']
    else:
        subj, obj = p['subj'], p['obj']
    pred_rels = eval_group(cur_predictions, p['qas'], p['subj'], p['obj'])
    if title in pred_rels:
        rels[title].algo_pos += 1
        rels[title].algo_correct += 1
        rels[all_rel].algo_pos += 1
        rels[all_rel].algo_correct += 1
        pred_rels.remove(title)
    for cur_rel in pred_rels:
        rels[cur_rel].algo_pos += 1
        rels[all_rel].algo_pos += 1

def q_rel(q):
    return q['rel'] + str(int(q['id']) % 2)

def eval_group_single_rel(cur_preds, cur_rel, rels):
    title = cur_rel['title']
    for p in  cur_rel['paragraphs']:
        for q in p['qas']:
            if 'subj' in p:  # tacred
                subj, obj = p['subj'], p['obj']
            else :  #docred
                subj, obj = q['subj'], q['answers']
            pred_rel = get_rel(cur_preds, q, subj, obj)
            if q['is_impossible'] and not pred_rel:
                continue
            if q['is_impossible'] and pred_rel:
                rels[q_rel(q)].algo_pos += 1
                rels[all_rel].algo_pos += 1
                continue
            rels[q_rel(q)].true_number += 1
            rels[all_rel].true_number += 1
            if pred_rel:
                rels[q_rel(q)].algo_pos += 1
                rels[q_rel(q)].algo_correct += 1
                rels[all_rel].algo_pos += 1
                rels[all_rel].algo_correct += 1
                continue
            else:
                rels[q_rel(q)].algo_pos += 1
                rels[all_rel].algo_pos += 1
                continue
def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation
def create_thresholds():
    outputs = defaultdict(list)
    counter = 0
    total = 0
    for thresh in np.linspace(-10, 30, 150):
        all_rel = 'all_rel'
        # thresh = float(thresh)
        rels = defaultdict(RelData)
        cur_preds = trim_preds(preds, na_probs, thresh)
        for r in gold:
            eval_group_single_rel(cur_preds, r, rels)
            # eval_group_rels(cur_preds, r)

        # for rel, values in rels.items():
        #     print('{}: p: {}, r:{}, f1:{}'.format(rel, values.precision(), values.recall(), values.f1()))
        for r, val in rels.items():
            outputs[r].append((val.precision(), val.recall(), val.f1(), thresh))
    best_f1 = {}
    for r, val in outputs.items():
         best_f1[r] = max(val, key = lambda x: x[-2])
    for k, v in outputs.items():
        print('{}: {}'.format(k, v))
    for k, v in best_f1.items():
        print('{}: {}'.format(k, v))
    for k, v in best_f1.items():
        best_f1[k] = v[-1]
    print(best_f1)
    with open(path_to_thresh, 'w') as f:
        json.dump(best_f1, f)
    return best_f1

def LCS(stringA, stringB):
    lenStringA = 1 + len(stringA)
    lenStringB = 1 + len(stringB)

    matrix = [[0] * (lenStringB) for i in range(lenStringA)]

    substringLength = 0
    endIndex = 0

    for aIndex in range(1, lenStringA):
        for bIndex in range(1, lenStringB):

            if stringA[aIndex - 1] == stringB[bIndex - 1]:

                matrix[aIndex][bIndex] = matrix[aIndex - 1][bIndex - 1] + 1

                if matrix[aIndex][bIndex] > substringLength:

                    substringLength = matrix[aIndex][bIndex]
                    endIndex = aIndex

            else:

                matrix[aIndex][bIndex] = 0

    return stringA[endIndex - substringLength: endIndex]
if __name__ == '__main__':
    dir = '/home/nlp/amirdnc/code/SpanBERT2/test'
    dir = '/home/nlp/amirdnc/code/SpanBERT2/unified_out'
    dir = '/home/nlp/amirdnc/code/SpanBERT2/bert_base_tacred2'
    # dir = '/home/nlp/amirdnc/code/SpanBERT2/bert_large_tac'
    dir = '/home/nlp/amirdnc/code/qa/span_bert2/test'
    # dir = "/home/nlp/amirdnc/code/qa/tac_raw"
    # dir = '/home/nlp/amirdnc/code/SpanBERT2/ref_model'
    # dir = '/home/nlp/amirdnc/code/qa/bert_large2/test'
    # gold_path = '/home/nlp/amirdnc/data/squad2/tac_test.json'
    gold_path = '/home/nlp/amirdnc/data/squad2/tac_trainv3.json'
    # gold_path = '/home/nlp/amirdnc/data/squad2/doc_devv2.json'
    gold_path = '/home/nlp/amirdnc/data/squad2/tac_testv4.json'
    pred_path = dir + '/predictions_.json'
    # pred_path = dir + '/predictions_train.json'
    # pred_path = '/home/nlp/amirdnc/code/SpanBERT2/test/predictions_train.json'
    na_probs_path = dir + '/na_probs.json'
    # na_probs_path = dir + '/na_probs_train.json'

    with open(gold_path, 'r') as f:
        gold = json.load(f)
        gold = gold['data']

    with open(pred_path, 'r') as f:
        preds = json.load(f)

    with open(na_probs_path, 'r') as f:
        na_probs = json.load(f)

    ####################
    # ev = squad_evaluate(gold, preds, na_probs)
    # print(ev)
    # exit()
    ####################
    thresh = -1.530  # train
    # cur_preds = trim_preds(preds, na_probs,thresh)
    # thresh = -2.5  #dev
    # thresh = -1  # test



    ##############################################
    all_rel = 'all_rel'
    rels = defaultdict(RelData)
    path_to_thresh = dir + '/thresh_q.json'
    if os.path.exists(path_to_thresh):
        with open(path_to_thresh, 'r') as f:
            na_threshs = json.load(f)
    else:
        na_threshs = create_thresholds()
        with open(path_to_thresh, 'w') as f:
            json.dump(na_threshs, f)

    rels = defaultdict(RelData) #reset the varible
    cur_preds = trim_preds_multi_thresh(preds, na_probs, na_threshs)
    # thresh = -0.3448275862068959
    # cur_preds = trim_preds(preds, na_probs, thresh)
    for r in gold:
        eval_group_rels(cur_preds, r)
        # eval_group_single_rel(cur_preds, r)

    for rel, values in rels.items():
        print('{}: p: {}, r:{}, f1:{}'.format(rel, values.precision(), values.recall(), values.f1()))
    exit()
    ################################################
    # outputs = defaultdict(list)
    # counter = 0
    # total = 0
    # for thresh in np.linspace(-10, 30, 150):
    #     all_rel = 'all_rel'
    #     # thresh = float(thresh)
    #     rels = defaultdict(RelData)
    #     cur_preds = trim_preds(preds, na_probs, thresh)
    #     for r in gold:
    #         eval_group_single_rel(cur_preds, r, rels)
    #         # eval_group_rels(cur_preds, r)
    #
    #     # for rel, values in rels.items():
    #     #     print('{}: p: {}, r:{}, f1:{}'.format(rel, values.precision(), values.recall(), values.f1()))
    #     for r, val in rels.items():
    #         outputs[r].append((val.precision(), val.recall(), val.f1(), thresh))
    # best_f1 = {}
    # for r, val in outputs.items():
    #      best_f1[r] = max(val, key = lambda x: x[-2])
    # for k, v in outputs.items():
    #     print('{}: {}'.format(k, v))
    # for k, v in best_f1.items():
    #     print('{}: {}'.format(k, v))
    # for k, v in best_f1.items():
    #     best_f1[k] = v[-1]
    # print(best_f1)
    # with open(dir + 'best_f1_train.json', 'w') as f:
    #     json.dump(best_f1, f)