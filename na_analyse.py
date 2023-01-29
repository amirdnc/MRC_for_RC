import json
import matplotlib.pyplot as plt
if __name__ == '__main__':

    # dir = '/home/nlp/amirdnc/code/SpanBERT2/test'
    dir = '/home/nlp/amirdnc/code/SpanBERT2/unified_out'
    # dir = '/home/nlp/amirdnc/code/SpanBERT2/ref_model'
    # gold_path = '/home/nlp/amirdnc/data/squad2/tac_test.json'
    gold_path = '/home/nlp/amirdnc/data/squad2/tac_dev.json'
    # gold_path = '/home/nlp/amirdnc/data/squad2/tac_trainv3.json'
    pred_path = dir + '/predictions_dev.json'
    # pred_path = '/home/nlp/amirdnc/code/SpanBERT2/test/predictions_train.json'
    na_probs_path = dir + '/na_probs_dev.json'
    with open(na_probs_path, 'r') as f:
        na_probs = json.load(f)

    with open(gold_path, 'r') as f:
        gold = json.load(f)
        gold = gold['data']

    with open(dir + 'best_f1_dev.json', 'r') as f:
        na_threshs = json.load(f)

    outputs, negs, pos = [], [], []
    rel = 'per:employee_of'
    # rel = 'per:siblings'
    # rel = 'per:religion'
    rel = 'org:founded_by'
    for d in gold:
        p = d['paragraphs'][0]['qas']
        for q in p:
            if q['rel'] == rel:
                outputs.append(na_probs[q['id']])
                if q['is_impossible']:
                    negs.append(na_probs[q['id']])
                else:
                    pos.append(na_probs[q['id']])
    plt.hist([outputs, negs, pos], label=['All', 'Negative', 'Positive'])
    plt.axvline(x=na_threshs[rel], linewidth=2, color='r')
    plt.legend()
    plt.title(rel)
    plt.show()





