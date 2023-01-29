import json
import random

if __name__ == '__main__':
    p1 = '/home/nlp/amirdnc/data/squad2/tac_trainv3.json'
    p2 = '/home/nlp/amirdnc/data/squad2/train-v2.0.json'
    out = '/home/nlp/amirdnc/data/squad2/unified_train.json'
    with open(p1, 'r') as f:
        d1 = json.load(f)
    with open(p2, 'r') as f:
        d2 = json.load(f)

    d1['data'].extend(d2['data'])
    random.shuffle(d1['data'])
    d1['version'] = 'v2.2'
    with open(out, 'w') as f:
        json.dump(d1, f)
    print('done')