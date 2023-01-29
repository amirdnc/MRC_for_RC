import json
import re
import string
from sklearn.metrics import classification_report

# from qa_tacred_amir import questions_dic



# RELS = 'per:age'
# RELS = 'org:founded_by'
# RELS = 'per:employee_of'
RELS = 'org:founded'
# RELS = 'per:date_of_birth'
from tools.tacred_squad import questions_dic

# RELS = 'per:schools_attended'


tacred_dev = '/home/nlp/amirdnc/data/tacred/data/json/dev.json'
squad_gold = '/home/nlp/amirdnc/data/squad2/tac_dev.json'
squad_pred = '/home/nlp/amirdnc/code/SpanBERT2/test/predictions.json'

with open(tacred_dev) as tacred_test_file:
    tacred_real_samples_list = json.load(tacred_test_file)

tacred_real_samples = {samp['id']: samp for samp in tacred_real_samples_list}



with open(squad_gold) as json_file:
    gold_data = json.load(json_file)

with open(squad_pred) as json_file:
    preds_data = json.load(json_file)



def get_span_of_subj_obj(sample):
    subj_span = " ".join(sample['token'][sample['subj_start']:sample['subj_end']+1])
    obj_span = " ".join(sample['token'][sample['obj_start']:sample['obj_end']+1])
    return subj_span, obj_span


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
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






annotated_data = {}
for item in gold_data['data']:
    for paragraph in item['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:

            new_qa = qa.copy()
            new_qa['context'] = context
            new_qa['subj'] = paragraph['subj']
            new_qa['obj'] = paragraph['obj']
            annotated_data[qa['id']] = new_qa


#  add new id to tacred
for k, v in annotated_data.items():
    tacred_real_samples[k] = tacred_real_samples[v['id_rel']]

count_err = 0
count_all_err = 0

error_else = 0
error_recognize = 0

error_else_2 = 0
error_recognize_2 = 0

marged_preds = {}

marged_preds_in_ans = {}
counter = 0
total = 0
for pred_id, pred in preds_data.items():
    total += 1
    if pred_id not in annotated_data:
        counter += 1
        continue
    aaa = annotated_data[pred_id]
    gold_answers = [a['text'] for a in annotated_data[pred_id]['answers'] if normalize_answer(a['text'])]
    if not gold_answers:
        # For unanswerable questions, only correct answer is empty string
        gold_answers = ['']


    slice_pred_id = pred_id.split('_')[0]


    if slice_pred_id not in marged_preds_in_ans:
        marged_preds_in_ans[slice_pred_id] = "no_relation"


    # subj, obj = get_span_of_subj_obj(tacred_real_samples[slice_pred_id])
    subj, obj = annotated_data[slice_pred_id]['subj'], annotated_data[slice_pred_id]['obj']
    for idxxx, (q, question_about) in enumerate(questions_dic[RELS]):

        curr_question = q(subj, obj)

        if annotated_data[pred_id]['question'] == curr_question and question_about == 'subj':
            if obj in pred or (pred != '' and pred in obj):
                marged_preds_in_ans[slice_pred_id] = RELS
                # print("TTTTTTTT_22222_22222")
                # print(curr_question)
                # print(obj, "  --  ", pred)
                # print("--------------------")



        elif annotated_data[pred_id]['question'] == curr_question and question_about == 'obj':
            if subj in pred or (pred != '' and pred in subj):
                marged_preds_in_ans[slice_pred_id] = RELS
                # print("TTTTTTTT_11111_11111")
                # print(curr_question)
                # print(subj, "  --  ", pred)
                # print(print("--------------------"))



y_true_in_ans = []
y_pred_in_ans = []

for pred_id in marged_preds_in_ans:

    if tacred_real_samples[pred_id]['relation'] != RELS:
        y_true_in_ans.append('no_relation')
    else:
        y_true_in_ans.append(tacred_real_samples[pred_id]['relation'])

    y_pred_in_ans.append(marged_preds_in_ans[pred_id])

print('counter is: ', counter/ total)
print(classification_report(y_true_in_ans, y_pred_in_ans))
print()

print("num of positive predictions:  ", sum([1 for y in y_pred_in_ans if y == RELS]))

print()

print("num of negative predictions:  ", sum([1 for y in y_true_in_ans if y != RELS]))

print()

print(RELS)


#                   DEBUG !!!!

# for pred_id, pred in preds_data.items():
#     slice_pred_id = pred_id.split('_')[0]
#     gold_answers = [a['text'] for a in annotated_data[pred_id]['answers'] if normalize_answer(a['text'])]
#
#     if marged_preds_in_ans[slice_pred_id] == RELS and tacred_real_samples[slice_pred_id]['relation'] != RELS:
#
#         print("dadasgd")
#         print(slice_pred_id)
#         print(annotated_data[pred_id]['context'])
#         print()
#         print(annotated_data[pred_id]['question'])
#         print()
#         print(gold_answers, "  ---  ", "'" + pred + "'")







