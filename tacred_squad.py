import json
import pickle

# RELS = 'per:age'
# RELS = 'org:founded_by'
# RELS = 'per:employee_of'
RELS = 'org:founded'
# RELS = 'per:date_of_birth'
# RELS = 'per:schools_attended'



RELS_TO_ANNOTATE = [RELS]
REL_TO_WRITE = "_".join(RELS.split(':'))




ALL_RELATIONS_TYPES = {'per:title': ['PERSON', 'TITLE'], 'org:top_members/employees': ['ORGANIZATION', 'PERSON'],
                       'org:country_of_headquarters': ['ORGANIZATION', 'COUNTRY'], 'per:parents': ['PERSON', 'PERSON'],
                       'per:age': ['PERSON', 'NUMBER'], 'per:countries_of_residence': ['PERSON', 'COUNTRY'],
                       'per:children': ['PERSON', 'PERSON'], 'org:alternate_names': ['ORGANIZATION', 'ORGANIZATION'],
                       'per:charges': ['PERSON', 'CRIMINAL_CHARGE'], 'per:cities_of_residence': ['PERSON', 'CITY'],
                       'per:origin': ['PERSON', 'NATIONALITY'], 'org:founded_by': ['ORGANIZATION', 'PERSON'],
                       'per:employee_of': ['PERSON', 'ORGANIZATION'], 'per:siblings': ['PERSON', 'PERSON'],
                       'per:alternate_names': ['PERSON', 'PERSON'], 'org:website': ['ORGANIZATION', 'URL'],
                       'per:religion': ['PERSON', 'RELIGION'], 'per:stateorprovince_of_death': ['PERSON', 'LOCATION'],
                       'org:parents': ['ORGANIZATION', 'ORGANIZATION'],
                       'org:subsidiaries': ['ORGANIZATION', 'ORGANIZATION'], 'per:other_family': ['PERSON', 'PERSON'],
                       'per:stateorprovinces_of_residence': ['PERSON', 'STATE_OR_PROVINCE'],
                       'org:members': ['ORGANIZATION', 'ORGANIZATION'],
                       'per:cause_of_death': ['PERSON', 'CAUSE_OF_DEATH'],
                       'org:member_of': ['ORGANIZATION', 'LOCATION'],
                       'org:number_of_employees/members': ['ORGANIZATION', 'NUMBER'],
                       'per:country_of_birth': ['PERSON', 'COUNTRY'],
                       'org:shareholders': ['ORGANIZATION', 'ORGANIZATION'],
                       'org:stateorprovince_of_headquarters': ['ORGANIZATION', 'STATE_OR_PROVINCE'],
                       'per:city_of_death': ['PERSON', 'CITY'], 'per:date_of_birth': ['PERSON', 'DATE'],
                       'per:spouse': ['PERSON', 'PERSON'], 'org:city_of_headquarters': ['ORGANIZATION', 'CITY'],
                       'per:date_of_death': ['PERSON', 'DATE'], 'per:schools_attended': ['PERSON', 'ORGANIZATION'],
                       'org:political/religious_affiliation': ['ORGANIZATION', 'RELIGION'],
                       'per:country_of_death': ['PERSON', 'COUNTRY'], 'org:founded': ['ORGANIZATION', 'DATE'],
                       'per:stateorprovince_of_birth': ['PERSON', 'STATE_OR_PROVINCE'],
                       'per:city_of_birth': ['PERSON', 'CITY'], 'org:dissolved': ['ORGANIZATION', 'DATE']}


def get_span_of_subj_obj(sample):
    subj_span = " ".join(sample['token'][sample['subj_start']:sample['subj_end']+1])
    obj_span = " ".join(sample['token'][sample['obj_start']:sample['obj_end']+1])
    return [subj_span, obj_span]


def get_indx_and_span_of_subj_obj(sample):
    subj_span = " ".join(sample['token'][sample['subj_start']:sample['subj_end']+1])
    obj_span = " ".join(sample['token'][sample['obj_start']:sample['obj_end']+1])
    return [(subj_span, sample['subj_start'], sample['subj_end']), (obj_span, sample['obj_start'], sample['obj_end'])]

with open("/home/nlp/amirdnc/data/squad2/train-v2.0.json") as json_file:
    squad_dev = json.load(json_file)




newDict = squad_dev.copy()
newDict['data'] = squad_dev['data'][0:1]
newDict['data'][0]['paragraphs'] = squad_dev['data'][0]['paragraphs'][0:2]

newDict['data'][0]['paragraphs'][0]['qas'] = squad_dev['data'][0]['paragraphs'][0]['qas'][0:2]
newDict['data'][0]['paragraphs'][1]['qas'] = squad_dev['data'][0]['paragraphs'][1]['qas'][0:2]


newDict['data'][0]['paragraphs'] = []

q1_org_founded = lambda org,date: 'When was ' + org + ' founded?'
q2_org_founded = lambda org,date: 'What was founded on ' + date + '?'
q3_org_founded = lambda org,date: f'On what date was {org} founded?'

qg1_org_founded = lambda org,date: f'What organization was founded on some date ?'  #  TODO: ask yoav
qg4_org_founded = lambda org,date: f"What organization's founding date is mentioned ?"  #  TODO: ask yoav
qg2_org_founded = lambda org,date: 'On what date was an organization founded?'
qg3_org_founded = lambda org,date: 'When was any organization founded?'

q1_per_age = lambda per,age: "What is " + per + "'s age?"
q2_per_age = lambda per,age: "Whose age is "+ age + "?"

qg1_per_age = lambda per,age: f"Which number in the test is an age?"
qg2_per_age = lambda per,age: f"Whose age is mentioned in the text?" #GOOD

q1_per_date_of_birth = lambda per,date_of_birth: "What is " + per + "'s date of birth?"
q2_per_date_of_birth = lambda per,date_of_birth: "Who was born on " + date_of_birth + "?"

qg1_per_date_of_birth = lambda per,date_of_birth: f"Whose date of birth is mentioned?"
qg2_per_date_of_birth = lambda per,date_of_birth: f"When was anyone born?"

q1_org_founded_by = lambda org,per: "Who founded " + org + "?"
q2_org_founded_by = lambda org,per: "What did " + per + " found?"
# q2_org_founded_by = lambda org,per: "What was founded by " + per + "?"

qg1_org_founded_by = lambda org,date: 'Who founded an organization?'
qg2_org_founded_by = lambda org,date: f"What organization someone found?"
qg3_org_founded_by = lambda org,date: f"Which organization's founder is mentioned?"

q1_per_schools_attended = lambda per,org: "Which school did " + per + " attend?"
q2_per_schools_attended = lambda per,org: "Who attended " + org + "?"

qg1_per_schools_attended = lambda per,org: "Which school did someone attend?"
qg2_per_schools_attended = lambda per,org: "Who attended a school?"

q1_per_employee_of = lambda per,org: "Who employs " + per + "?"
q2_per_employee_of = lambda per,org: "Who is employee of " + org + "?"

qgg1_per_employee_of = lambda per,org: "Who employs someone?"
qgg2_per_employee_of = lambda per,org: "Who was employed by an organization?"

qg1_per_employee_of = lambda per,org: "Which organization's employee is mentioned?"
qg2_per_employee_of = lambda per,org: "Whose employing organization is mentioned?"

# q1_per_employee_of = lambda per,org: "What company does " + per + " work for?"
# q2_per_employee_of = lambda per,org: "Who works for " + org + "?"





questions_dic = {}

questions_dic['org:founded'] = [[q1_org_founded, 'subj'], [q2_org_founded, 'obj']]
questions_dic['per:age'] = [[q1_per_age, 'subj'], [q2_per_age, 'obj']]
questions_dic['per:date_of_birth'] = [[q1_per_date_of_birth, 'subj'], [q2_per_date_of_birth, 'obj']]
questions_dic['org:founded_by'] = [[q1_org_founded_by, 'subj'], [q2_org_founded_by, 'obj']]
questions_dic['per:schools_attended'] = [[q1_per_schools_attended, 'subj'], [q2_per_schools_attended, 'obj']]
questions_dic['per:employee_of'] = [[q1_per_employee_of, 'subj'], [q2_per_employee_of, 'obj']]






count_with_rels = 0
if '__main__' ==__name__:
    with open("/home/nlp/amirdnc/data/tacred/data/json/train.json") as tacred_test_file:
        tacred_real_samples = json.load(tacred_test_file)
    for samp in tacred_real_samples:

        curr_context = ' '.join(samp['token'])
        curr_paragraph = {'context': curr_context, 'qas': []}

        subj, obj = get_span_of_subj_obj(samp)

        for idx, (q, question_about) in enumerate(questions_dic[RELS]):

            curr_question = q(subj, obj)


            if samp['subj_type'] != ALL_RELATIONS_TYPES[RELS][0] or samp['obj_type'] != ALL_RELATIONS_TYPES[RELS][1]:
                continue

            if samp['relation'] != RELS and samp['relation'] != "no_relation":
                continue



            print(samp['subj_type'],samp['obj_type'])
            print()

            if question_about == 'subj':
                if samp['relation'] == RELS:
                    count_with_rels += 1
                    answers = [{'answer_start': curr_context.find(obj), 'text': obj}]

                elif samp['relation'] != RELS:
                    answers = []


            elif question_about == 'obj':
                if samp['relation'] == RELS:
                    count_with_rels += 1
                    answers = [{'answer_start': curr_context.find(subj), 'text': subj}]

                elif samp['relation'] != RELS:
                    answers = []


            is_impossible = True if len(answers) == 0 else False

            curr_qa = {'id': samp['id'] + '_' + str(idx), 'is_impossible': is_impossible, 'question': q(subj, obj),
                       'answers': answers}
            curr_paragraph['qas'].append(curr_qa)


        if curr_paragraph['qas'] != []:
            newDict['data'][0]['paragraphs'].append(curr_paragraph)

    print(json.dumps(newDict, indent=4, sort_keys=True))




    # with open(REL_TO_WRITE+"_tacred"+'.json', 'w') as outfile:
    #     json.dump(newDict, outfile)


    print("count_with_rels", count_with_rels)

    print(RELS)