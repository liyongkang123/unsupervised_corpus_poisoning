
import  json
from openai import OpenAI

'''
setx OPENAI_API_KEY "sk-"
'''
client = OpenAI(
  organization='org-',
api_key="sk-",
)

def llm_q_d(query, document):
    response = client.chat.completions.create(
        model='gpt-4o-mini', # gpt-4o-mini  or  gpt-4o
        messages=[
            {
                "role": "system",
                "content": "You are an expert in relevance assessment. I will provide you with a query posed by a user, followed by a document. Your task is to determine whether the document answers the user's question. Please respond directly and solely with 'Yes' or 'No'."},
            {
                "role": "user",
                "content": "Query: " + query + '\n Document: ' + document
            }
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )
    result = response.choices[0].message.content
    return result


def llm_d_d(document_1, document_2):
    response = client.chat.completions.create(
    model= 'gpt-4o',
    messages=[
        {
        "role": "system",
        "content": "You are an expert in relevance assessment. I will provide you with two documents, and you need to assess whether these two documents express the same information. Please respond directly and only with 'Yes' or 'No'."},
        {
        "role": "user",
        "content": "Document 1: "+document_1 +'\n Document 2: '+document_2
        }
    ],
    temperature=0.7,
    max_tokens=64,
    top_p=1
    )
    result = response.choices[0].message.content
    return result

# datasets_name_list = ['trec_dl19', "trec_dl20",   "nq",  "quora",  "fiqa", "webis-touche2020"]
# methods_list = ['random_noise', 'random_token','hotflip', 'unsupervised', ]
# seed_list = [2024, 2025, 2026]
datasets_name_list = ['webis-touche2020',]
methods_list = ['random_noise', 'random_token','hotflip', 'unsupervised', ] #'random_noise',
seed_list = [2024, 2025, 2026]


attack_mode_code = 'simlm-msmarco'
attack_split = 'test'
learning_rate = 0.0005
lm_loss_clip_max = 5.0
random_noise_rate = 0.5
random_token_replacement_rate = 0.3
for eval_datset in datasets_name_list:
    for method in methods_list:

        llm_q1,llm_q2 =[], []

        for seed in seed_list:
            print(eval_datset, method, seed)

            if method =='random_noise':
                attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                       f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}"
                                       f"_noise_rate-{random_noise_rate}.json")
            elif method == 'random_token':
                attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                       f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}"
                                       f"random_token-{random_token_replacement_rate}.json")

            elif method == 'hotflip':

                attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                       f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}.json")

            elif method == 'unsupervised':
                lm_loss_clip_max = 5.0
                attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                       f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}.json")
            print(attack_results_path)

            #
            with open(attack_results_path, 'r', encoding='utf-8') as file:
                attack_results = json.load(file)

            for q in attack_results.keys():
                top_1_target_text = attack_results[q]['top_1_target_text']
                adv_text = attack_results[q]['adv_text']
                q_query = attack_results[q]['query_text']
                result_q_d = llm_q_d(q_query, adv_text).lower()
                result_d_d = llm_d_d(adv_text, top_1_target_text).lower() 
                print( result_d_d, result_q_d)
                if 'no' in result_q_d:
                    llm_q1.append(1)
                else:
                    llm_q1.append(0)
                if 'no' in result_d_d:
                    llm_q2.append(1)
                else:
                    llm_q2.append(0)
        llm_q1_percent = sum(llm_q1)/len(llm_q1)
        llm_q2_percent = sum(llm_q2)/len(llm_q2) #
        print(f"{eval_datset} {method} llm_q1_percent: {llm_q1_percent} llm_q2_percent: {llm_q2_percent},seed {seed}")