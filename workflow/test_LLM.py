import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys,os
from tqdm import tqdm
from src.models import LLMModel


map_prompts = json.load(open("../prompt/test_prompt.json","r",encoding="utf-8"))
map = {"awareness":"../data/awareness_translated_large.json",
       "bias":"../data/bias_translated_large.json"}
parser = argparse.ArgumentParser(description="DAG")

parser.add_argument(
    "--Eval_model",
    type=str,
    help="Eval_model",
    required=True
)
parser.add_argument(
    "--mode",
    type=str,
    help="mode",
    required=True,
    choices=["bias","awareness"]
)
#["mode","awareness"]


args = parser.parse_args()
mode = args.mode
datas = json.load(open(map[mode],"r+",encoding="utf-8"))

Eval_model = args.Eval_model
model_path = Eval_model
model = LLMModel(model_path,
                max_new_tokens=15000,
                temperature=0.7,
                model_dir=None,
                openai_key=None,
                sleep_time=3,
                )


openai_list = ["o1-preview","o1-mini","yi-lightning","gpt-4-turbo","gpt-4o","o1-pro-all","deepseek-v3","deepseek-r1","o3-mini-2025-01-31","o1-2024-12-17"]


model_name = Eval_model.split("/")[-1]
if os.path.exists(f"../data/_out{mode}_{model_name}_out.json"):
    f = open(f"../data/_out{mode}_{model_name}_out.json", "r",encoding="utf-8")
    already_file = json.load(f)
    if Eval_model in openai_list:
        dic_id = {}
        for key in datas:
            dic_id[key] = {}
        for key, value in already_file.items():
            for item in value:
                dic_id[key][item["number_id"]] = item
        already_idx = [item["number_id"] for key in already_file for item in already_file[key]]
else:
        already_file = {}
        dic_id = {}
        for key in datas:
            dic_id[key] = {}
            already_idx = []

results = defaultdict(list)


def extract_boxed_answers(text):
    answers = []
    for piece in text.split('\\textbf{')[1:]:
        # print(f"{piece=}")
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    if len(answers)==0:
        return ""
    elif len(answers)==1:
        return answers[0]
    else:
        return answers[1]
    # return answers[1]


question_types = ["original_question","counterfactual_question","confounding_question"]


languages_lis = ["English","Chinese","French","German","Italian","Spanish","Korean","Swedish","Japanese","Dutch","Polish","Norwegian","Indonesian"]
def call_openai_api(data):
    question,dic,key,question_type,language_now = data
    # print(f"{dic_id[key][dic["number_id"]]=}")
    if dic["number_id"] in dic_id[key] and f"{question_type}_{language_now}_out" in dic_id[key][dic["number_id"]]:
        return []
    raw_pred = model(question)[0]
    return raw_pred, dic, key, question_type, language_now


preds = []
answers = []
data_out = []
if mode == "bias":
    all_questions = []
    all_sent_dicts = []
    all_keys = []
    pre_ = []
    for key in datas:
        for sent_dict in datas[key]:
            if sent_dict["number_id"] in already_idx:
                continue
            else:
                for question_type in question_types:
                    for language_now in languages_lis:
                        prompts = map_prompts[mode][language_now]
                        prompt_eval = prompts
                        question = sent_dict[f"{question_type}_{language_now}"]
                        question = question.replace("**", "").strip()
                        question_prompt = prompt_eval.replace("{question}", question)
                        if Eval_model in openai_list:
                            pre_.append((question_prompt,sent_dict,key,question_type,language_now))

                        all_questions.append(question_prompt)
                all_sent_dicts.append(sent_dict)
                all_keys.append(key)

    if Eval_model in openai_list:
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交任务到线程池
            temp_data_out = []
            futures = [executor.submit(call_openai_api, data) for data in pre_]
            # 获取结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts", unit="prompt"):
                try:
                    result = future.result()

                    if len(result) == 0:
                        continue
                    out,dic,key,question_type,language_now = result
                    if dic["number_id"] not in dic_id[key]:
                        dic_id[key][dic["number_id"]] = dic
                        dic_id[key][dic["number_id"]][f"{question_type}_{language_now}_out"] = out

                    else:
                        dic_id[key][dic["number_id"]][f"{question_type}_{language_now}_out"] = out
                except Exception as e:
                    print(f"Error in future: {e}")
                    temp_data_out.append(f"Error in future: {e}")
        results = {key:[dic_id[key][item] for item in dic_id[key]] for key in dic_id.keys()}
        json.dump(results, open(f"../data/Eval_out/{mode}_{model_name}_out.json", "w+",encoding="utf-8"), indent=2, ensure_ascii=False)

    else:
        # Process all questions in one batch
        raw_preds = model(all_questions)
        # Assign predictions back to the corresponding sent_dicts
        for idx, (sent_dict, key) in enumerate(zip(all_sent_dicts, all_keys)):
            for inum, question_type in enumerate(question_types):
                for iq,language_now in enumerate(languages_lis):
                    pred_index = idx * len(question_types) * len(languages_lis) + inum * len(languages_lis) + iq
                    sent_dict[f"{question_type}_{language_now}_out"] = raw_preds[pred_index]
            print(f"sent_dict: {sent_dict}")
            results[key].append(sent_dict)
            json.dump(results, open(f"../data/Eval_out/{mode}_{model_name}_out.json", "w+"), indent=2, ensure_ascii=False)
elif mode == "awareness":
    all_questions = []
    all_sent_dicts = []
    all_keys = []
    pre_ = []
    for key in datas:
        for sent_dict in datas[key]:
            if sent_dict["number_id"] in already_idx:
                continue
            else:
                for question_type in question_types:
                    for language_now in languages_lis:
                        prompts = map_prompts[mode][language_now]
                        prompt_eval = prompts
                        question = sent_dict[f"{question_type}_{language_now}"]
                        question = question.replace("**", "").strip()
                        question_prompt = prompt_eval.replace("{question}", question)
                        if Eval_model in openai_list:
                            pre_.append((question_prompt,sent_dict,key,question_type,language_now))

                        all_questions.append(question_prompt)
                all_sent_dicts.append(sent_dict)
                all_keys.append(key)

    if Eval_model in openai_list:
        with ThreadPoolExecutor(max_workers=10) as executor:
            temp_data_out = []
            futures = [executor.submit(call_openai_api, data) for data in pre_]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts", unit="prompt"):
                try:
                    result = future.result()
                    if len(result) == 0:
                        continue
                    out,dic,key,question_type,language_now = result
                    if dic["number_id"] not in dic_id[key]:
                        dic_id[key][dic["number_id"]] = dic
                        dic_id[key][dic["number_id"]][f"{question_type}_{language_now}_out"] = out
                    else:
                        dic_id[key][dic["number_id"]][f"{question_type}_{language_now}_out"] = out
                except Exception as e:
                    print(f"Error in future: {e}")
                    temp_data_out.append(f"Error in future: {e}")

        results = {key:[dic_id[key][item] for item in dic_id[key]] for key in dic_id.keys()}
        json.dump(results, open(f"../data/Eval_out/{mode}_{model_name}_out.json", "w+",encoding="utf-8"), indent=2, ensure_ascii=False)

    else:
        # Process all questions in one batch
        raw_preds = model(all_questions)
        # Assign predictions back to the corresponding sent_dicts
        for idx, (sent_dict, key) in enumerate(zip(all_sent_dicts, all_keys)):
            for inum, question_type in enumerate(question_types):
                for iq,language_now in enumerate(languages_lis):
                    pred_index = idx * len(question_types) * len(languages_lis) + inum * len(languages_lis) + iq
                    sent_dict[f"{question_type}_{language_now}_out"] = raw_preds[pred_index]
            print(f"sent_dict: {sent_dict}")
            results[key].append(sent_dict)
            json.dump(results, open(f"../data/Eval_out/{mode}_{model_name}_out.json", "w+"), indent=2, ensure_ascii=False)


