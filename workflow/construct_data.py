from tqdm import tqdm
import json
from src.models import LLMModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import Verifier_generator,Verifier_rephrase

mode = "awareness"
#["awareness","bias"]
map = {"awareness":"../data/awareness_info.json",
       "bias":"../data/bias_info.json"}
f = open(map[mode], "r",encoding="utf-8")
js = json.load(f)
prompts = json.load(open("../prompt/construct_prompt.json","r",encoding="utf-8"))
prompt = prompts[f"prompt_{mode}"]
paraphraser = LLMModel("gpt-4o", max_new_tokens=15000, temperature=0.7)
selected_data = []
already_f_name = map[mode].replace("info","data")
f = open(already_f_name, "r",encoding="utf-8")
already_file = json.load(f)
if mode == "awareness":
       already_idx = []
       selected_data = js
       for key in already_file.keys():
              already_idx.extend([item["number_id"] for item in already_file[key]])

elif mode == "bias":
       already_idx = []
       selected_data = js
       for key in already_file.keys():
              already_idx.extend([item["number_id"] for item in already_file[key]])
       # for item in js:
       #        selected_data[item]=js[item]

print(f"{len(selected_data)=}")


def call_openai_api_bias(sent_dict,key,prompt_temp):
       if sent_dict["number_id"] in already_idx:
              return {}
       temp = prompt + sent_dict["bias_info"]
       print(f"{temp=}")
       pred = paraphraser(temp)
       print(f"{pred=}")
       # print(f"{pred.split('\n\n')=}")
       lis = pred.split("\n")
       print(f"{lis=}")
       lis  = [item for item in lis if len(item)!=0]
       if len(lis)==6:
              if len(lis[0].split(":"))!=2:
                     return {}
              a,b = lis[0].split(":")
              sent_dict["original_question"]=b.strip()
              c,d = lis[1].split(":")
              d = d.replace("*","")
              sent_dict[f"original_answer"]=d.strip()
              a,b = lis[2].split(":")
              sent_dict["counterfactual_question"]=b.strip()
              c,d = lis[3].split(":")
              d = d.replace("*","")
              sent_dict[f"counterfactual_answer"]=d.strip()
              a,b = lis[4].split(":")
              sent_dict["confounding_question"]=b.strip()
              c,d = lis[5].split(":")
              d = d.replace("*","")
              sent_dict[f"confounding_answer"]=d.strip()
              # already_file.append(sent_dict)
       else:
              return {}
       if not Verifier_rephrase(sent_dict):
              return {}
       print(f"{sent_dict=}")
       return sent_dict

def call_openai_api_awareness(sent_dict,key,prompt_temp):
       if sent_dict["number_id"] in already_idx:
              return {}
       temp = prompt_temp + sent_dict["awareness_info"]
       print(f"{temp=}")
       pred = paraphraser(temp)
       print(f"{pred=}")
       # exit()
       lis = pred.split("\n\n")
       print(f"{lis=}")
       # if "Cultural Tradition:" in lis[0]:
       #        sent_dict["behavior"] = lis[1].strip()
       #        a, b = lis[2].split("Cultural Awareness Question:")
       #        sent_dict["original_question"] = b.strip()
       #        c, d = lis[3].split(":")
       #        d = d.replace("*", "")
       #        sent_dict[f"original_answer"] = d.strip()
       # else:
       sent_dict["behavior"] = lis[0].strip()
       a, b = lis[1].split("Cultural Awareness Question:")
       sent_dict["original_question"] = b.strip()
       c, d = lis[2].split(":")
       d = d.replace("*", "")
       sent_dict[f"original_answer"] = d.strip()
       if not Verifier_generator(sent_dict):
              return {}
       dic_culture = {"Culture": sent_dict["awareness_info"], "Awareness_question": sent_dict["original_question"],
                      "Answer": sent_dict[f"original_answer"]}
       temp = prompt.format(**dic_culture)
       print(f"{temp=}")
       pred = paraphraser(temp)
       print(f"{pred=}")
       # exit()
       if "\n\n" in pred:
              lis = pred.split("\n\n")
       else:
              return {}
       # else:
       #        lis = pred.split("\n")

       print(f"{lis=}")
       if len(lis) == 4:
              if len(lis[0].split(":")) != 2:
                     return {}
              a, b = lis[0].split(":")
              sent_dict["counterfactual_question"] = b.strip()
              c, d = lis[1].split(":")
              d = d.replace("*", "")
              sent_dict[f"counterfactual_answer"] = d.strip()
              a, b = lis[2].split(":")
              sent_dict["confounding_question"] = b.strip()
              c, d = lis[3].split(":")
              d = d.replace("*", "")
              sent_dict[f"confounding_answer"] = d.strip()
       else:
              return {}
       if not Verifier_rephrase(sent_dict):
              return {}
       print(f"{sent_dict=}")
       return sent_dict


if mode == "bias":
       call_openai_api = call_openai_api_bias
elif mode == "awareness":
       call_openai_api = call_openai_api_awareness
with ThreadPoolExecutor(max_workers=50) as executor:
       # 提交任务到线程池
       temp_data_out = []
       if mode == "awareness":
              prompt_temp = prompts[f"prompt_{mode}_temp"]
              futures = [executor.submit(call_openai_api, sent_dict, key, prompt_temp)
                         for key, value in selected_data.items()
                         for sent_dict in value]
       elif mode == "bias":
              futures = [executor.submit(call_openai_api, sent_dict, key)
                         for key, value in selected_data.items()
                         for sent_dict in value]
       for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts", unit="prompt"):
              try:
                     result = future.result()
                     if len(result) == 0:
                            continue
                     if result["key"] not in already_file:
                            already_file[result["key"]] = [result]
                     else:
                            already_file[result["key"]].append(result)
              except Exception as e:
                     print(f"Error in future: {e}")
       f = open(f"../data/{mode}_data.json", "w+", encoding="utf-8")
       json.dump(already_file, f, ensure_ascii=False, indent=2)