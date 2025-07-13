from transformers import AutoTokenizer
import json
from tqdm import tqdm
import argparse
import os
from src.models.modeling_xalma import XALMAForCausalLM
from src.utils import Verifier_translate
import torch

path = "haoranxu/X-ALMA"
model = XALMAForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
map = {"awareness":"../data/awareness_data.json",
       "bias":"../data/bias_data.json"}

parser = argparse.ArgumentParser(description="culture")
parser.add_argument(
    "--Language",
    type=str,
    help="Language",
    required=True,
    choices=["German","French","Italian","Spanish","Korean","Swedish","Japanese","Dutch","Polish","Norwegian","Indonesian","Chinese"]
)
parser.add_argument(
    "--bsz",
    type=int,
    help="bsz"
)
parser.add_argument(
    "--mode",
    type=str,
    help="mode",
    required=True,
    choices=["mode","awareness"]
)
args = parser.parse_args()
mode = args.mode
f = open(map[mode], "r",encoding="utf-8")
Language = args.Language

if os.path.exists(f"../data/Language/{mode}_translated_{Language}.json"):
    already_file = json.load(open(f"../data/Language/{mode}_translated_{Language}.json", 'r', encoding="utf-8"))
else:
    already_file = {}

already_idx = []
id2results = {}
selected_data = already_file
for key in already_file.keys():
    already_idx.extend([item["number_id"] for item in already_file[key]])
    for item in already_file[key]:
        id2results[item["number_id"]] = item
data_dic = json.load(f)


mapping_lan = {"Chinese":"zh","German":"de","French":"fr","Italian":"it","Spanish":"es","Korean":"ko","Swedish":"sv","Japanese":"ja","Dutch":"nl","Polish":"pl","Norwegian":"no","Indonesian":"id"}


def translate(contents,Language):
    messages_batch = [
        [
            {"role": "user",
             "content": f"Translate this from English to {Language}:\nEnglish: {input_text.strip()}\n{Language}:"}
        ] for input_text in contents
    ]
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in messages_batch
    ]
    input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, num_beams=5, do_sample=True, temperature=0.7,
                                   top_p=0.9, lang=mapping_lan[Language])
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, outputs)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

# bsz = 5


for Language in [args.Language]:
    if mode == "bias":
        BATCH_SIZE = args.bsz

        for key in tqdm(data_dic.keys()):
            sent_dicts = data_dic[key]

            # Process in batches of BATCH_SIZE
            for i in tqdm(range(0, len(sent_dicts), BATCH_SIZE)):
                batch = sent_dicts[i:i + BATCH_SIZE]

                # Collect data for batch processing
                batch_data_origins = []
                batch_indices = []
                batch_to_process = []

                # First pass: identify which items need processing
                for sent_dict in batch:
                    if sent_dict["number_id"] in already_idx and f"original_question_{Language}" in id2results[
                        sent_dict["number_id"]] and f"Verifier_out_{Language}" in id2results[sent_dict["number_id"]] and id2results[sent_dict["number_id"]][f"Verifier_out_{Language}"]==True:
                        continue
                    else:
                        # Store data for this item
                        data_origins = []

                        temp = sent_dict["original_question"].replace("**", "")
                        data_origins.append(temp)

                        temp = sent_dict["counterfactual_question"].replace("**", "")
                        data_origins.append(temp)

                        temp = sent_dict["confounding_question"].replace("**", "")
                        data_origins.append(temp)

                        batch_data_origins.append(data_origins)
                        batch_indices.append(len(batch_to_process))
                        batch_to_process.append(sent_dict)

                # If batch has items to process
                if batch_to_process:
                    # Flatten the list for batch translation
                    all_questions = [q for sublist in batch_data_origins for q in sublist]

                    # Batch translate all questions at once
                    all_translated = translate(all_questions, Language)

                    # Process results for each item
                    for idx, sent_dict in enumerate(batch_to_process):
                        # Each item has 3 questions, so calculate the starting index
                        start_idx = idx * 3

                        # Get the translated questions for this item
                        question_1_translated = all_translated[start_idx]
                        question_2_translated = all_translated[start_idx + 1]
                        question_3_translated = all_translated[start_idx + 2]

                        # Update the sent_dict with translations
                        sent_dict[f"original_question_{Language}"] = question_1_translated
                        sent_dict[f"counterfactual_question_{Language}"] = question_2_translated
                        sent_dict[f"confounding_question_{Language}"] = question_3_translated
                        if Verifier_translate(question_1_translated,sent_dict["original_question"],Language) and Verifier_translate(question_2_translated,sent_dict["counterfactual_question"],Language) and Verifier_translate(question_3_translated,sent_dict["confounding_question"],Language):
                            sent_dict[f"Verifier_out_{Language}"]=True
                        else:
                            sent_dict[f"Verifier_out_{Language}"]=False

                        # Update id2results
                        if sent_dict["number_id"] in id2results:
                            id2results[sent_dict["number_id"]].update(sent_dict)
                        else:
                            id2results[sent_dict["number_id"]] = sent_dict

                    # Write results after each batch is processed
                    write_dic = {}
                    for key, value in id2results.items():
                        if value["key"] in write_dic:
                            write_dic[value["key"]].append(value)
                        else:
                            write_dic[value["key"]] = [value]

                    json.dump(write_dic, open(f"../data/Language/{mode}_translated_{Language}.json", "w"), indent=2, ensure_ascii=False)

    elif mode == "awareness":
        BATCH_SIZE = args.bsz

        for key in tqdm(data_dic.keys()):
            sent_dicts = data_dic[key]

            # Process in batches of BATCH_SIZE
            for i in tqdm(range(0, len(sent_dicts), BATCH_SIZE)):
                batch = sent_dicts[i:i + BATCH_SIZE]

                # Collect data for batch processing
                batch_data_origins = []
                batch_indices = []
                batch_to_process = []

                # First pass: identify which items need processing
                for sent_dict in batch:
                    if sent_dict["number_id"] in already_idx and f"original_question_{Language}" in id2results[
                        sent_dict["number_id"]] and f"Verifier_out_{Language}" in id2results[sent_dict["number_id"]] and id2results[sent_dict["number_id"]][f"Verifier_out_{Language}"]==True:
                        continue
                    else:
                        # Store data for this item
                        data_origins = []

                        temp = sent_dict["original_question"].replace("**", "")
                        data_origins.append(temp)

                        temp = sent_dict["counterfactual_question"].replace("**", "")
                        data_origins.append(temp)

                        temp = sent_dict["confounding_question"].replace("**", "")
                        data_origins.append(temp)

                        batch_data_origins.append(data_origins)
                        batch_indices.append(len(batch_to_process))
                        batch_to_process.append(sent_dict)

                # If batch has items to process
                if batch_to_process:
                    # Flatten the list for batch translation
                    all_questions = [q for sublist in batch_data_origins for q in sublist]

                    # Batch translate all questions at once
                    all_translated = translate(all_questions, Language)

                    # Process results for each item
                    for idx, sent_dict in enumerate(batch_to_process):
                        # Each item has 3 questions, so calculate the starting index
                        start_idx = idx * 3

                        # Get the translated questions for this item
                        question_1_translated = all_translated[start_idx]
                        question_2_translated = all_translated[start_idx + 1]
                        question_3_translated = all_translated[start_idx + 2]

                        # Update the sent_dict with translations
                        sent_dict[f"original_question_{Language}"] = question_1_translated
                        sent_dict[f"counterfactual_question_{Language}"] = question_2_translated
                        sent_dict[f"confounding_question_{Language}"] = question_3_translated
                        if Verifier_translate(question_1_translated,sent_dict["original_question"],Language) and Verifier_translate(question_2_translated,sent_dict["counterfactual_question"],Language) and Verifier_translate(question_3_translated,sent_dict["confounding_question"],Language):
                            sent_dict[f"Verifier_out_{Language}"]=True
                        else:
                            sent_dict[f"Verifier_out_{Language}"]=False


                        # Update id2results
                        if sent_dict["number_id"] in id2results:
                            id2results[sent_dict["number_id"]].update(sent_dict)
                        else:
                            id2results[sent_dict["number_id"]] = sent_dict

                    # Write results after each batch is processed
                    write_dic = {}
                    for key, value in id2results.items():
                        if value["key"] in write_dic:
                            write_dic[value["key"]].append(value)
                        else:
                            write_dic[value["key"]] = [value]

                    json.dump(write_dic, open(f"../data/Language/{mode}_translated_{Language}.json", "w"), indent=2, ensure_ascii=False)