import os
import json
mode = "awareness"
#["awareness","bias"]
lis = os.listdir("../data/Language")
f_origin = open(f"../data/{mode}_data.json","r",encoding="utf-8")
origin = json.load(f_origin)
id2origin = {}
for key in origin.keys():
    for item in origin[key]:
        id2origin[item["number_id"]] = item
for file in lis:
    if file.endswith(".json") and mode in file:
        print(f"{file=}")
        f_new = open("../Language/"+file,"r",encoding="utf-8")
        new_content = json.load(f_new)
        for key,value in new_content.items():
            for item in value:
                # print(f"{item["number_id"]=}")
                id2origin[item["number_id"]].update(item)
write_dic = {}
for key, value in id2origin.items():
    if value["key"] in write_dic:
        write_dic[value["key"]].append(value)
    else:
        write_dic[value["key"]] = [value]
json.dump(write_dic,open(f"../data/{mode}_translated_large.json","w",encoding="utf-8"),ensure_ascii=False,indent=4)

