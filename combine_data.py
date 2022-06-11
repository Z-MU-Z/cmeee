import json
import os


path1 = 'data/CBLUEDatasets/CMeEE/CMeEE_train.json'
path2 = 'data/CBLUEDatasets-aug100/train.json'
save = 'data/CBLUEDatasets-aug15/CMeEE/CMeEE_train.json'


def read_json(path):
    with open(path, encoding="utf8") as f:
        json_list = json.load(f)
    return json_list


json_list1 = read_json(path1)

json_list2 = read_json(path2)
import random

sample_num = int(0.15 * len(json_list2))
json_list2 = random.sample(json_list2, sample_num)

result = json_list1 + json_list2
print(len(result))

dirname = os.path.dirname(save)
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(save, encoding='utf-8', mode='w') as f1:
    json.dump(result, f1, indent=4, ensure_ascii=False)
