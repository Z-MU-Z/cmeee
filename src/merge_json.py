import json
path1 = '/dssg/home/acct-stu/stu915/cmeee/data/train_100.json'
path2 = '/dssg/home/acct-stu/stu915/cmeee/data/CBLUEDatasets/CMeEE/CMeEE_train.json'
save = '/dssg/home/acct-stu/stu915/cmeee/data/CBLUEDatasets/CMeEE/CMeEE_train_100.json'

def read_json(path):
    with open(path, encoding="utf8") as f:
        json_list = json.load(f)
    return json_list

json_list1 = read_json(path1)

json_list2 = read_json(path2)


result = json_list1 + json_list2
print(len(result))
with open(save, encoding='utf-8', mode='w') as f1:
    json.dump(result, f1, indent=4, ensure_ascii=False)

