# coding=utf-8
import copy
import json
import re
from tqdm import trange, tqdm
import os


def get_all_deps(train_json):
    with open(train_json, encoding='utf-8') as f:
        train_set = json.load(f)

    dep_names = []
    for item in train_set:
        for ent in item['entities']:
            if ent['type'] == 'dep':
                dep_names.append(ent['entity'])
    dep_names = list(set(dep_names))
    dep_itself_count = [0 for _ in dep_names]
    dep_other_count = [0 for _ in dep_names]

    # delete those who exist in other entities more than itself
    for i, dn in tqdm(enumerate(copy.deepcopy(dep_names))):
        # deleted = False
        for item in train_set:
            # if not deleted:
            for ent in item['entities']:
                if (dn in ent['entity']) and (ent['type'] != 'dep'):
                    dep_other_count[i] += 1
                if (ent['type'] == 'dep') and (dn == ent['entity']):
                    dep_itself_count[i] += 1
                # TODO: stat the case that this matched dep instance is not found by any entity
                # for example, 临床 may occur frequently

    print(dep_itself_count)
    print(dep_other_count)
    # exit(1)
    # print(f"Remaining: {dep_names}")
    return dep_names


# with open('data/CBLUEDatasets/CMeEE/CMeEE_dev.json', encoding='utf-8') as f:
#     dev_set = json.load(f)
#
# for i in trange(len(dev_set)):
#     text = dev_set[i]['text']
#     ents = []
#     for name in dep_names:
#         itr = re.finditer(name, text)
#         for f in itr:
#             ents.append({
#                 "start_idx": f.span()[0],
#                 "end_idx": f.span()[1] - 1,
#                 "type": "dep",
#                 "entity": name
#             })
#     dev_set[i]['entities'] = copy.deepcopy(ents)
#
# with open('rule_for_dep_dev.json', 'w', encoding='utf-8') as fw:
#     json.dump(dev_set, fw, ensure_ascii=False, indent=4)


def update_json(to_be_updated, dep_names):
    with open(to_be_updated, encoding='utf-8') as f:
        data = json.load(f)
    for i in trange(len(data)):
        text = data[i]['text']
        existing_ents = data[i]['entities']
        for name in dep_names:
            itr = re.finditer(name, text)
            for f in itr:  # for all the founded matches to this dep-name in this text
                start_idx, end_idx = f.span()
                end_idx = end_idx - 1
                find_cover = False
                for e in existing_ents:  # check if it has already been covered by existing ones
                    if (start_idx > e['end_idx']) or (end_idx < e['start_idx']):
                        # no cover
                        continue
                    else:
                        find_cover = True
                        break
                if not find_cover:
                    existing_ents.append({
                            "start_idx": f.span()[0],
                            "end_idx": f.span()[1] - 1,
                            "type": "dep",
                            "entity": name
                        })
        data[i]['entities'] = existing_ents
    return data


if __name__ == '__main__':
    depnames = get_all_deps(train_json="data/CBLUEDatasets/CMeEE/CMeEE_train.json")
    # depnames.pop(depnames.index("临床"))
    to_be_updated = "ckpts/baseline_crf_nested/CMeEE_dev.json"
    updated = update_json(to_be_updated, depnames)
    dirname = os.path.dirname(to_be_updated)
    basename = os.path.basename(to_be_updated)
    basename_withouth_suffix = '.'.join(basename.split('.')[:-1])
    suffix = basename.split(".")[-1]
    with open(f"{dirname}/{basename_withouth_suffix}_updated_by_rule.{suffix}", 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=4)
