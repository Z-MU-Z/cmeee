# coding=utf-8
import copy
import json
import re
from tqdm import trange, tqdm
import os


def get_deps_and_filter(train_json):
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
    dep_not_in_ent_count = [0 for _ in dep_names]

    # delete those who exist in other entities more than itself
    for i, dn in tqdm(enumerate(copy.deepcopy(dep_names))):
        # deleted = False
        for item in train_set:
            # if not deleted:
            appear_in_ent_num = 0
            for ent in item['entities']:
                if (dn in ent['entity']) and (ent['type'] != 'dep'):
                    dep_other_count[i] += 1
                    appear_in_ent_num += 1
                if (ent['type'] == 'dep') and (dn == ent['entity']):
                    dep_itself_count[i] += 1
                    appear_in_ent_num += 1

                # TODO: stat the case that this matched dep instance is not found by any entity
                # for example, 临床 may occur frequently
            text = item['text']
            itr = re.finditer(dn, text)
            appear_num = len(list(itr))
            appear_not_in_ent_num = appear_num - appear_in_ent_num
            dep_not_in_ent_count[i] += appear_not_in_ent_num

    print(dep_itself_count)
    print(dep_other_count)
    print(dep_not_in_ent_count)

    keep_dep_names = []
    for i in range(len(dep_names)):
        if dep_itself_count[i] <= dep_other_count[i] + dep_not_in_ent_count[i]:
            continue
        else:
            keep_dep_names.append(dep_names[i])
    # exit(1)
    print(f"Remaining: {len(keep_dep_names)} out of {len(dep_names)}")
    return keep_dep_names


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
    depnames = get_deps_and_filter(train_json="data/CBLUEDatasets/CMeEE/CMeEE_train.json")
    print(depnames)
    # exit(1)
    # depnames.pop(depnames.index("临床"))

    # ============== update some json ==========================
    to_be_updated = "ckpts/bert_crf_nested_2022_aug60/CMeEE_test.json"
    # to_be_updated = "ckpts/global_pointer/CMeEE_dev.json"

    updated = update_json(to_be_updated, depnames)
    dirname = os.path.dirname(to_be_updated)
    basename = os.path.basename(to_be_updated)
    basename_withouth_suffix = '.'.join(basename.split('.')[:-1])
    suffix = basename.split(".")[-1]
    with open(f"{dirname}/{basename_withouth_suffix}_updated_by_dep.{suffix}", 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
