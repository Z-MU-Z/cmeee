# coding=utf-8
import copy
import json
import re
from tqdm import trange, tqdm
import os


def get_ites_and_filter(train_json):
    with open(train_json, encoding='utf-8') as f:
        train_set = json.load(f)

    ite_names = []
    for item in train_set:
        for ent in item['entities']:
            if ent['type'] == 'ite':
                ite_names.append(ent['entity'])
    ite_names = list(set(ite_names))
    print(f"There are {len(ite_names)} unique ite names")
    ite_itself_count = [0 for _ in ite_names]
    ite_other_count = [0 for _ in ite_names]
    ite_not_in_ent_count = [0 for _ in ite_names]

    # delete those who exist in other entities more than itself
    for i, dn in tqdm(enumerate(copy.deepcopy(ite_names))):
        # deleted = False
        for item in train_set:
            # if not deleted:
            appear_in_ent_num = 0
            for ent in item['entities']:
                if (dn in ent['entity']) and (ent['type'] != 'ite'):
                    ite_other_count[i] += 1
                    appear_in_ent_num += 1
                if (ent['type'] == 'ite') and (dn == ent['entity']):
                    ite_itself_count[i] += 1
                    appear_in_ent_num += 1

                # TODO: stat the case that this matched ite instance is not found by any entity
                # for example, 临床 may occur frequently
            text = item['text']
            itr = re.finditer(dn, text)
            appear_num = len(list(itr))
            appear_not_in_ent_num = appear_num - appear_in_ent_num
            # if appear_not_in_ent_num<0:
            #     print()
            ite_not_in_ent_count[i] += appear_not_in_ent_num

    print(ite_itself_count)
    print(ite_other_count)
    print(ite_not_in_ent_count)

    keep_ite_names = []
    for i in range(len(ite_names)):
        if ite_itself_count[i] <= 2 * (ite_other_count[i] + ite_not_in_ent_count[i]):
            continue
        else:
            keep_ite_names.append(ite_names[i])
    # exit(1)
    print(f"Remaining: {len(keep_ite_names)} out of {len(ite_names)}")
    return keep_ite_names


def update_json(to_be_updated, ite_names):
    with open(to_be_updated, encoding='utf-8') as f:
        data = json.load(f)
    for i in trange(len(data)):
        text = data[i]['text']
        existing_ents = data[i]['entities']
        for name in ite_names:
            itr = re.finditer(name, text)
            for f in itr:  # for all the founded matches to this ite-name in this text
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
                            "type": "ite",
                            "entity": name
                        })
        data[i]['entities'] = existing_ents
    return data


if __name__ == '__main__':
    itenames = get_ites_and_filter(train_json="data/CBLUEDatasets/CMeEE/CMeEE_train.json")
    print(itenames)
    # exit(1)
    # itenames.pop(itenames.index("临床"))

    # ============== update some json ==========================
    to_be_updated = "ckpts/baseline_crf_nested/CMeEE_dev.json"
    # to_be_updated = "ckpts/global_pointer/CMeEE_dev.json"

    updated = update_json(to_be_updated, itenames)
    dirname = os.path.dirname(to_be_updated)
    basename = os.path.basename(to_be_updated)
    basename_withouth_suffix = '.'.join(basename.split('.')[:-1])
    suffix = basename.split(".")[-1]
    with open(f"{dirname}/{basename_withouth_suffix}_updated_by_ite.{suffix}", 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=4)
