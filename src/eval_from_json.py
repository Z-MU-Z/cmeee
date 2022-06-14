# coding=utf-8

from ee_data import EE_label2id, EE_id2label, EE_id2label1, EE_id2label2, EE_label2id1, EE_label2id2, LABEL1, LABEL2, \
    LABEL
from metrics import ComputeMetricsForNestedNER, ComputeMetricsForNER, EvalPrediction
import json
import numpy as np
import random
from tqdm import tqdm
import time
import datetime
import sys
import pandas as pd


def f1_from_jsons(gt_label_json, pred_label_json, nested=True):
    gt_label_dict = json.load(open(gt_label_json))
    pred_label_dict = json.load(open(pred_label_json))

    max_len = 0
    assert len(gt_label_dict) == len(pred_label_dict)
    for idx, (label_term, pred_term) in enumerate(zip(gt_label_dict, pred_label_dict)):
        assert label_term['text'] == pred_term['text']
        l = len(label_term['text'])
        if l > max_len:
            print(label_term['text'])
        max_len = max(l, max_len)
    print(max_len)

    if nested:
        label_array = np.ones(shape=(len(gt_label_dict), max_len, 2)).astype(int)
        pred_array = np.ones(shape=(len(gt_label_dict), max_len, 2)).astype(int)
    else:
        label_array = np.ones(shape=(len(gt_label_dict), max_len)).astype(int)
        pred_array = np.ones(shape=(len(gt_label_dict), max_len)).astype(int)

    for idx, (label_term, pred_term) in enumerate(zip(gt_label_dict, pred_label_dict)):
        len_cur_stn = len(label_term['text'])
        if nested:
            label_array[idx, len_cur_stn:, :] = -100
            pred_array[idx, len_cur_stn:, :] = -100
            for ent in label_term['entities']:
                start_idx, end_idx = ent['start_idx'], ent['end_idx']
                if ent['type'] in LABEL2:
                    label_array[idx, start_idx, 1] = EE_label2id2[f"B-{ent['type']}"]
                    label_array[idx, start_idx + 1:end_idx + 1, 1] = EE_label2id2[f'I-{ent["type"]}']
                else:
                    label_array[idx, start_idx, 0] = EE_label2id1[f"B-{ent['type']}"]
                    label_array[idx, start_idx + 1:end_idx + 1, 0] = EE_label2id1[f'I-{ent["type"]}']
            for ent in pred_term['entities']:
                start_idx, end_idx = ent['start_idx'], ent['end_idx']
                if ent['type'] in LABEL2:
                    pred_array[idx, start_idx, 1] = EE_label2id2[f"B-{ent['type']}"]
                    pred_array[idx, start_idx + 1:end_idx + 1, 1] = EE_label2id2[f'I-{ent["type"]}']
                else:
                    pred_array[idx, start_idx, 0] = EE_label2id1[f"B-{ent['type']}"]
                    pred_array[idx, start_idx + 1:end_idx + 1, 0] = EE_label2id1[f'I-{ent["type"]}']
        else:
            label_array[idx, len_cur_stn:] = -100
            pred_array[idx, len_cur_stn:] = -100
            for ent in label_term['entities']:
                start_idx, end_idx = ent['start_idx'], ent['end_idx']
                label_array[idx, start_idx] = EE_label2id[f"B-{ent['type']}"]
                label_array[idx, start_idx + 1:end_idx + 1] = EE_label2id[f'I-{ent["type"]}']
            for ent in pred_term['entities']:
                start_idx, end_idx = ent['start_idx'], ent['end_idx']
                pred_array[idx, start_idx] = EE_label2id[f"B-{ent['type']}"]
                pred_array[idx, start_idx + 1:end_idx + 1] = EE_label2id[f'I-{ent["type"]}']

    if nested:
        metrics = ComputeMetricsForNestedNER()(EvalPrediction(pred_array, (label_array[:, :, 0], label_array[:, :, 1])))
    else:
        metrics = ComputeMetricsForNER()(EvalPrediction(pred_array, label_array))
    return metrics


def get_type_wise_metric(gt_json, pred_json, nested=True):
    # return : dataframe
    gt_json = json.load(open(gt_json))
    pred_json = json.load(open(pred_json))
    gt_label_sets = []
    for item in gt_json:
        for ent in item['entities']:
            gt_label_sets.append(ent)
    pred_sets = []
    for item in pred_json:
        for ent in item['entities']:
            if "entity" not in ent.keys():
                ent['entity'] = item['text'][ent['start_idx']:ent['end_idx'] + 1]
            pred_sets.append(ent)

    eps = 1e-7

    # ======= get general metrics =======
    # correct = len(list(filter(lambda x:x in gt_label_sets, pred_sets)))
    # r = correct / (len(gt_label_sets)+eps)
    # p = correct / (len(pred_sets) + eps)
    # f1 = 2 * r * p / (r + p)
    # print(f"Overall metrics: recall {r} \t precision {p} \t f1 {f1}")
    res = []
    for type in tqdm(LABEL):
        this_type_gt = list(filter(lambda x: x['type'] == type, gt_label_sets))
        this_type_pred = list(filter(lambda x: x['type'] == type, pred_sets))
        this_type_correct = sum(map(lambda x: x in this_type_gt, this_type_pred))

        recall = this_type_correct / (len(this_type_gt) + eps)
        precision = this_type_correct / (len(this_type_pred) + eps)
        f1 = 2 * recall * precision / (recall + precision + eps)
        res.append({
            "type": type,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "gt_count": len(this_type_gt),
            "pred_count": len(this_type_pred)
        })
    return pd.DataFrame(res)


def create_html(gt_label_json, pred_label_json, file_path='../result.html', sample_num=1000, nested=True):
    gt_label_dict = json.load(open(gt_label_json))
    pred_label_dict = json.load(open(pred_label_json))
    itr = random.sample(list(enumerate(zip(gt_label_dict, pred_label_dict))), sample_num)

    with open(file_path, 'w') as f:
        if not nested:
            f.write("<!DOCTYPE html>\n"
                    "<html lang=\"en\">\n"
                    "<head>\n"
                    "<meta charset=\"UTF-8\">\n"
                    "<title>CMeEE Prediction</title>\n"
                    "</head>\n"
                    "<body>\n"
                    "<h1>CMeEE prediction</h1>\n"
                    f"Created on {datetime.datetime.now()}\n<br>\nlabel file: <code>{gt_label_json}</code>\n<br>\nprediction file: <code>{pred_label_json}</code>\n<br>\nNested: <code>False</code>\n"
                    "<h3>Format: Text, Label, Prediction</h3>\n"
                    "<pre>\n")
            for idx, (label_term, pred_term) in itr:
                text_str = "&#9;".join(list(label_term['text']))
                f.write(text_str + "\n")
                label_list = ["O" for _ in range(len(label_term['text']))]
                pred_list = ["O" for _ in range(len(label_term['text']))]
                for i, ent in enumerate(label_term['entities']):
                    label_list[ent['start_idx']] = f"B-{ent['type']}"
                    label_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (
                            ent['end_idx'] - ent['start_idx'])
                for i, ent in enumerate(pred_term['entities']):
                    pred_list[ent['start_idx']] = f"B-{ent['type']}"
                    pred_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (
                            ent['end_idx'] - ent['start_idx'])
                for ent in pred_term['entities']:
                    if "entity" not in ent.keys():
                        ent['entity'] = pred_term['text'][ent['start_idx']:ent['end_idx'] + 1]
                    if ent in label_term['entities']:
                        label_list[ent['start_idx']] = "<span style=\"color: green\">" + label_list[ent['start_idx']]
                        label_list[ent['end_idx']] = label_list[ent['end_idx']] + "</span>"
                        pred_list[ent['start_idx']] = "<span style=\"color: green\">" + pred_list[ent['start_idx']]
                        pred_list[ent['end_idx']] = pred_list[ent['end_idx']] + "</span>"
                    else:
                        pred_list[ent['start_idx']] = "<span style=\"color: red\">" + pred_list[ent['start_idx']]
                        pred_list[ent['end_idx']] = pred_list[ent['end_idx']] + "</span>"
                for ent in label_term['entities']:
                    if ent not in pred_term['entities']:
                        label_list[ent['start_idx']] = "<span style=\"color: red\">" + label_list[ent['start_idx']]
                        label_list[ent['end_idx']] = label_list[ent['end_idx']] + "</span>"
                label_line = "&#9;".join(label_list) + '\n'
                label_line = "<span style=\"background-color: #F0F0F0\">" + label_line + "</span>"
                pred_line = "&#9;".join(pred_list) + '\n'
                f.write(label_line)
                f.write(pred_line)
                f.write("<HR>\n")
        else:
            f.write("<!DOCTYPE html>\n"
                    "<html lang=\"en\">\n"
                    "<head>\n"
                    "<meta charset=\"UTF-8\">\n"
                    "<title>CMeEE Prediction</title>\n"
                    "</head>\n"
                    "<body>\n"
                    "<h1>CMeEE prediction</h1>\n"
                    f"Created on {datetime.datetime.now()}\n<br>\nlabel file: <code>{gt_label_json}</code>\n<br>\nprediction file: <code>{pred_label_json}</code>\n<br>\nNested: <code>True</code>\n"
                    "<h3>Format: Text, Label1, Prediction1, Label2, Prediction2</h3>\n"
                    "<pre>\n")
            for idx, (label_term, pred_term) in itr:
                text_str = "&#9;".join(list(label_term['text']))
                f.write(text_str + "\n")
                label1_list = ["O" for _ in range(len(label_term['text']))]
                pred1_list = ["O" for _ in range(len(label_term['text']))]
                label2_list = ["O" for _ in range(len(label_term['text']))]
                pred2_list = ["O" for _ in range(len(label_term['text']))]

                for i, ent in enumerate(label_term['entities']):
                    if not ent['type'] in LABEL2:
                        label1_list[ent['start_idx']] = f"B-{ent['type']}"
                        label1_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (
                                ent['end_idx'] - ent['start_idx'])
                    else:
                        label2_list[ent['start_idx']] = f"B-{ent['type']}"
                        label2_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (
                                ent['end_idx'] - ent['start_idx'])

                for i, ent in enumerate(pred_term['entities']):
                    if not ent['type'] in LABEL2:
                        pred1_list[ent['start_idx']] = f"B-{ent['type']}"
                        pred1_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (
                                ent['end_idx'] - ent['start_idx'])
                    else:
                        pred2_list[ent['start_idx']] = f"B-{ent['type']}"
                        pred2_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (
                                ent['end_idx'] - ent['start_idx'])

                for ent in pred_term['entities']:
                    if not ent['type'] in LABEL2:  # pred 中不是LABEL2的那些实体
                        if ent in label_term['entities']:
                            label1_list[ent['start_idx']] = "<span style=\"color: green\">" + label1_list[
                                ent['start_idx']]
                            label1_list[ent['end_idx']] = label1_list[ent['end_idx']] + "</span>"
                            pred1_list[ent['start_idx']] = "<span style=\"color: green\">" + pred1_list[
                                ent['start_idx']]
                            pred1_list[ent['end_idx']] = pred1_list[ent['end_idx']] + "</span>"
                        else:
                            pred1_list[ent['start_idx']] = "<span style=\"color: red\">" + pred1_list[ent['start_idx']]
                            pred1_list[ent['end_idx']] = pred1_list[ent['end_idx']] + "</span>"
                    else:
                        if ent in label_term['entities']:
                            label2_list[ent['start_idx']] = "<span style=\"color: green\">" + label2_list[
                                ent['start_idx']]
                            label2_list[ent['end_idx']] = label2_list[ent['end_idx']] + "</span>"
                            pred2_list[ent['start_idx']] = "<span style=\"color: green\">" + pred2_list[
                                ent['start_idx']]
                            pred2_list[ent['end_idx']] = pred2_list[ent['end_idx']] + "</span>"
                        else:
                            pred2_list[ent['start_idx']] = "<span style=\"color: red\">" + pred2_list[ent['start_idx']]
                            pred2_list[ent['end_idx']] = pred2_list[ent['end_idx']] + "</span>"

                for ent in label_term['entities']:
                    if not ent['type'] in LABEL2:  # pred 中不是LABEL2的那些实体
                        if ent not in pred_term['entities']:
                            label1_list[ent['start_idx']] = "<span style=\"color: red\">" + label1_list[
                                ent['start_idx']]
                            label1_list[ent['end_idx']] = label1_list[ent['end_idx']] + "</span>"
                    else:
                        if ent not in pred_term['entities']:
                            label2_list[ent['start_idx']] = "<span style=\"color: red\">" + label2_list[
                                ent['start_idx']]
                            label2_list[ent['end_idx']] = label2_list[ent['end_idx']] + "</span>"

                label1_line = "&#9;".join(label1_list) + '\n'
                label1_line = "<span style=\"background-color: #F0F0F0\">" + label1_line + "</span>"
                pred1_line = "&#9;".join(pred1_list) + '\n'
                label2_line = "&#9;".join(label2_list) + '\n'
                label2_line = "<span style=\"background-color: #F0F0F0\">" + label2_line + "</span>"

                pred2_line = "&#9;".join(pred2_list) + '\n'
                f.write(label1_line)
                f.write(pred1_line)
                f.write(label2_line)
                f.write(pred2_line)
                f.write("<HR>\n")

        f.write("</pre>\n"
                "</body>\n"
                "</html>\n")


if __name__ == '__main__':
    gt_json = '../data/CBLUEDatasets/CMeEE/CMeEE_dev.json'
    # pred_json = '../ckpts/baseline_crf_nested/CMeEE_dev_updated_by_bod.json'
    # pred_json = '../ckpts/baseline_crf_nested/CMeEE_dev.json'
    # pred_json = '../ckpts/bert_crf_nested_2022_flat/CMeEE_dev_updated_by_dep.json'
    # pred_json = '../ckpts/bert_crf_nested_2022/CMeEE_dev.json'
    # pred_json = '../ckpts/bert_crf_nested_2022/CMeEE_dev_updated_by_equ.json'
    # pred_json = "../ckpts/bert_crf_nested_2022_aug60/CMeEE_dev_updated_by_bod.json"
    pred_json = '../ckpts/bert_crf_nested_2022_flat/CMeEE_dev_updated_by_dep.json'
    # pred_json = '../ckpts/bert_crf_nested_2022_flat/CMeEE_dev.json'

    # pred_json = '../ckpts/global_pointer/CMeEE_dev.json'
    # pred_json = '../ckpts/global_pointer/CMeEE_dev_updated_by_ite.json'
    # pred_json = "../ckpts/roberta_large_crf_nested_2022/CMeEE_dev.json"

    metrics = f1_from_jsons(gt_label_json=gt_json,
                            pred_label_json=pred_json,
                            nested=True)

    f1 = metrics['f1']
    recall = metrics['recall']
    precision = metrics['precision']
    print(f"F1 = {str(f1*100)[:6]} \t Recall = {str(recall*100)[:6]} \t Precision = { str(precision*100)[:6] }")

    create_html(gt_label_json=gt_json,
                pred_label_json=pred_json,
                file_path='result.html',
                sample_num=200,
                nested=True)

    type_wise_metric = get_type_wise_metric(gt_json, pred_json)
    # type_wise_metric.to_csv("../figs/baseline_typewise.csv")
    print(type_wise_metric)
