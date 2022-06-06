from ee_data import EE_label2id, EE_id2label, EE_id2label1, EE_id2label2, EE_label2id1, EE_label2id2, LABEL1, LABEL2, \
    LABEL
from metrics import ComputeMetricsForNestedNER, ComputeMetricsForNER, EvalPrediction
import json
import numpy as np
import random
import time
import datetime
import sys


def f1_from_jsons(gt_label_json, pred_label_json, nested=True):
    # gt_label_json = '../data/CBLUEDatasets/CMeEE/CMeEE_dev.json'
    # pred_label_json = '../ckpts/bert_crf_nested_2022/CMeEE_test.json'
    # nested = True

    gt_label_dict = json.load(open(gt_label_json))
    pred_label_dict = json.load(open(pred_label_json))

    max_len = 0
    assert len(gt_label_dict) == len(pred_label_dict)
    for idx, (label_term, pred_term) in enumerate(zip(gt_label_dict, pred_label_dict)):
        assert label_term['text'] == pred_term['text']
        l = len(label_term['text'])
        # if l > max_len:
        #     print(label_term['text'])
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
    # print(metrics['f1'])
    return metrics['f1']


def create_html(gt_label_json, pred_label_json, file_path='../result.html', sample_num=1000):
    gt_label_dict = json.load(open(gt_label_json))
    pred_label_dict = json.load(open(pred_label_json))
    with open(file_path, 'w') as f:
        f.write("<!DOCTYPE html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "<meta charset=\"UTF-8\">\n"
                "<title>CMeEE Prediction</title>\n"
                "</head>\n"
                "<body>\n"
                "<h1>CMeEE prediction</h1>\n"
                f"Created on {datetime.datetime.now()}\n<br>\nlabel file: <code>{gt_label_json}</code>\n<br>\nprediction file: <code>{pred_label_json}</code>\n"
                "<h3>Format: Text, Label, Prediction</h3>\n"
                "<pre>\n")
        itr = random.sample(list(enumerate(zip(gt_label_dict, pred_label_dict))), sample_num)
        for idx, (label_term, pred_term) in itr:
            text_str = "&#9;".join(list(label_term['text']))
            f.write(text_str + "\n")
            label_list = ["O" for _ in range(len(label_term['text']))]
            pred_list = ["O" for _ in range(len(label_term['text']))]
            for i, ent in enumerate(label_term['entities']):
                label_list[ent['start_idx']] = f"B-{ent['type']}"
                label_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (ent['end_idx'] -ent['start_idx'])
            for i, ent in enumerate(pred_term['entities']):
                pred_list[ent['start_idx']] = f"B-{ent['type']}"
                pred_list[ent['start_idx'] + 1: ent['end_idx'] + 1] = [f"I-{ent['type']}"] * (ent['end_idx'] -ent['start_idx'])
            for ent in pred_term['entities']:
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
            pred_line = "&#9;".join(pred_list) + '\n'
            f.write(label_line)
            f.write(pred_line)
            # f.write('\n')
            # f.write("="*500 + '\n')
            f.write("<HR>\n")
        f.write("</pre>\n"
                "</body>\n"
                "</html>\n")


if __name__ == '__main__':
    f1 = f1_from_jsons('../data/CBLUEDatasets/CMeEE/CMeEE_dev.json',
                       '../ckpts/bert_crf_nested_2022/CMeEE_test.json',
                       nested=True)
    print(f"F1 = {f1}")

    create_html('../data/CBLUEDatasets/CMeEE/CMeEE_dev.json',
                '../ckpts/bert_crf_nested_2022/CMeEE_test.json',
                file_path='result.html',
                sample_num=100)
