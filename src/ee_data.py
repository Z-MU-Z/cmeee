import json
import logging
import pickle
import re
from collections import Counter
from itertools import repeat
from os.path import join, exists
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NER_PAD, NO_ENT = '[PAD]', 'O'

LABEL1 = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'dis', 'bod']  # 按照出现频率从低到高排序
LABEL2 = ['sym']

LABEL = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod']

# 标签出现频率映射，从低到高
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL for P in ("B", "I")]

EE_label2id1 = {b: a for a, b in enumerate(EE_id2label1)}
EE_label2id2 = {b: a for a, b in enumerate(EE_id2label2)}
EE_label2id = {b: a for a, b in enumerate(EE_id2label)}

EE_NUM_LABELS1 = len(EE_id2label1)
EE_NUM_LABELS2 = len(EE_id2label2)
EE_NUM_LABELS = len(EE_id2label)


class InputExample:
    def __init__(self, sentence_id: str, text: str, entities: List[dict] = None):
        self.sentence_id = sentence_id
        self.text = text
        self.entities = entities
        # for d in self.entities:
        #     if d['type'] in LABEL2:
        #         pass

    def to_ner_task(self, for_nested_ner: bool = False):
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        if self.entities is None:
            return self.sentence_id, self.text
        else:
            if not for_nested_ner:
                label = [NO_ENT] * len(self.text)
            else:
                label1 = [NO_ENT] * len(self.text)
                label2 = [NO_ENT] * len(self.text)

            def _write_label(_label: list, _type: str, _start: int, _end: int):
                for i in range(_start, _end + 1):
                    if i == _start:
                        _label[i] = f"B-{_type}"
                    else:
                        _label[i] = f"I-{_type}"

            for entity in self.entities:
                start_idx = entity["start_idx"]
                end_idx = entity["end_idx"]
                entity_type = entity["type"]

                assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"

                if not for_nested_ner:
                    _write_label(label, entity_type, start_idx, end_idx)
                else:
                    '''NOTE
                    '''
                    if entity_type in LABEL2:
                        _write_label(label2, entity_type, start_idx, end_idx)
                    else:
                        _write_label(label1, entity_type, start_idx, end_idx)

            if not for_nested_ner:
                return self.sentence_id, self.text, label
            else:
                return self.sentence_id, self.text, (label1, label2)
    def back_translation(self,min_lenth=0): #进行backtranslate的最短长度
        mask = np.zeros(len(self.text))
        for entity in self.entities:
            start_idx = entity["start_idx"]
            end_idx = entity["end_idx"]
            entity_type = entity["type"]

            assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"
            mask[start_idx:end_idx+1] = 1

        #print(mask)
        not_entities = []
        start = -1
        end = -1
        flag = 0
        for i in range(len(self.text)):
            #print(i)
            if mask[i] == 0:
                if flag == 0: #遇到了新的非实体
                    start = i
                    flag = 1
                if flag == 1:
                    continue
            if mask[i]:
                if flag == 0: #still in entity
                    continue
                if flag == 1:
                    flag = 0
                    not_entities.append((start,i-1))
                    #print(start, i - 1)
        if flag == 1:
            not_entities.append((start,len(self.text)-1))
        #print(self.text)
        #print(self.entities)
        #print(not_entities)#相当于记录了所有非实体区间
        tem_text = ''
        last_end = 0
        bias = 0
        entity_index = 0
        for i in range(len(not_entities)):
            start, end = not_entities[i]
            if last_end < start:#为了应对一开始0的情况
                while(self.entities[entity_index]["end_idx"]<start):#属于这块区域内的实体
                    self.entities[entity_index]["end_idx"] += bias
                    self.entities[entity_index]["start_idx"] += bias #进行相对位置的偏移
                    entity_index += 1
                    #print(entity_index)
                    if entity_index == len(self.entities):
                        break

                tem_text += self.text[last_end:start]  # 保留实体部分
            last_end = end + 1
            if end - start + 1 < min_lenth: #太短就不翻译了
                tem_text += self.text[start:end+1]

            else:
                #result = translate(self.text[start:end+1])
                result = '只是为了测试'
                bias += len(result) - (end + 1 - start)
                tem_text += result
        while entity_index < len(self.entities):
            self.entities[entity_index]["end_idx"] += bias
            self.entities[entity_index]["start_idx"] += bias  # 进行相对位置的偏移
            entity_index += 1
            tem_text += self.text[last_end:len(self.text)]
            #print(tem_text)
        print(self.text)
        print(self.entities)
        print(tem_text)
        #print(bias,len(self.text),len(tem_text))
        assert len(self.text)+bias == len(tem_text),"长度不一致，字词缺失？"
        self.text = tem_text
















class EEDataloader:
    def __init__(self, cblue_root: str):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")

    @staticmethod
    def _load_json(filename: str) -> List[dict]:
        with open(filename, encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def _parse(cmeee_data: List[dict]) -> List[InputExample]:
        return [InputExample(sentence_id=str(i), **data) for i, data in enumerate(cmeee_data)]

    def get_data(self, mode: str):
        if mode not in ("train", "dev", "test"):
            raise ValueError(f"Unrecognized mode: {mode}")
        return self._parse(self._load_json(join(self.data_root, f"CMeEE_{mode}.json")))


class EEDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, tokenizer, for_nested_ner: bool):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"

        # Processed data can vary from using different tokenizers
        _tk_class = re.match("<class '.*\.(.*)'>", str(type(tokenizer))).group(1)
        _head = 2 if for_nested_ner else 1
        cache_file = join(self.data_root, f"cache_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        if False and exists(cache_file):
            with open(cache_file, "rb") as f:
                self.examples, self.data = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode)  # get original data
            # print(self.examples[0].text)
            # print(self.examples[0].entities)
            for i in range(len(self.examples)):
                self.examples[i].back_translation(5)
            self.data = self._preprocess(self.examples, tokenizer)  # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        is_test = examples[0].entities is None
        data = []

        label2id = EE_label2id

        for example in examples:
            if is_test:
                _sentence_id, text = example.to_ner_task(self.for_nested_ner)
                if self.for_nested_ner:
                    label = [repeat(None, len(text))] * 2
                else:
                    label = repeat(None, len(text))
            else:
                _sentence_id, text, label = example.to_ner_task(self.for_nested_ner)
                # if nested, then label: (list, list)

            tokens = []
            label_ids = None if is_test else []

            if not self.for_nested_ner:
                for word, L in zip(text, label):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    if not is_test:
                        label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

                tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                if not is_test:
                    label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]

                    data.append((token_ids, label_ids))
                else:
                    data.append((token_ids,))
            else:
                label2id = EE_label2id1  # NOTE: switch to this branch
                for word, L in zip(text, label[0]):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)
                    if not is_test:
                        label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

                tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                label2id = EE_label2id2
                label_ids2 = None if is_test else []
                for word, L in zip(text, label[1]):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)
                    if not is_test:
                        label_ids2.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

                if not is_test:
                    label_ids = [EE_label2id1[NO_ENT]] + label_ids[: self.max_length - 2] + [EE_label2id1[NO_ENT]]
                    label_ids2 = [EE_label2id2[NO_ENT]] + label_ids2[: self.max_length - 2] + [EE_label2id2[NO_ENT]]

                    data.append((token_ids, label_ids, label_ids2))
                else:
                    data.append((token_ids,))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class CollateFnForEE:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner

    def __call__(self, batch) -> dict:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        # if nested: batch [(token_ids, label_ids, label_ids2), ...]
        inputs = [x[0] for x in batch]
        no_decode_flag = batch[0][1]

        input_ids = [x[0] for x in inputs]
        labels = [x[1] for x in inputs] if len(inputs[0]) > 1 else None
        if self.for_nested_ner:
            labels2 = [x[2] for x in inputs] if len(inputs[0]) > 1 else None

        max_len = max(map(len, input_ids))
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, _ids in enumerate(input_ids):
            attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            input_ids[i] += [self.pad_token_id] * _delta_len

            if labels is not None:
                labels[i] += [self.label_pad_token_id] * _delta_len
            if (self.for_nested_ner) and (labels2 is not None):
                labels2[i] += [self.label_pad_token_id] * _delta_len

        if not self.for_nested_ner:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "no_decode": no_decode_flag
            }
        else:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,  # modify this
                "labels2": torch.tensor(labels2, dtype=torch.long) if labels2 is not None else None,  # modify this
                "no_decode": no_decode_flag
            }

        return inputs


if __name__ == '__main__':
    import os
    from os.path import expanduser
    from transformers import BertTokenizer

    MODEL_NAME = "../bert-base-chinese"
    CBLUE_ROOT = "../data/CBLUEDatasets"


    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # dataset = EEDataset(CBLUE_ROOT, mode="dev", max_length=10, tokenizer=tokenizer, for_nested_ner=False)
    dataset = EEDataset(CBLUE_ROOT, mode="dev", max_length=10, tokenizer=tokenizer, for_nested_ner=True)

    batch = [dataset[0], dataset[1], dataset[2]]
    inputs = CollateFnForEE(pad_token_id=tokenizer.pad_token_id, for_nested_ner=True)(batch)
    print(inputs)
