# coding=utf-8
import copy
import functools
import json
import logging
import pickle
import re
from collections import Counter
from itertools import repeat
from tqdm import tqdm, trange
from os.path import join, exists
from typing import List

import numpy as np

import torch
from torch.utils.data import Dataset
from BackTranslation import BackTranslation, BackTranslation_Baidu

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

# trans = BackTranslation(url=[
#     'translate.google.com',
#     'translate.google.co.kr',
# ])
trans = BackTranslation_Baidu(appid='20220607001240909', secretKey='OcuqIUjBpSMfw1w0SCOF')


def translate(text):
    result = trans.translate(text, src='zh', tmp='en')
    return result.result_text


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

    def back_translation(self, min_lenth=0):  # 进行backtranslate的最短长度
        mask = np.zeros(len(self.text))
        for entity in self.entities:
            start_idx = entity["start_idx"]
            end_idx = entity["end_idx"]
            entity_type = entity["type"]

            assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"
            mask[start_idx:end_idx + 1] = 1

        # print(mask)
        not_entities = []
        start = -1
        end = -1
        flag = 0
        for i in range(len(self.text)):
            # print(i)
            if mask[i] == 0:
                if flag == 0:  # 遇到了新的非实体
                    start = i
                    flag = 1
                if flag == 1:
                    continue
            if mask[i]:
                if flag == 0:  # still in entity
                    continue
                if flag == 1:
                    flag = 0
                    not_entities.append((start, i - 1))
                    # print(start, i - 1)
        if flag == 1:
            not_entities.append((start, len(self.text) - 1))
        # print(self.text)
        # print(self.entities)
        # print(not_entities)#相当于记录了所有非实体区间
        tem_text = ''
        last_end = 0
        bias = 0
        entity_index = 0
        for i in range(len(not_entities)):
            start, end = not_entities[i]
            if last_end < start:  # 为了应对一开始0的情况
                while (self.entities[entity_index]["end_idx"] < start):  # 属于这块区域内的实体
                    self.entities[entity_index]["end_idx"] += bias
                    self.entities[entity_index]["start_idx"] += bias  # 进行相对位置的偏移
                    entity_index += 1
                    # print(entity_index)
                    if entity_index == len(self.entities):
                        break

                tem_text += self.text[last_end:start]  # 保留实体部分
            last_end = end + 1
            if end - start + 1 < min_lenth:  # 太短就不翻译了
                tem_text += self.text[start:end + 1]

            else:
                result = translate(self.text[start:end + 1])
                # result = '只是为了测试'
                bias += len(result) - (end + 1 - start)
                tem_text += result
        while entity_index < len(self.entities):
            self.entities[entity_index]["end_idx"] += bias
            self.entities[entity_index]["start_idx"] += bias  # 进行相对位置的偏移
            entity_index += 1
            tem_text += self.text[last_end:len(self.text)]
            # print(tem_text)
        print(self.text)
        print(self.entities)
        print(tem_text)
        # print(bias,len(self.text),len(tem_text))
        assert len(self.text) + bias == len(tem_text), "长度不一致，字词缺失？"
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


def norm_example(ex, is_test=False):
    ex.text = ex.text.lower()
    # ======================== Replace special chars ====================
    special_chars = ['℃', u'\ufeff', '\n']
    special_chars_replace = ["°C", "", ""]
    # NOTE: add all those kind of god-damn numbers
    for n in range(8560, 8569):  # from roman 1 to roman 9
        special_chars.append(chr(n))
        special_chars_replace.append(chr(n - 8511))  # from arabic 1 to 9
    special_chars.append(chr(8569))
    special_chars_replace.append("10")
    special_chars.append(chr(8570))
    special_chars_replace.append("11")
    special_chars.append(chr(8571))
    special_chars_replace.append("12")
    for n in range(9321, 9332):  # numbers in circles
        special_chars.append(chr(n))
        special_chars_replace.append(str(n - 9311))
    # ===================================================================

    trans_mat_list = []
    for i in range(len(special_chars)):
        special_char = special_chars[i]
        special_char_replace = special_chars_replace[i]
        while True:
            if special_char not in ex.text:
                break
            else:  # do one norm step
                # if "℃" in ex.text:
                #     print()
                old_len = len(ex.text)
                special_char_idx = ex.text.index(special_char)
                ex.text = ex.text[:special_char_idx] + special_char_replace + \
                          ex.text[special_char_idx + len(special_char):]
                new_len = len(ex.text)

                # ====================== construct transition matrix ======================
                trans_mat = np.zeros(shape=(old_len, new_len))
                # diags for both ends
                trans_mat[:special_char_idx, :special_char_idx] = np.diag([1] * special_char_idx)
                trans_mat[special_char_idx + len(special_char):, special_char_idx + len(special_char_replace):] = \
                    np.diag([1] * (trans_mat.shape[0] - special_char_idx - len(special_char)))
                if len(special_char_replace) != 0:
                    # we assume the replaced str is at least equal to original by length, if len(replaced)=1
                    if len(special_char_replace) == 1:
                        trans_mat[special_char_idx, special_char_idx] = 1
                    else:
                        # the first of replaced str points to the first of original
                        # the rest of replaced str points to the last of original
                        trans_mat[special_char_idx, special_char_idx] = 1
                        trans_mat[special_char_idx + len(special_char) - 1,
                        special_char_idx + 1:special_char_idx + len(special_char_replace)] = 1
                trans_mat_list.append(trans_mat)
                # ==========================================================================

                # ================================ adjust entity idx ==========================
                if not is_test:
                    for i, ent in enumerate(ex.entities):
                        if ent['start_idx'] > special_char_idx:
                            ex.entities[i]['start_idx'] = ent['start_idx'] + len(special_char_replace) - len(
                                special_char)
                            ex.entities[i]['end_idx'] = ent['end_idx'] + len(special_char_replace) - len(
                                special_char)
                        elif ent['end_idx'] > special_char_idx + len(special_char) - 1:
                            ex.entities[i]['end_idx'] = ent['end_idx'] + len(special_char_replace) - len(
                                special_char)
                        elif ent['end_idx'] == special_char_idx + len(special_char) - 1:
                            ex.entities[i]['end_idx'] = ent['end_idx'] + len(special_char_replace) - 1
                        elif ent['end_idx'] < special_char_idx:
                            continue
                # ==============================================================================
    if not is_test:
        for i, ent in enumerate(ex.entities):
            # NOTE: as we have done lower() and special char replacement, we have to re-alloc the entities
            ex.entities[i]['entity'] = ex.text[ex.entities[i]['start_idx']:ex.entities[i]['end_idx'] + 1]
    if len(trans_mat_list) == 0:
        final_trans_mat = np.diag([1] * len(ex.text))
    else:
        final_trans_mat = functools.reduce(lambda x, y: x @ y, trans_mat_list)
    # ===================================================================
    return ex, final_trans_mat


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

        if exists(cache_file):  # NOTE: whether load cache
            with open(cache_file, "rb") as f:
                self.examples, self.data, self.old_examples, self.trans_mat_list = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode)  # get original data

            # ============ back translation usage =============
            # for i in trange(len(self.examples)):
            #     self.examples[i].back_translation(min_lenth=5)
            # =================================================

            self.data = self._preprocess(self.examples, tokenizer)  # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data, self.old_examples, self.trans_mat_list), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        is_test = examples[0].entities is None
        data = []

        label2id = EE_label2id

        # copy old examples
        old_samples = copy.deepcopy(examples)
        self.old_examples = old_samples

        self.trans_mat_list = []

        for example in tqdm(examples):
            # if "℃" in example.text:
            #     print()
            example, trans_mat = norm_example(example,
                                              is_test=is_test)  # NOTE: this changes the objects in examples list
            self.trans_mat_list.append(trans_mat)

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

                tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]  # NOTE: truncate
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                if not is_test:
                    label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]

                    data.append((token_ids, label_ids))
                else:
                    data.append((token_ids,))
            else:
                label2id = EE_label2id1
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
        self.examples = examples
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class EEWordDataset(Dataset):
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
        cache_file = join(self.data_root, f"cache_word_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        if exists(cache_file):  # NOTE: whether load cache
            with open(cache_file, "rb") as f:
                self.examples, self.data = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode)  # get original data
            self.data = self._preprocess(self.examples, tokenizer)  # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        is_test = examples[0].entities is None
        data = []

        for example in tqdm(examples):
            # if "Guidelinesfortreatmentofneonataljaundice，Isthereaplaceforevidence-basedmedicine" in example.text:
            #     print()
            example, trans_mat = norm_example(example, is_test=is_test)
            # self.trans_mat = trans_mat  # old len x new len

            if is_test:
                _sentence_id, text = example.to_ner_task(self.for_nested_ner)
            else:
                _sentence_id, text, label = example.to_ner_task(self.for_nested_ner)
                # if nested, then label: (list, list)

            tokens = tokenizer.tokenize(text[:self.max_length - 2])  # NOTE: tokenize on the truncated chars
            if len(tokens[0]) == 1:
                tokens = tokens[1:]
            else:
                tokens[0] = tokens[0][1:]  # NOTE: remove the strange underline
            word_lens = [len(w) for w in tokens]

            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]  # has been already truncated
            word_lens = [1] + word_lens[: self.max_length - 2] + [1]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            data.append((token_ids, word_lens))

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


class EEDatasetWordChar(Dataset):
    def __init__(self, ee_dataset: EEDataset, ee_word_dataset: EEWordDataset):
        self.char_dataset = ee_dataset
        self.word_dataset = ee_word_dataset
        assert len(self.char_dataset) == len(
            self.word_dataset), f"Length mismatch between char ({len(self.char_dataset)}) and word ({len(self.word_dataset)}) datasets"

    def __getitem__(self, idx):
        char_data, no_decode = self.char_dataset.__getitem__(idx)
        word_data, _ = self.word_dataset.__getitem__(idx)
        res_dict = dict()
        res_dict['no_decode'] = no_decode

        if len(char_data) == 1:
            char_token_ids = char_data[0]
        else:
            if self.char_dataset.for_nested_ner:
                char_token_ids, label_ids1, label_ids2 = char_data
                res_dict['label_ids1'] = label_ids1
                res_dict['label_ids2'] = label_ids2
            else:
                char_token_ids, label_ids = char_data
                res_dict['label_ids'] = label_ids

        res_dict['char_token_ids'] = char_token_ids
        word_token_ids, word_lens = word_data
        res_dict['word_token_ids'] = word_token_ids
        res_dict['word_lens'] = word_lens
        return res_dict

    def __len__(self):
        return len(self.char_dataset)


class CollateFnForEEWordChar:
    def __init__(self, char_pad_token_id: int,
                 word_pad_token_id: int,
                 label_pad_token_id: int = EE_label2id[NER_PAD],
                 for_nested_ner: bool = False):
        self.char_pad_token_id = char_pad_token_id
        self.word_pad_token_id = word_pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner

    def __call__(self, batch) -> dict:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        # if nested: batch [(token_ids, label_ids, label_ids2), ...]
        # batch: list of dicts, every value is list
        char_inputs = [x['char_token_ids'] for x in batch]
        word_inputs = [x['word_token_ids'] for x in batch]
        word_lens = [x['word_lens'] for x in batch]

        # extended_word_inputs = [[] for _ in batch]
        # for i in range(len(char_inputs)):
        #     extended_word_input = []
        #     for word_idx, word_len in zip(word_inputs[i], word_lens[i]):
        #         extended_word_input += [word_idx] * word_len
        #     extended_word_inputs[i] = extended_word_input

        no_decode_flag = batch[0]['no_decode']

        if self.for_nested_ner:
            if 'label_ids1' in batch[0].keys():
                labels = [x['label_ids1'] for x in batch]
                labels2 = [x['label_ids2'] for x in batch]
            else:
                labels = None
                labels2 = None
        else:
            if 'label_ids' in batch[0].keys():
                labels = [x['label_ids'] for x in batch]
            else:
                labels = None

        max_len = max(map(len, char_inputs))
        char_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, _ids in enumerate(char_inputs):
            # pad elements  TODO: add word-levels
            char_attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            char_inputs[i] += [self.char_pad_token_id] * _delta_len
            # extended_word_inputs[i] += [self.pad_token_id] * _delta_len

            if labels is not None:
                labels[i] += [self.label_pad_token_id] * _delta_len
            if (self.for_nested_ner) and (labels2 is not None):
                labels2[i] += [self.label_pad_token_id] * _delta_len

        max_word_len = max(map(len, word_inputs))
        word_attention_mask = torch.zeros((len(batch), max_word_len), dtype=torch.long)
        for i, _ids in enumerate(word_inputs):
            word_attention_mask[i][:len(_ids)] = 1
            _delta_len = max_word_len - len(_ids)
            word_inputs[i] += [self.word_pad_token_id] * _delta_len
            word_lens[i] += [self.word_pad_token_id] * _delta_len

        if not self.for_nested_ner:
            inputs = {
                "char_input_ids": torch.tensor(char_inputs, dtype=torch.long),
                "word_input_ids": torch.tensor(word_inputs, dtype=torch.long),
                "word_lens": torch.tensor(word_lens, dtype=torch.long),
                "char_attention_mask": char_attention_mask,
                "word_attention_mask": word_attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "no_decode": no_decode_flag
            }
        else:
            inputs = {
                "char_input_ids": torch.tensor(char_inputs, dtype=torch.long),
                "word_input_ids": torch.tensor(word_inputs, dtype=torch.long),
                "word_lens": torch.tensor(word_lens, dtype=torch.long),
                "char_attention_mask": char_attention_mask,
                "word_attention_mask": word_attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "labels2": torch.tensor(labels2, dtype=torch.long) if labels2 is not None else None,
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
