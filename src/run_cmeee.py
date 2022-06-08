import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List

from sklearn.metrics import precision_recall_fscore_support
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer, BertConfig

from args import ModelConstructArgs, CBLUEDataArgs
import numpy as np
import os
from logger import get_logger
from ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id, EEWordDataset, EEDatasetWordChar, CollateFnForEEWordChar
from model import BertForCRFHeadNER, BertForLinearHeadNER, BertForLinearHeadNestedNER, CRFClassifier, LinearClassifier, \
    BertForCRFHeadNestedNER, BertForCRFHeadNestedNERWordChar
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities
from torch.nn import LSTM
import sys
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers.configuration_utils import PretrainedConfig
import transformers
from transformers import AlbertTokenizer, BertModel

MODEL_CLASS = {
    'linear': BertForLinearHeadNER,
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested': BertForCRFHeadNestedNER,
}


def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args


def get_model_with_tokenizer(model_args):
    model_class = MODEL_CLASS[model_args.head_type]

    if 'nested' not in model_args.head_type:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    else:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1,
                                            num_labels2=EE_NUM_LABELS2)

    tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False, use_word=False):
    if use_word:
        test_dataset = test_dataset.char_dataset
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for idx, (p1, p2, example) in enumerate(zip(pred_entities1, pred_entities2, test_dataset.old_examples)):
        text = example.text  # original text
        trans_mat = test_dataset.trans_mat_list[idx]

        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            true_start_idx = int(np.where(trans_mat[:, start_idx])[0][0])
            true_end_idx = int(np.where(trans_mat[:, end_idx])[0][0])
            entities.append({
                "start_idx": true_start_idx,
                "end_idx": true_end_idx,
                "type": entity_type,
                "entity": text[true_start_idx: true_end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def check_word_level_integrity(dataset, word_dataset):
    assert_predicate = [(dataset.data[x][0].__len__() == sum(word_dataset.data[x][1])) for x in
                        range(len(dataset))]
    if False in assert_predicate:
        idx = assert_predicate.index(False)
        print(idx)
        f = open('tmp.csv', 'w')
        print(",".join(['CLS'] + list(dataset.examples[idx].text) + ["END"]), file=f)
        print(",".join(list(map(str, dataset.data[idx][0]))), file=f)
        tmp = []
        for i in word_dataset.data[idx][1]:
            tmp.append(str(i))
            for j in range(i - 1):
                tmp.append('-')

        print(",".join(tmp), file=f)
        print()
        print(word_dataset.data[idx][0])
        f.close()

        print("ERROR: integrity constraint does not meet. Check tmp.csv for detail")
        raise AssertionError


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    logger.info(f"===============> Called command line: \n {' '.join(sys.argv)}")
    logger.info(f"You can copy paste it to debug")

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    if model_args.use_word:
        # NOTE probably use the line above is better. Here we hacked model and word_tokenizer
        word_tokenizer = AlbertTokenizer.from_pretrained(model_args.word_model_path)
        # char_bert_config = BertConfig.from_pretrained(model_args.model_path)
        word_bert_config = BertConfig.from_pretrained(model_args.word_model_path)
        model = BertForCRFHeadNestedNERWordChar.from_pretrained(model_args.model_path,
                                                                config_word=word_bert_config,
                                                                num_labels1=EE_NUM_LABELS1,
                                                                num_labels2=EE_NUM_LABELS2)
    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get datasets =====
    if train_args.do_train:
        train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer,
                                  for_nested_ner=for_nested_ner)
        dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer,
                                for_nested_ner=for_nested_ner)
        test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer,
                                 for_nested_ner=for_nested_ner)
        if model_args.use_word:
            train_word_dataset = EEWordDataset(data_args.cblue_root, "train", data_args.max_length, word_tokenizer,
                                               for_nested_ner=for_nested_ner)
            dev_word_dataset = EEWordDataset(data_args.cblue_root, "dev", data_args.max_length, word_tokenizer,
                                             for_nested_ner=for_nested_ner)
            test_word_dataset = EEWordDataset(data_args.cblue_root, "test", data_args.max_length, word_tokenizer,
                                              for_nested_ner=for_nested_ner)

            check_word_level_integrity(train_dataset, train_word_dataset)
            check_word_level_integrity(dev_dataset, dev_word_dataset)
            check_word_level_integrity(test_dataset, test_word_dataset)
            logger.info("Congratulations for passing integrity tests on all datasets")

            train_dataset = EEDatasetWordChar(train_dataset, train_word_dataset)
            dev_dataset = EEDatasetWordChar(dev_dataset, dev_word_dataset)

        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    if model_args.use_word:
        data_collator = CollateFnForEEWordChar(
            char_pad_token_id=tokenizer.pad_token_id,
            word_pad_token_id=word_tokenizer.pad_token_id,
            for_nested_ner=for_nested_ner
        )
    else:
        data_collator = CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
    # ===== Trainer =====
    compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    if train_args.do_train:
        try:
            trainer.train(resume_from_checkpoint=True)  # resume_from_checkpoint=True
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        # ================================= LOAD model like trainer.train() =======================================
        resume_from_checkpoint = get_last_checkpoint(trainer.args.output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({trainer.args.output_dir})")
        WEIGHTS_NAME = "pytorch_model.bin"
        CONFIG_NAME = "config.json"

        if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
        logger.info(f"Loading model from {resume_from_checkpoint}).")
        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != transformers.__version__:
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {transformers.__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )
        if trainer.args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            trainer._load_state_dict_in_model(state_dict)

            # release memory
            del state_dict
        # =====================================================================================================

        set_to_do_predict = "test"
        test_dataset = EEDataset(data_args.cblue_root, set_to_do_predict, data_args.max_length, tokenizer,
                                 for_nested_ner=for_nested_ner)
        if model_args.use_word:
            test_word_dataset = EEWordDataset(data_args.cblue_root, set_to_do_predict, data_args.max_length, word_tokenizer,
                                              for_nested_ner=for_nested_ner)
            test_dataset = EEDatasetWordChar(test_dataset, test_word_dataset)
        logger.info(f"Testset: {len(test_dataset)} samples")
        print(test_dataset, model_args.use_word)
        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner, use_word=model_args.use_word)


if __name__ == '__main__':
    main()
