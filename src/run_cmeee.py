
import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List

from sklearn.metrics import precision_recall_fscore_support
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer,AdapterTrainer
import transformers
from args import ModelConstructArgs, CBLUEDataArgs
from logger import get_logger
from ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id
from model import BertForCRFHeadNER, BertForLinearHeadNER,  BertForLinearHeadNestedNER, CRFClassifier, LinearClassifier,\
    BertForCRFHeadNestedNER
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities
from torch.nn import LSTM
import sys
from model import *
MODEL_CLASS = {
    'linear': BertForLinearHeadNER, 
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested':BertForCRFHeadNestedNER,
    'adapter_linear':BertAdapterForLinearHeadNER,
    'layer_linear':BertlayerForLinearHeadNER
}
n_tokens = 20
initialize_from_vocab = True
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
        if "layer" in model_args.head_type:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS,layer=5)
    else:
        #model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)
        #model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1,num_labels2=EE_NUM_LABELS2,cache_dir = '/dssg/home/acct-stu/stu915/.cache/huggingface/transformers',local_files_only = True)
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1,
                                            num_labels2=EE_NUM_LABELS2,
                                            cache_dir='/dssg/home/acct-stu/stu915/.cache/huggingface/transformers',
                                            local_files_only=True
                                            )
    tokenizer = BertTokenizer.from_pretrained(model_args.model_path,cache_dir = '/dssg/home/acct-stu/stu915/.cache/huggingface/transformers',
                                              local_files_only = True
                                              )
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    logger.info(f"===============> Called command line: \n {' '.join(sys.argv)}")
    logger.info(f"You can copy paste it to debug")

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    for_nested_ner = 'nested' in model_args.head_type
    if model_args.prompt_tuning:
        print("use_prompt_tuning")
        s_wte = SoftEmbedding(model.get_input_embeddings(),
                              n_tokens=n_tokens,
                              initialize_from_vocab=initialize_from_vocab)
        model.set_input_embeddings(s_wte)
        for param in model.bert.encoder.parameters():
            param.requires_grad = False
        #print(s_wte.learned_embedding)


    # ===== Get datasets =====
    if train_args.do_train:
        train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()

    if 'adapter' in model_args.head_type:
        print("use_adapter")
        model.bert.add_adapter("ner1")
        model.bert.train_adapter("ner1")
        trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    else:
        if model_args.layer_decay:
            print("use_layer_decay")
            def bert_base_AdamW_grouped_LLRD(model, init_lr=train_args.learning_rate):

                opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
                named_parameters = list(model.named_parameters())
                modelname = 'bert.'
                # According to AAAMLP book by A. Thakur, we generally do not use any decay
                # for bias and LayerNorm.weight layers.
                no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
                set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]

                for i, (name, params) in enumerate(named_parameters):

                    weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

                    if name.startswith(modelname+"embeddings") or name.startswith(modelname+"encoder"):
                        # For first set, set lr to 1e-6 (i.e. 0.000001)
                        lr = init_lr

                        # For set_2, increase lr to 0.00000175
                        lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

                        # For set_3, increase lr to 0.0000035
                        lr = init_lr * 3.5 if any(p in name for p in set_3) else lr

                        opt_parameters.append({"params": params,
                                               "weight_decay": weight_decay,
                                               "lr": lr})
                        continue

                    # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).
                    if name.startswith(modelname+"regressor") or name.startswith(modelname+"pooler"):
                        lr = init_lr * 3.6

                        opt_parameters.append({"params": params,
                                               "weight_decay": weight_decay,
                                               "lr": lr})
                        continue
                    else:
                        lr = init_lr * 10
                        print(name)
                        opt_parameters.append({"params": params,
                                               "weight_decay": weight_decay,
                                               "lr": lr})

                return transformers.AdamW(opt_parameters, lr=init_lr)

            opt = bert_base_AdamW_grouped_LLRD(model)
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=50,
                num_training_steps=2000000,
            )
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=train_args,
                data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                compute_metrics=compute_metrics,
                optimizers= (opt,scheduler)
            )
        else:
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=train_args,
                data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                compute_metrics=compute_metrics,
            )

        print(trainer)



    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
    #print(s_wte.learned_embedding)
    if train_args.do_predict:
        test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()
