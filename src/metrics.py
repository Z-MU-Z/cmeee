
import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        #print(predictions[1])
        #print(labels[0])
        #'''NOTE: You need to finish the code of computing f1-score.

        #'''
        pred_list = extract_entities(predictions)
        label_list = extract_entities(labels)
        #print(pred_list[1])
        total_true_and_pred = 0
        total_true = 0
        total_pred = 0
        for i in range(len(pred_list)):
            pred = set(pred_list[i])
            total_pred += len(pred)
            true = set(label_list[i])
            total_true += len(true)
            true_and_pred = set.intersection(true,pred)
            total_true_and_pred += len(true_and_pred)
        #print(total_pred)
        #print(total_true)
        #print(total_true_and_pred)
        f1 = 2*total_true_and_pred/(total_pred+total_true)
        #print(f1)
        return { "f1": f1 }


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        # '''NOTE: You need to finish the code of computing f1-score.

        # '''

        # print(predictions[1,:,1])
        # print(labels1[1])
        # print(labels2[1])
        pred_list1 = extract_entities(predictions[:,:,0],True, True)
        pred_list2 = extract_entities(predictions[:, :, 1],True, False)
        label_list1 = extract_entities(labels1, True, True)
        label_list2 = extract_entities(labels2, True, False)
        total_true_and_pred = 0
        total_true = 0
        total_pred = 0
        #print(label_list2)
        #print(pred_list2)
        for i in range(len(pred_list1)):
            # print(pred_list1[i])
            # print(pred_list2[i])
            pred = set(pred_list1[i])|set(pred_list2[i])
            #print(pred)
            total_pred += len(pred)
            true = set(label_list1[i])|set(label_list2[i])
            total_true += len(true)
            true_and_pred = set.intersection(true,pred)
            total_true_and_pred += len(true_and_pred)
        f1 = 2 * total_true_and_pred / (total_pred + total_true)
        # print(total_pred)
        # print(total_true)
        # print(total_true_and_pred)
        # print(f1)
        return { "f1": f1 }


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2
    #print(id2label)
    entity_set = set(_LABEL_RANK.keys())
    def dicide_type(cache):
        c = Counter(cache)
        rank_c = c.most_common(len(list(c.keys())))
        if len(rank_c) == 1:
            type = rank_c[0][0]
        else:
            i = 0
            candidate = []
            max_fre = rank_c[0][1]
            for i in range(len(rank_c)):
                if rank_c[i][1] == max_fre:
                    candidate.append(rank_c[i][0])
                else:
                    break
            #print(candidate)
            max_rank = -1
            for type in candidate:
                if _LABEL_RANK[type] > max_rank:
                    temp_type = type
                    max_rank = _LABEL_RANK[type]
            type = temp_type
        return type
    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    bs , max_len = batch_labels_or_preds.shape
    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    for i in  range(bs):
        entity_list = []
        index = 0
        start_idx = 0
        start_id = 0
        #end_id = 0
        #end_idx = 0
        cache = []
        while id2label[batch_labels_or_preds[i][index]] != '[PAD]' and index<max_len:
            if start_id:
                # if batch_labels_or_preds[i][index] == start_id:
                #     entity_list.append((start_index,start_index,id2label[batch_labels_or_preds[i][index]][2:]))
                #     start_index += 1
                if id2label[batch_labels_or_preds[i][index]][0] == 'I':
                    cache.append(id2label[batch_labels_or_preds[i][index]][2:])
                    index += 1
                    #print(index)
                    if index == max_len:
                        type = dicide_type(cache)
                        # print(c)
                        entity_list.append((start_idx, index-1, type))

                    continue
                else:
                    end_idx = index-1
                    #print(id2label[batch_labels_or_preds[i][index]])
                    #print(cache)
                    type = dicide_type(cache)
                    #print(c)
                    entity_list.append((start_idx, end_idx, type))
                    start_id = 0

            if id2label[batch_labels_or_preds[i][index]][0] == 'B':
                cache = []
                start_id = batch_labels_or_preds[i][index]
                start_idx = index
                cache.append(id2label[batch_labels_or_preds[i][index]][2:])

            index += 1
            if index == max_len:
                if start_id:
                    print(index)
                    #cache.append(id2label[batch_labels_or_preds[i][index]][2:])
                    entity_list.append((start_idx, index-1, id2label[batch_labels_or_preds[i][start_idx]][2:]))
                break
        batch_entities.append(entity_list)
        #print(c)
        #print(c.most_common(len(list(c.keys()))))

                

    # '''
    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
    