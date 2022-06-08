from audioop import bias
from typing import Optional
from unicodedata import bidirectional

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.file_utils import ModelOutput

from ee_data import EE_label2id1, NER_PAD

NER_PAD_ID = EE_label2id1[NER_PAD]


@dataclass
class NEROutputs(ModelOutput):
    """
    NOTE: `logits` here is the CRF decoding result.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.LongTensor] = None


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.loss_fct = CrossEntropyLoss()

    def _pred_labels(self, _logits):
        return torch.argmax(_logits, dim=-1)

    def forward(self, hidden_states, labels=None, no_decode=False):
        _logits = self.layers(hidden_states)
        loss, pred_labels = None, None

        if labels is None:
            pred_labels = self._pred_labels(_logits)
        else:
            loss = self.loss_fct(_logits.view(-1, self.num_labels), labels.view(-1))
            if not no_decode:
                pred_labels = self._pred_labels(_logits)

        return NEROutputs(loss, pred_labels)


class CRFClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

        '''NOTE: This is where to modify for CRF.
        '''
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def _pred_labels(self, x, mask=None):
        '''NOTE: This is where to modify for CRF.
        
        You need to finish the code to predict labels.

        You can add input arguments.
        
        '''
        # print(f"In _pred_labels: x.shape {x.shape}, mask shape {mask.shape}")
        decode_results = self.crf.decode(x, mask=mask)  # now a list of list
        max_len = max([len(tmp) for tmp in decode_results])
        tensor_result = torch.zeros(size=(len(decode_results), max_len))
        for i, batch_item in enumerate(decode_results):
            tensor_result[i][:len(batch_item)] = torch.tensor(batch_item)
        return tensor_result.long()
        # return pred_labels

    def forward(self, hidden_states, attention_mask, labels=None, no_decode=False, label_pad_token_id=NER_PAD_ID):
        '''NOTE: This is where to modify for CRF.
        You need to finish the code to compute loss and predict labels.
        '''
        attention_mask = attention_mask.bool()
        x = self.dropout(hidden_states)
        x = self.linear(x)
        loss_crf, pred = None, None
        if labels is None:
            pred = self._pred_labels(x, attention_mask)
        else:
            log_prob = self.crf(x, labels, mask=attention_mask)
            loss_crf = -log_prob
            if not no_decode:
                pred = self._pred_labels(x, attention_mask)

        return NEROutputs(loss_crf, pred)


def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)


class BertForLinearHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, labels, no_decode=no_decode)
        return output


class BertForLinearHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        '''NOTE: This is where to modify for Nested NER.
        '''
        self.classifier1 = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        '''NOTE: This is where to modify for Nested NER.
        Use the above function _group_ner_outputs for combining results.
        '''
        output = self.classifier1.forward(sequence_output, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, labels2, no_decode=no_decode)
        return _group_ner_outputs(output, output2)


class BertForCRFHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.classifier = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)

        return output


class BertForCRFHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        '''NOTE: This is where to modify for Nested NER.
        '''
        self.classifier1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        '''NOTE: This is where to modify for Nested NER.
        Use the above function _group_ner_outputs for combining results.
        '''
        output = self.classifier1.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(output, output2)


class BertForCRFHeadNestedNERWordChar(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config_char: BertConfig, config_word: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config_char)
        self.char_config = config_char
        self.word_config = config_word

        self.char_bert = BertModel(config_char)
        self.word_bert = BertModel(config_word)
        '''NOTE: This is where to modify for Nested NER.
        '''
        self.classifier1 = CRFClassifier(config_char.hidden_size + config_word.hidden_size, num_labels1,
                                         config_char.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config_char.hidden_size + config_word.hidden_size, num_labels2,
                                         config_char.hidden_dropout_prob)
        self.init_weights()

    def wl2cl(self, wl_embs, word_lens, word_mask, L_char):
        """
        :param wl_embs: B, L_word, 768
        :param word_lens: B, L_word
        :param word_mask: B, L_word, 1 denotes meaningful value
        :param L_char: max chars num
        :return: B, L_char, 768
        """
        # attn_mat should be an L_word x L_char matrix
        attn_mat = torch.zeros(size=(word_lens.shape[0], word_lens.shape[1], L_char)).float()
        attn_mat = attn_mat.to(wl_embs.device)
        for b in range(word_lens.shape[0]):
            start = 0
            for i, L in enumerate(word_lens[b]):
                attn_mat[b, i, start:start + L] = 1
                start += L
        return torch.bmm(wl_embs.permute(0, 2, 1), attn_mat).permute(0, 2, 1)

    def forward(
            self,
            char_input_ids=None,
            word_input_ids=None,
            word_lens=None,
            char_attention_mask=None,
            word_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        char_sequence_output = self.char_bert(  # NOTE: B, char_max_len, 768
            char_input_ids,
            attention_mask=char_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        word_sequence_output = self.word_bert(  # NOTE: B, word_max_len, 768
            word_input_ids,
            attention_mask=word_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        extended_word_sequence_output = self.wl2cl(word_sequence_output, word_lens, word_mask=word_attention_mask,
                                                   L_char=char_sequence_output.shape[1])

        combined_output = torch.cat([
            char_sequence_output,
            extended_word_sequence_output
        ], dim=-1)
        # TODO: duplicate word to char-level

        output = self.classifier1.forward(combined_output, char_attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(combined_output, char_attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(output, output2)


class BertForCRFHeadNestedNERWordCharAdd(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config_char: BertConfig, config_word: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config_char)
        self.char_config = config_char
        self.word_config = config_word

        self.char_bert = BertModel(config_char)
        self.word_bert = BertModel(config_word)
        '''NOTE: This is where to modify for Nested NER.
        '''
        self.word_to_char = nn.Linear(config_word.hidden_size, config_char.hidden_size)
        self.classifier1 = CRFClassifier(config_char.hidden_size, num_labels1,
                                         config_char.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config_char.hidden_size, num_labels2,
                                         config_char.hidden_dropout_prob)
        self.init_weights()

    def wl2cl(self, wl_embs, word_lens, word_mask, L_char):
        """
        :param wl_embs: B, L_word, 768
        :param word_lens: B, L_word
        :param word_mask: B, L_word, 1 denotes meaningful value
        :param L_char: max chars num
        :return: B, L_char, 768
        """
        # attn_mat should be an L_word x L_char matrix
        attn_mat = torch.zeros(size=(word_lens.shape[0], word_lens.shape[1], L_char)).float()
        attn_mat = attn_mat.to(wl_embs.device)
        for b in range(word_lens.shape[0]):
            start = 0
            for i, L in enumerate(word_lens[b]):
                attn_mat[b, i, start:start + L] = 1
                start += L
        return torch.bmm(wl_embs.permute(0, 2, 1), attn_mat).permute(0, 2, 1)

    def forward(
            self,
            char_input_ids=None,
            word_input_ids=None,
            word_lens=None,
            char_attention_mask=None,
            word_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        char_sequence_output = self.char_bert(  # NOTE: B, char_max_len, 768
            char_input_ids,
            attention_mask=char_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        word_sequence_output = self.word_bert(  # NOTE: B, word_max_len, 768
            word_input_ids,
            attention_mask=word_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        extended_word_sequence_output = self.wl2cl(word_sequence_output, word_lens, word_mask=word_attention_mask,
                                                   L_char=char_sequence_output.shape[1])

        combined_output = char_sequence_output + self.word_to_char(extended_word_sequence_output)

        output = self.classifier1.forward(combined_output, char_attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(combined_output, char_attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(output, output2)
