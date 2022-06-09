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
from collections import defaultdict

from flat import Transformer_Encoder_Layer, FourPosFusionEmbedding, Transformer_Encoder

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


class BertForCRFHeadNestedNERFlat(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config_char: BertConfig, config_word: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config_char)
        self.char_config = config_char
        self.word_config = config_word
        self.hidden_dim = 128

        self.char_bert = BertModel(config_char)
        self.word_bert = BertModel(config_word)
        '''NOTE: This is where to modify for Nested NER.
        '''
        self.char_to_hidden = nn.Sequential(
            nn.Linear(config_char.hidden_size, self.hidden_dim),
            nn.Dropout(p=0.1),
            # nn.ReLU()
        )
        self.word_to_hidden = nn.Sequential(
            nn.Linear(config_word.hidden_size, self.hidden_dim),
            nn.Dropout(p=0.1),
            # nn.ReLU()
        )

        self.num_heads = 4
        self.max_len = 512
        self.num_layers = 2
        self.dropout = {'embed': 0.5, 'gaz': 0.5, 'output': 0.3, 'pre': 0.5, 'post': 0.3, 'ff': 0.15, 'ff_2': 0.15,
                        'attn': 0}
        self.pe_ss = nn.Embedding(2 * self.max_len + 1, self.num_heads)
        self.pe_se = nn.Embedding(2 * self.max_len + 1, self.num_heads)
        self.pe_es = nn.Embedding(2 * self.max_len + 1, self.num_heads)
        self.pe_ee = nn.Embedding(2 * self.max_len + 1, self.num_heads)

        self.pe = nn.Embedding(2 * self.max_len + 1, self.num_heads)

        self.four_pos_fusion_embedding = FourPosFusionEmbedding(
            num_heads=self.num_heads,
            pe_ss=self.pe_ss,
            pe_se=self.pe_se,
            pe_es=self.pe_es,
            pe_ee=self.pe_ee,
            max_seq_len=self.max_len,
            hidden_size=self.hidden_dim
        )

        self.encoder = Transformer_Encoder(self.hidden_dim, self.num_heads, self.num_layers,
                                           relative_position=True,
                                           learnable_position=False,
                                           add_position=False,
                                           layer_preprocess_sequence="",
                                           layer_postprocess_sequence="an",
                                           dropout=self.dropout,
                                           scaled=True,
                                           ff_size=self.hidden_dim,
                                           dvc=None,
                                           max_seq_len=self.max_len,
                                           pe_ss=self.pe_ss,
                                           pe_se=self.pe_se,
                                           pe_es=self.pe_es,
                                           pe_ee=self.pe_ee,
                                           k_proj=False,
                                           q_proj=True,
                                           v_proj=True,
                                           r_proj=True,
                                           attn_ff=False,
                                           ff_activate='relu',
                                           lattice=True,
                                           four_pos_fusion="ff_two",
                                           four_pos_fusion_shared=True)

        self.classifier1 = CRFClassifier(self.hidden_dim, num_labels1,
                                         config_char.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(self.hidden_dim, num_labels2,
                                         config_char.hidden_dropout_prob)
        self.init_weights()

    def select_lexical_word(self, word_seq, word_lens, word_mask):
        # word_seq: [B, L, D]
        # word_lens: [B, L]
        # word_mask: [B, L]
        B = word_seq.shape[0]
        D = word_seq.shape[-1]
        selected_word_max_num = (word_lens > 1).sum(dim=1).max().item()
        result = torch.zeros(size=(B, selected_word_max_num, D)).float()
        pos_s = torch.zeros(size=(B, selected_word_max_num)).long()
        pos_e = torch.zeros(size=(B, selected_word_max_num)).long()
        new_mask = torch.zeros(size=(B, selected_word_max_num)).long()

        for b in range(B):
            idx = 0
            cur_char_idx = 0
            for i in range(word_lens.shape[1]):  # along sequence length
                if word_mask[b, i] == 0:
                    break
                if word_lens[b, i] > 1:
                    result[b, idx] = word_seq[b, i]
                    pos_s[b, idx] = cur_char_idx
                    pos_e[b, idx] = cur_char_idx + word_lens[b, i] - 1
                    new_mask[b, idx] = 1
                    idx += 1
                cur_char_idx += word_lens[b, i]
        result = result.to(word_seq.device)
        pos_s = pos_s.to(word_seq.device)
        pos_e = pos_e.to(word_seq.device)
        new_mask = new_mask.to(word_seq.device)
        return result, pos_s, pos_e, new_mask

    def create_word_char_pos(self, word_pos_s, word_pos_e, char_lens, word_lens):
        # char_lens: how many chars in a sentence
        # word_lens: how many words in a sentence
        combined_max_len = (char_lens + word_lens).max().item()
        # combined_max_len = char_lens.max() + word_lens.max()
        # combined_max_len = combined_max_len.item()
        B = word_pos_s.shape[0]
        word_char_pos_s = torch.zeros(size=(B, combined_max_len)).long().to(word_pos_s.device)
        word_char_pos_e = torch.zeros(size=(B, combined_max_len)).long().to(word_pos_e.device)
        for b in range(B):
            cur_char_len = char_lens[b].item()
            word_char_pos_s[b, :cur_char_len] = torch.arange(cur_char_len)
            word_char_pos_e[b, :cur_char_len] = torch.arange(cur_char_len)
            cur_word_len = word_lens[b].item()
            word_char_pos_s[b, cur_char_len:cur_char_len + cur_word_len] = word_pos_s[b, :cur_word_len]
            word_char_pos_e[b, cur_char_len:cur_char_len + cur_word_len] = word_pos_e[b, :cur_word_len]
        return word_char_pos_s, word_char_pos_e

    def compact_char_word_output(self, char_out, word_out, char_lens, word_lens):
        combined_max_len = (char_lens + word_lens).max().item()
        B = char_out.shape[0]
        D = char_out.shape[-1]
        result = torch.zeros(size=(B, combined_max_len, D)).float().to(char_out.device)
        for b in range(B):
            cur_char_len = char_lens[b].item()
            cur_word_len = word_lens[b].item()
            result[b, :cur_char_len, :] = char_out[b, :cur_char_len, :]
            result[b, cur_char_len:cur_word_len + cur_char_len, :] = word_out[b, :cur_word_len, :]
        return result

    def forward(self,
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
                no_decode=False
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
        char_sequence_output = self.char_to_hidden(char_sequence_output)
        # print(f"========> char output shape {char_sequence_output.shape}")

        word_sequence_output = self.word_to_hidden(word_sequence_output)
        # TODO: select those words with len > 1 and concat with chars
        word_sequence_output, pos_s, pos_e, lex_mask = self.select_lexical_word(word_seq=word_sequence_output,
                                                                                word_lens=word_lens,
                                                                                word_mask=word_attention_mask)
        char_len = char_sequence_output.shape[1]
        char_lens = char_attention_mask.sum(1)
        lex_lens = lex_mask.sum(1)

        # transformer_input = torch.cat(
        #     [char_sequence_output, word_sequence_output], dim=1
        # )
        transformer_input = self.compact_char_word_output(char_sequence_output, word_sequence_output, char_lens,
                                                          lex_lens)
        # print(f"=====================+> char lens {char_lens}")
        # print(f"=========================+> Transformer input {transformer_input}")

        word_char_pos_s, word_char_pos_e = self.create_word_char_pos(pos_s, pos_e, char_lens, lex_lens)
        rel_pos_embedding = self.four_pos_fusion_embedding(word_char_pos_s, word_char_pos_e)

        encoded = self.encoder.forward(transformer_input, char_lens, lex_num=lex_lens,
                                       rel_pos_embedding=rel_pos_embedding)
        encoded = encoded[:, :char_len, :]
        # print(f"===========================>", encoded.shape, labels.shape)
        pred1 = self.classifier1.forward(encoded, char_attention_mask, labels, no_decode=no_decode)
        pred2 = self.classifier2.forward(encoded, char_attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(pred1, pred2)

