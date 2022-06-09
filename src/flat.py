import torch
from torch import nn
from fastNLP import seq_len_to_mask
import collections
import math
import torch.nn.functional as F


class FourPosFusionEmbedding(nn.Module):
    def __init__(self, num_heads, pe_ss, pe_se, pe_es, pe_ee, max_seq_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.num_heads = num_heads
        self.pos_fusion_forward = nn.Sequential(nn.Linear(self.num_heads * 4, self.num_heads),
                                                nn.ReLU(inplace=True))

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)
        # 这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.max_seq_len
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.max_seq_len
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.max_seq_len
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.max_seq_len

        max_seq_len = pos_s.size(1)

        pe_ss = self.pe_ss(pos_ss).view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_se = self.pe_se(pos_se).view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es(pos_es).view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee(pos_ee).view(size=[batch, max_seq_len, max_seq_len, -1])

        pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
        rel_pos_embedding = self.pos_fusion_forward(pe_4)
        rel_pos_embedding = rel_pos_embedding.permute(0, 3, 1, 2)

        return rel_pos_embedding


class Transformer_Encoder_Layer(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 max_seq_len=-1, pe=None,
                 pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 dvc=None,
                 k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_ff=True, ff_activate='relu', lattice=False,
                 four_pos_shared=True, four_pos_fusion=None, four_pos_fusion_embedding=None
                 ):
        super().__init__()
        self.four_pos_fusion_embedding = four_pos_fusion_embedding
        self.four_pos_shared = four_pos_shared
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.lattice = lattice
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if self.relative_position and max_seq_len < 0:
            print('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        import copy
        # if self.relative_position:
        #     if pe is None:
        #         pe = get_embedding(max_seq_len, hidden_size, rel_pos_init=self.rel_pos_init)
        #         pe_sum = pe.sum(dim=-1, keepdim=True)
        #         if self.pos_norm:
        #             with torch.no_grad():
        #                 pe = pe / pe_sum
        #         self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
        #         if self.four_pos_shared:
        #             self.pe_ss = self.pe
        #             self.pe_se = self.pe
        #             self.pe_es = self.pe
        #             self.pe_ee = self.pe
        #         else:
        #             self.pe_ss = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        #             self.pe_se = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        #             self.pe_es = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        #             self.pe_ee = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        #     else:
        #         self.pe = pe
        #         self.pe_ss = pe_ss
        #         self.pe_se = pe_se
        #         self.pe_es = pe_es
        #         self.pe_ee = pe_ee
        # if self.four_pos_fusion_embedding is None:
        #     self.four_pos_fusion_embedding = \
        #         Four_Pos_Fusion_Embedding(self.pe, num_heads, self.four_pos_fusion, self.pe_ss, self.pe_se, self.pe_es,
        #                                   self.pe_ee,
        #                                   self.max_seq_len, self.hidden_size, self.mode)

        if dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        ff_size = hidden_size
        self.ff_size = ff_size
        # print('dropout:{}'.format(self.dropout))
        self.layer_preprocess = LayerProcess(self.layer_preprocess_sequence, self.hidden_size, self.dropout['pre'])
        self.layer_postprocess = LayerProcess(self.layer_postprocess_sequence, self.hidden_size, self.dropout['post'])
        # if self.relative_position:
        self.attn = MultiHead_Attention_Lattice_rel_save_gpumm(self.hidden_size, self.num_heads,
                                                               pe_ss=self.pe_ss,
                                                               pe_se=self.pe_se,
                                                               pe_es=self.pe_es,
                                                               pe_ee=self.pe_ee,
                                                               scaled=self.scaled,
                                                               max_seq_len=self.max_seq_len,
                                                               dvc=self.dvc,
                                                               k_proj=self.k_proj,
                                                               q_proj=self.q_proj,
                                                               v_proj=self.v_proj,
                                                               r_proj=self.r_proj,
                                                               attn_dropout=self.dropout['attn'],
                                                               ff_final=self.attn_ff,
                                                               four_pos_fusion=self.four_pos_fusion)

        # else:
        #     self.attn = MultiHead_Attention(self.hidden_size, self.num_heads, self.scaled, mode=self.mode,
        #                                     k_proj=self.k_proj, q_proj=self.q_proj, v_proj=self.v_proj,
        #                                     attn_dropout=self.dropout['attn'],
        #                                     ff_final=self.attn_ff)

        self.ff = PositionwiseFeedForward([hidden_size, ff_size, hidden_size], self.dropout,
                                          ff_activate=self.ff_activate)

    def forward(self, inp, seq_len, lex_num=0, rel_pos_embedding=None):
        output = inp
        output = self.layer_preprocess(output)
        if self.lattice:
            if self.relative_position:
                output = self.attn(output, output, output, seq_len, lex_num=lex_num, rel_pos_embedding=rel_pos_embedding)
            else:
                output = self.attn(output, output, output, seq_len, lex_num)
        else:
            output = self.attn(output, output, output, seq_len)
        output = self.layer_postprocess(output)
        output = self.layer_preprocess(output)
        output = self.ff(output)
        output = self.layer_postprocess(output)

        return output


class LayerProcess(nn.Module):
    def __init__(self, process_sequence, hidden_size, dropout=0, ):
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = MyDropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp):
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = output + inp
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)

        return output


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0) / (1 - self.p)
        return x


class MultiHead_Attention_Lattice_rel_save_gpumm(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 pe_ss, pe_se, pe_es, pe_ee,
                 scaled=True, max_seq_len=-1,
                 dvc=None, k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_dropout=None,
                 ff_final=True,
                 four_pos_fusion=None):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 4, 4),
                                                nn.Softmax(dim=-1))

        elif self.four_pos_fusion == 'gate':
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 2, 4 * self.hidden_size))

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

        # self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, key, query, value, seq_len, lex_num, rel_pos_embedding):
        batch = key.size(0)

        if self.k_proj:
            key = self.w_k(key)
        if self.q_proj:
            query = self.w_q(query)
        if self.v_proj:
            value = self.w_v(value)

        batch = key.size(0)
        max_seq_len = key.size(1)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        attention_scores = torch.matmul(query, key)

        if self.scaled:
            attention_scores = attention_scores / math.sqrt(self.per_head_size)
        attention_scores = attention_scores + rel_pos_embedding
        mask = seq_len_to_mask(seq_len + lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attention_scores.masked_fill(~mask, -1e15)
        # if self.mode['debug']:
        #     print('attn_score_raw_masked:{}'.format(attn_score_raw_masked))
        #     print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        if hasattr(self, 'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self, max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index


class PositionwiseFeedForward(nn.Module):
    def __init__(self, sizes, dropout=None, ff_activate='relu'):
        super().__init__()
        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))

        if dropout == None:
            dropout = collections.defaultdict(int)

        self.dropout = MyDropout(dropout['ff'])
        self.dropout_2 = MyDropout(dropout['ff_2'])
        if ff_activate == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif ff_activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0:
                output = self.activate(output)
            w = getattr(self, 'w' + str(i))
            output = w(output)
            if i == 0:
                output = self.dropout(output)
            if i == 1:
                output = self.dropout_2(output)

        return output


class Transformer_Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 dvc=None, max_seq_len=-1,
                 pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_ff=True, ff_activate='relu', lattice=False,
                 four_pos_shared=True, four_pos_fusion=None, four_pos_fusion_shared=True):
        '''

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param relative_position: bool
        :param learnable_position: bool
        :param add_position: bool, if False, concat
        :param layer_preprocess:
        :param layer_postprocess:
        '''
        super().__init__()
        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.four_pos_shared = four_pos_shared
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        # if self.four_pos_fusion_shared:
        #     self.four_pos_fusion_embedding = \
        #         FourPosFusionEmbedding(self.pe, num_heads, self.four_pos_fusion, self.pe_ss, self.pe_se, self.pe_es,
        #                                   self.pe_ee,
        #                                   self.max_seq_len, self.hidden_size, self.mode)
        # else:
        #     self.four_pos_fusion_embedding = None

        self.lattice = lattice
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc

        if self.relative_position and max_seq_len < 0:
            print('max_seq_len should be set if relative position encode')
            exit(1208)

        if dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size

        for i in range(self.num_layers):
            setattr(self, 'layer_{}'.format(i), Transformer_Encoder_Layer(hidden_size, num_heads,
                                                                          relative_position, learnable_position,
                                                                          add_position,
                                                                          layer_preprocess_sequence,
                                                                          layer_postprocess_sequence,
                                                                          dropout, scaled, ff_size,
                                                                          max_seq_len=self.max_seq_len,
                                                                          pe_ss=self.pe_ss,
                                                                          pe_se=self.pe_se,
                                                                          pe_es=self.pe_es,
                                                                          pe_ee=self.pe_ee,
                                                                          k_proj=self.k_proj,
                                                                          q_proj=self.q_proj,
                                                                          v_proj=self.v_proj,
                                                                          r_proj=self.r_proj,
                                                                          attn_ff=self.attn_ff,
                                                                          ff_activate=self.ff_activate,
                                                                          lattice=self.lattice,
                                                                          four_pos_shared=self.four_pos_shared,
                                                                          four_pos_fusion=self.four_pos_fusion,
                                                                          ))

        self.layer_preprocess = LayerProcess(self.layer_preprocess_sequence, self.hidden_size)

    def forward(self, inp, seq_len, lex_num=0, rel_pos_embedding=None):
        output = inp
        for i in range(self.num_layers):
            now_layer = getattr(self, 'layer_{}'.format(i))
            output = now_layer(output, seq_len, lex_num=lex_num, rel_pos_embedding=rel_pos_embedding)

        output = self.layer_preprocess(output)

        return output
