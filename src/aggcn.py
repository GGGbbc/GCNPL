"""
GCN model for relation extraction.
"""
import copy
import math

"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from tree import head_to_tree, tree_to_adj
from utils import constant, torch_utils


# class GCNClassifier(nn.Module):
#     """ A wrapper classifier for GCNRelationModel. """
#
#     def __init__(self, args, emb_matrix=None):
#         super().__init__()
#         self.gcn_model = GCNRelationModel(args, emb_matrix=emb_matrix)
#         in_dim = args.hidden_dim
#         self.classifier = nn.Linear(in_dim, args['num_class'])
#         self.args = args
#
#     def forward(self, inputs):
#         outputs, pooling_output = self.gcn_model(inputs)
#         logits = self.classifier(outputs)
#         return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), args.pos_dim) if args.pos_dim > 0 else None
        # self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), args['ner_dim']) if args['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = AGGCN(args, embeddings)

        # mlp output layer
        in_dim = args.hidden_dim * 3
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(self.args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
    # 初始化嵌入层的权重，可以从外部加载预训练的嵌入向量
    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.args.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.args.topn < self.args.vocab_size:
            print("Finetune top {} word embeddings.".format(self.args.topn))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.args.topn))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words = inputs['words']
        masks = inputs['masks']
        pos = inputs['stanford_pos']
        deprel = inputs['stanford_deprel']
        head = inputs['stanford_head']
        subj_pos = inputs['subj_positions']
        obj_pos = inputs['obj_positions']

        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = self.args.max_seq_length

        # 将依赖关系信息转换为树表示

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.args.cuda else Variable(adj)

        adj = inputs_to_tree_reps(head.data, l)
        h, pool_mask = self.gcn(adj, inputs)#获得隐藏层和邻接矩阵

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = self.args.pooling
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type="max")
        obj_out = pool(h, obj_mask, type="max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)

        return outputs, h_out

class AGGCN(nn.Module):
    def __init__(self, args, embeddings):
        super().__init__()
        self.args = args
        self.in_dim = args.emb_dim + args.pos_dim
        self.emb, self.pos_emb = embeddings
        self.use_cuda = args.cuda
        self.mem_dim = args.hidden_dim

        # rnn layer
        if self.args.rnn:
            self.input_W_R = nn.Linear(self.in_dim, args.rnn_hidden)
            self.rnn = nn.LSTM(args.rnn_hidden, args.rnn_hidden, args.rnn_layers, batch_first=True, \
                               dropout=args.rnn_dropout, bidirectional=True)
            self.in_dim = args.rnn_hidden * 2
            self.rnn_drop = nn.Dropout(args.rnn_dropout)  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(args.input_dropout)
        self.num_layers = args.num_layers

        self.layers = nn.ModuleList()

        self.heads = args.heads
        self.sublayer_first = args.sublayer_first
        self.sublayer_second = args.sublayer_second

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(GraphConvLayer(args, self.mem_dim, self.sublayer_first))
                self.layers.append(GraphConvLayer(args, self.mem_dim, self.sublayer_second))
            else:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))

        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, self.args.rnn_layers)
        #rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        #rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        # words, masks, pos, deprel, head, subj_pos, obj_pos = inputs # unpack
        words = inputs['words']
        masks = inputs['masks']
        pos = inputs['stanford_pos']
        deprel = inputs['stanford_deprel']
        head = inputs['stanford_head']
        subj_pos = inputs['subj_positions']
        obj_pos = inputs['obj_positions']

        src_mask = (words != constant.PAD_ID).unsqueeze(-2)

        word_embs = self.emb(words)
        embs = [word_embs]

        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        if self.args.rnn:
            embs = self.input_W_R(embs)
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs
        gcn_inputs = self.input_W_G(gcn_inputs)

        layer_list = []
        outputs = gcn_inputs
        for i in range(len(self.layers)):
            if i < 2:
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)
            else:
                attn_tensor = self.attn(outputs, outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                outputs = self.layers[i](attn_adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        return dcgcn_output, mask


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def pool(h, mask, type='max'):
    if type == 'max':
        #mask = mask.unsqueeze(1)
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)




def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        # query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn

