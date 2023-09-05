import torch
import torch.nn as nn
import numpy as np
from arguments import get_model_classes, get_args
from utils.tree import Tree, head_to_tree, tree_to_adj
from aggcn import *


class Model(torch.nn.Module):

    def __init__(self, args, tokenizer=None, prompt_label_idx=None):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        self.prompt_label_idx = prompt_label_idx

        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
        )

        self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)

        self.aggcn = GCNRelationModel(args)

        self.linear = nn.Linear(360, 1024)

        self.gate = GateMechanism(1024)

    def forward(self, input_ids, attention_mask, token_type_ids, input_flags, mlm_labels, labels, l, **kwargs):
        # words, masks, pos, deprel, head, subj_pos, obj_pos = kwargs
        # -------------------------------------gbc--------------------
        # 调用aggcn抽取出带有语法信息的隐藏层状态
        outputs, h_out = self.aggcn(kwargs)
        # 使用unsqueeze在第二维度添加一个新维度
        outputs = outputs.unsqueeze(1)
        # 使用expand方法在第二维度复制5次
        outputs = outputs.expand(-1, 5, -1)

        gcn_out = self.linear(outputs)

        raw_embeddings = self.model.embeddings.word_embeddings(input_ids)
        new_token_embeddings = self.mlp(self.extra_token_embeddings.weight)
        new_embeddings = new_token_embeddings[input_flags]
        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings, raw_embeddings)
        hidden_states, _ = self.model(inputs_embeds=inputs_embeds,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)
        hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), len(self.prompt_label_idx), -1)

        #hidden_states = self.gate(hidden_states,gcn_out)
        hidden_states = hidden_states + gcn_out
        logits = [
            torch.mm(
                hidden_states[:, index, :],
                self.model.embeddings.word_embeddings.weight[i].transpose(1, 0)
            )
            for index, i in enumerate(self.prompt_label_idx)
        ]
        print(logits)
        return logits


def get_model(tokenizer, prompt_label_idx):
    args = get_args()
    model = Model(args, tokenizer, prompt_label_idx)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model


def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer


# -----------------------gbc------门控单元--------
class GateMechanism(nn.Module):
    def __init__(self, input_dim):
        super(GateMechanism, self).__init__()
        self.gate_layer = nn.Linear(input_dim * 2, 1)  # 输入维度是两个特征拼接后的维度，输出维度是1

    def forward(self, input_A, input_B):
        # 将A和B的特征拼接起来
        combined_input = torch.cat((input_A, input_B), dim=-1)
        # 计算门控单元的输出
        gate_output = torch.sigmoid(self.gate_layer(combined_input))

        # 应用门控单元的输出，融合A和B的特征
        fused_feature = gate_output * input_A + (1 - gate_output) * input_B
        return fused_feature
# ---------------------------------------------------------

class GateMechanism(nn.Module):
    def __init__(self, input_dim):
        super(GateMechanism, self).__init__()
        self.gate_layer = nn.Linear(input_dim * 2, 1) # 输入维度是两个特征拼接后的维度，输出维度是1

    def forward(self, input_A, input_B):
        # 将A和B的特征拼接起来
        combined_input = torch.cat((input_A, input_B), dim=-1)
        # 计算门控单元的输出
        gate_output = torch.sigmoid(self.gate_layer(combined_input))

        # 应用门控单元的输出，融合A和B的特征
        fused_feature = gate_output * input_A + (1 - gate_output) * input_B
        return fused_feature
#---------------------------------------------------------