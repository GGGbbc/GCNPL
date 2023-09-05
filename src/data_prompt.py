import torch
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from arguments import get_args
import constant
from tree import Tree, head_to_tree, tree_to_adj
import sys
import pickle


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class REPromptDataset(DictDataset):

    def __init__(self, path=None, name=None, rel2id=None, tokenizer=None, temps=None, features=None):

        with open(rel2id, "r") as f:
            self.rel2id = json.loads(f.read())
        if not 'None' in self.rel2id:
            self.NA_NUM = self.rel2id['False']
        else:
            self.NA_NUM = self.rel2id['None']

        self.num_class = len(self.rel2id)
        self.temps = temps
        self.get_labels(tokenizer)
        self.b = ''
        if features is None:
            self.args = get_args()
            with open(path + "/" + name, "r") as f:
                features = []
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) > 0:
                        features.append(eval(line))
            features = self.list2tensor(features, tokenizer)

        super().__init__(**features)

    def get_labels(self, tokenizer):
        total = {}
        self.temp_ids = {}

        for name in self.temps:
            last = 0
            self.temp_ids[name] = {}
            self.temp_ids[name]['label_ids'] = []
            self.temp_ids[name]['mask_ids'] = []

            for index, temp in enumerate(self.temps[name]['temp']):
                _temp = temp.copy()
                _labels = self.temps[name]['labels'][index]
                _labels_index = []

                for i in range(len(_temp)):
                    if _temp[i] == tokenizer.mask_token:
                        _temp[i] = _labels[len(_labels_index)]
                        _labels_index.append(i)

                original = tokenizer.encode(" ".join(temp), add_special_tokens=False)
                final = tokenizer.encode(" ".join(_temp), add_special_tokens=False)

                assert len(original) == len(final)
                self.temp_ids[name]['label_ids'] += [final[pos] for pos in _labels_index]

                for pos in _labels_index:
                    if not last in total:
                        total[last] = {}
                    total[last][final[pos]] = 1
                    last += 1
                self.temp_ids[name]['mask_ids'].append(original)

        # 0
        self.set = [(list)((sorted)(set(total[i]))) for i in range(len(total))]
        # print ("=================================")
        # for i in self.set:
        #     print (i)
        # print ("=================================")

        for name in self.temp_ids:
            for j in range(len(self.temp_ids[name]['label_ids'])):
                self.temp_ids[name]['label_ids'][j] = self.set[j].index(
                    self.temp_ids[name]['label_ids'][j])

        self.prompt_id_2_label = torch.zeros(len(self.temp_ids), len(self.set)).long()

        for name in self.temp_ids:
            for j in range(len(self.prompt_id_2_label[self.rel2id[name]])):
                self.prompt_id_2_label[self.rel2id[name]][j] = self.temp_ids[name]['label_ids'][j]

        self.prompt_id_2_label = self.prompt_id_2_label.long().cuda()

        self.prompt_label_idx = [
            torch.Tensor(i).long() for i in self.set
        ]

    def save(self, path=None, name=None):
        path = path + "/" + name + "/"
        np.save(path + "input_ids", self.tensors['input_ids'].numpy())
        np.save(path + "token_type_ids", self.tensors['token_type_ids'].numpy())
        np.save(path + "attention_mask", self.tensors['attention_mask'].numpy())
        np.save(path + "labels", self.tensors['labels'].numpy())
        np.save(path + "mlm_labels", self.tensors['mlm_labels'].numpy())
        np.save(path + "input_flags", self.tensors['input_flags'].numpy())

        # -------------------------------将依存句法加入-gbc--------------------------
        np.save(path + "stanford_head", self.tensors['stanford_head'].numpy())
        np.save(path + "stanford_deprel", self.tensors['stanford_deprel'].numpy())
        np.save(path + "stanford_pos", self.tensors['stanford_pos'].numpy())
        np.save(path + "subj_start", self.tensors['subj_start'].numpy())
        np.save(path + "subj_end", self.tensors['subj_end'].numpy())
        np.save(path + "obj_start", self.tensors['obj_start'].numpy())
        np.save(path + "obj_end", self.tensors['obj_end'].numpy())
        np.save(path + "subj_positions", self.tensors['subj_positions'].numpy())
        np.save(path + "obj_positions", self.tensors['obj_positions'].numpy())
        np.save(path + "l", self.tensors['l'].numpy())
        np.save(path + "words", self.tensors['words'].numpy())
        np.save(path + "masks", self.tensors['masks'].numpy())

    @classmethod
    def load(cls, path=None, name=None, rel2id=None, temps=None, tokenizer=None):
        path = path + "/" + name + "/"
        features = {}
        features['input_ids'] = torch.Tensor(np.load(path + "input_ids.npy")).long()
        features['token_type_ids'] = torch.Tensor(np.load(path + "token_type_ids.npy")).long()
        features['attention_mask'] = torch.Tensor(np.load(path + "attention_mask.npy")).long()
        features['labels'] = torch.Tensor(np.load(path + "labels.npy")).long()
        features['input_flags'] = torch.Tensor(np.load(path + "input_flags.npy")).long()
        features['mlm_labels'] = torch.Tensor(np.load(path + "mlm_labels.npy")).long()

        # ------------------------------------gbc--------------------------------------------
        features['stanford_head'] = torch.Tensor(np.load(path + "stanford_head.npy")).long()
        features['stanford_deprel'] = torch.Tensor(np.load(path + "stanford_deprel.npy")).long()
        features['stanford_pos'] = torch.Tensor(np.load(path + "stanford_pos.npy")).long()
        features['subj_start'] = torch.Tensor(np.load(path + "subj_start.npy")).long()
        features['subj_end'] = torch.Tensor(np.load(path + "subj_end.npy")).long()
        features['obj_start'] = torch.Tensor(np.load(path + "obj_start.npy")).long()
        features['obj_end'] = torch.Tensor(np.load(path + "obj_end.npy")).long()
        features['subj_positions'] = torch.Tensor(np.load(path + "subj_positions.npy")).long()
        features['obj_positions'] = torch.Tensor(np.load(path + "obj_positions.npy")).long()
        features['l'] = torch.Tensor(np.load(path + "l.npy")).long()
        features['words'] = torch.Tensor(np.load(path + "words.npy")).long()
        features['masks'] = torch.Tensor(np.load(path + "masks.npy")).long()

        # -----------------------gbc---------------------------
        # features['stanford_head'] = torch.Tensor(np.load(path + "stanford_head.npy")).long()
        # features['stanford_deprel'] = torch.Tensor(np.load(path + "stanford_deprel.npy")).long()

        res = cls(rel2id=rel2id, features=features, temps=temps, tokenizer=tokenizer)

        return res

    # ----------------------------------gbc--------------------------------
    # 定义得到位置标签
    def get_positions(start_idx, end_idx, length):
        """ Get subj/obj position sequence. """
        return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
            list(range(1, length - end_idx))

    def list2tensor(self, data, tokenizer):
        res = {}

        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['input_flags'] = []
        res['mlm_labels'] = []
        res['labels'] = []

        # ---------------------------gbc-------------
        res['stanford_head'] = []
        res['stanford_deprel'] = []
        res['stanford_pos'] = []
        res['subj_start'] = []
        res['subj_end'] = []
        res['obj_start'] = []
        res['obj_end'] = []
        res['subj_positions'] = []
        res['obj_positions'] = []
        res['words'] = []
        res['masks'] = []

        ll = []
        res['l'] = []
        for index, i in enumerate(tqdm(data)):

            input_ids, token_type_ids, input_flags = self.tokenize(i, tokenizer)
            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            ll.append(len(i['token']))

            # ---------------------------------------------gbc----------------------将deprel以及pos转换成ids----------
            i['stanford_deprel'] = map_to_ids(i['stanford_deprel'], constant.DEPREL_TO_ID)
            i['stanford_pos'] = map_to_ids(i['stanford_pos'], constant.POS_TO_ID)
            l = len(i['token'])
            i['subj_positions'] = get_positions(i['subj_start'], i['subj_end'], l)
            i['obj_positions'] = get_positions(i['obj_start'], i['obj_end'], l)
            # ------------------------------gbc----------------------使用训练好的voca以及embeding-------------
            vocab_file = '../dataset/vocab/vocab.pkl'
            vocab = load_vocab_from_pickle(vocab_file)

            def map_tokens_to_ids(tokens, vocab):
                unk_id = constant.UNK_ID  # 未知标记的索引为vocab列表的长度
                ids = [vocab.index(token) if token in vocab else unk_id for token in tokens]
                return ids

            tokens = i['token']
            words = map_tokens_to_ids(tokens, vocab)

            i['words'] = words

            if padding_length > 0:
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                input_flags = input_flags + ([0] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            assert len(input_flags) == self.args.max_seq_length

            label = self.rel2id[i['relation']]
            res['input_ids'].append(np.array(input_ids))
            res['token_type_ids'].append(np.array(token_type_ids))
            res['attention_mask'].append(np.array(attention_mask))
            res['input_flags'].append(np.array(input_flags))
            res['labels'].append(np.array(label))
            mask_pos = np.where(res['input_ids'][-1] == tokenizer.mask_token_id)[0]
            mlm_labels = np.ones(self.args.max_seq_length) * (-1)
            mlm_labels[mask_pos] = 1
            res['mlm_labels'].append(mlm_labels)

            # -------------------------gbc--------------------------
            res['stanford_head'].append(i['stanford_head'])
            res['stanford_deprel'].append(i['stanford_deprel'])
            res['stanford_pos'].append(i['stanford_pos'])
            res['subj_start'].append(i['subj_start'])
            res['subj_end'].append(i['subj_end'])
            res['obj_start'].append(i['obj_start'])
            res['obj_end'].append(i['obj_end'])
            res['subj_positions'].append(i['subj_positions'])
            res['obj_positions'].append(i['obj_positions'])
            res['words'].append(i['words'])
        with open('../errortxt.txt','w') as f:
            f.write(self.b)
        # -------------------------gbc----将stanford_head写入--------------
        for key in res:
            if key in ['input_ids', 'token_type_ids', 'attention_mask', 'mlm_labels', 'input_flags', 'labels',
                       'subj_start', 'subj_end', 'obj_start', 'obj_end']:
                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

            if key == 'stanford_head':
                tem = []
                for ite in res[key]:
                    pad_len = self.args.max_seq_length - len(ite)
                    ite = ite + ([0] * pad_len)
                    tem.append(np.array(ite, dtype=int))
                    res['stanford_head'] = tem

                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

            if key == 'stanford_deprel':
                dep = []
                for deprel in res[key]:
                    pad_len = self.args.max_seq_length - len(deprel)
                    deprel = deprel + ([0] * pad_len)
                    dep.append(np.array(deprel, dtype=int))
                    res['stanford_deprel'] = dep

                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

            if key == 'stanford_pos':
                pos = []
                for poss in res[key]:
                    pad_len = self.args.max_seq_length - len(poss)
                    poss = poss + ([0] * pad_len)
                    pos.append(np.array(poss, dtype=int))
                    res['stanford_pos'] = pos

                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

            if key == 'subj_positions':
                sub_pos = []
                for sub_postion in res[key]:
                    pad_len = self.args.max_seq_length - len(sub_postion)
                    sub_postion = sub_postion + ([0] * pad_len)
                    sub_pos.append(np.array(sub_postion, dtype=int))
                    res['subj_positions'] = sub_pos

                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

            if key == 'obj_positions':
                obj_pos = []
                for obj_position in res[key]:
                    pad_len = self.args.max_seq_length - len(obj_position)
                    obj_position = obj_position + ([0] * pad_len)
                    obj_pos.append(np.array(obj_position, dtype=int))
                    res['obj_positions'] = obj_pos

                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

            if key == 'words':
                words = []
                for word in res[key]:
                    pad_len = self.args.max_seq_length - len(word)
                    word = word + ([0] * pad_len)
                    words.append(np.array(word, dtype=int))
                    res['words'] = words

                res[key] = np.array(res[key])
                res[key] = torch.Tensor(res[key]).long()

                masks = torch.eq(res['words'], 0)

                res['masks'] = masks



        res['l'] = ll
        res['l'] = np.array(res['l'])
        res['l'] = torch.Tensor(res['l']).long()

        result = res
        # # ---------------------------------测试gbc-------------------------------------
        # maxlen = self.args.max_seq_length
        # torch.set_printoptions(edgeitems=256, threshold=256 * 256)
        # l = len(i['stanford_head'])
        # l = np.array([l])
        #
        # def inputs_to_tree_reps(head, l):
        #     trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
        #     adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
        #     adj = np.concatenate(adj, axis=0)
        #     adj = torch.from_numpy(adj)
        #     return adj
        #
        # adj = inputs_to_tree_reps(result['stanford_head'], l)
        #
        # count = (adj == 1).sum().item()
        # print(count)
        # print(result['stanford_head'][0])
        # print(adj[0])
        return result
        # return ""

    def tokenize(self, item, tokenizer):

        sentence = item['token']
        pos_head = item['h']
        pos_tail = item['t']
        rel_name = item['relation']

        temp = self.temps[rel_name]

        sentence = " ".join(sentence)
        sentence = tokenizer.encode(sentence, add_special_tokens=False)
        e1 = tokenizer.encode(" ".join(['was', pos_head['name']]), add_special_tokens=False)[1:]
        e2 = tokenizer.encode(" ".join(['was', pos_tail['name']]), add_special_tokens=False)[1:]

        # prompt =  [tokenizer.unk_token_id, tokenizer.unk_token_id] + \
        prompt = self.temp_ids[rel_name]['mask_ids'][0] + e1 + \
                 self.temp_ids[rel_name]['mask_ids'][1] + \
                 self.temp_ids[rel_name]['mask_ids'][2] + e2
        #  + \
        #  [tokenizer.unk_token_id, tokenizer.unk_token_id]

        flags = []
        last = 0
        for i in prompt:
            # if i == tokenizer.unk_token_id:
            #     last+=1
            #     flags.append(last)
            # else:
            flags.append(0)

        tokens = sentence + prompt
        flags = [0 for i in range(len(sentence))] + flags
        # tokens = prompt + sentence
        # flags =  flags + [0 for i in range(len(sentence))]        

        tokens = self.truncate(tokens,
                               max_length=self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))
        flags = self.truncate(flags,
                              max_length=self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens)
        input_flags = tokenizer.build_inputs_with_special_tokens(flags)
        input_flags[0] = 0
        input_flags[-1] = 0
        assert len(input_ids) == len(input_flags)
        assert len(input_ids) == len(token_type_ids)
        return input_ids, token_type_ids, input_flags
    def truncate(self, seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            print("=========")


            a = tokenizer.decode(seq)
            self.b += a + '\n'
            return seq[len(seq) - max_length:]


# ----------------------------------------gbc---将deprel以及pos转化为ids--------------------------------

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
        list(range(1, length - end_idx))


def load_vocab_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
