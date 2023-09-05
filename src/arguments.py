import argparse
import torch
import transformers
from utils.vocab import *
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertModel,
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaModel,
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertModel,
    }
}

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default="albert", type=str, required=True, choices=_MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--new_tokens", default=5, type=int, 
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_for_new_token", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--temps", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
#--------------------------------------gbc---------------------
    parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=360, help='RNN hidden state size.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='GCN layer dropout rate.')
    parser.add_argument('--cnn_dropout', type=float, default=0.5, help='CNN layer dropout rate.')
    parser.add_argument('--word_dropout', type=float, default=0.04,
                        help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
    parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)

    parser.add_argument('--heads', type=int, default=3, help='Num of heads in multi-head attention.')
    parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')
    parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

    parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
    parser.add_argument('--pooling', choices=['max', 'avg', 'sum', 'self-att', 'cnn'], default='max',
                        help='Pooling function type. Default max.')
    parser.add_argument('--pooling_l2', type=float, default=0.002, help='L2-penalty for all pooling output.')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

    parser.add_argument('--rnn', type=bool,  default=True, help='Do not use RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

    parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--id', type=str, default='1', help='Model ID under which to save models.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())

    vocab_file = '../dataset/vocab/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    vocab_size = vocab.size
    parser.add_argument('--vocab_size', type=str, default=vocab_size, help='Optional info for the experiment.')

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES