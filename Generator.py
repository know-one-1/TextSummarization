""" train the abstractor"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.copy_summ import CopySumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from gen_trainer import GenPipeline, GenTrainer, get_PG_loss, basic_validate_gen, GenAbstractor, GenRLExtractor

from data.data import CnnDmDataset
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy, batchify_fn_copy
from data.batcher import BucketedGenerater
from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from copyDiscri_evaluate import CopyDiscriModel
from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding
# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 300

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class GenDataset(CnnDmDataset):

    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['RLextract'])
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts, abs_sents[:len(extracts)]


class DisModel(CopyDiscriModel):
    def __init__(self,model_dir,cuda):
        super().__init__(model_dir, cuda)


    def __call__(self, processed_batch):
        self._net.eval()
        out_args = self._net(*processed_batch)
        sigmoid=torch.nn.Sigmoid()
        out_args=sigmoid(out_args)
        return out_args



def configure_net(model_dir, max_len, cuda):

    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    abstractor = GenAbstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
    net = abstractor._net
    discriminator = DisModel(args.dis_dir, cuda)

    net_args={}
    net_args['generator']= meta['net_args']['abstractor']
    net_args['discrimiator'] = discriminator._args

    return net,  discriminator, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    def criterion(prob, rewards):
        return get_PG_loss(prob, rewards)

    return criterion, train_params

def build_batchers(word2id, cuda, debug):
    prepro = prepro_fn(args.max_art, args.max_abs)
    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy(PAD, START, END, cuda=cuda),
        convert_batch_copy(UNK, word2id)
    )


    train_loader = DataLoader(
        GenDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )

    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        GenDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher

def main(args):
    # create data batcher, vocabulary
    # batcher


    # make net
    net,  dis, net_args = configure_net(args.model_dir, args.max_art, args.cuda)

    train_batcher, val_batcher = build_batchers(dis._word2id,
                                                args.cuda, args.debug)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            {i: w for w, i in dis._word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(dis._word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'base-abstractor'
    meta['net_args']      = net_args['generator']
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate_gen(net, dis, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = GenPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer,  dis, grad_fn)
    trainer = GenTrainer(pipeline, args.path,  GenDataset('train'),
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--path', required=True, help=' model to be saved')
    parser.add_argument('--model_dir', required=True, help='root of the model')

    parser.add_argument('--dis_dir', required=True,help='discriminator directory')


    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=30,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
