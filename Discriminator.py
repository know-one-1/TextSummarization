""" train disriminator (ML)"""
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

from model.extract import ExtractSumm, PtrExtractSumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer
from model.discriminate import ConvDiscrim

from utils import PAD, UNK
from utils import make_vocab, make_embedding

from data.data import CnnDmDataset, DiscrimDataset
from data.batcher import coll_fn_extract, prepro_fn_discrim
from data.batcher import batchify_fn_discrim_conv
from data.batcher import convert_batch_discrim_conv
from data.batcher import BucketedGenerater
from data.batcher import coll_fn_discrim


BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
    DECODED_DATA_DIR=os.environ['DECODED']
except KeyError:
    print('please use environment variable to specify data directories')

class GetDataset1(CnnDmDataset):
    
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        abstract, target_class = [" ".join(js_data['abstract'])] , 1
        return abstract, target_class


class GetDataset2(DiscrimDataset):
    
    def __init__(self, split):
        super().__init__(split, DECODED_DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        abstract, target_class = [" ".join(js_data['abstract'])], 0
        return abstract, target_class


def build_batchers(net_type, word2id, cuda, debug):
    assert net_type in ['cnn']
    prepro = prepro_fn_discrim(args.max_word)
    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)
    batchify_fn = batchify_fn_discrim_conv 
    convert_batch = convert_batch_discrim_conv 
                     
    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id))
    
    trainset1=GetDataset1('train')
    trainset2=GetDataset2('train')
    Extract_train_set=torch.utils.data.ConcatDataset([trainset1,trainset2])



    train_loader = DataLoader(
        Extract_train_set, batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_discrim
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)


    valset1=GetDataset1('val')
    valset2=GetDataset2('val')
    Extract_val_set=torch.utils.data.ConcatDataset([valset1,valset2])

    val_loader = DataLoader(
        Extract_val_set, batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_discrim
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def configure_net(net_type, vocab_size, emb_dim, conv_hidden,drop_out):
    assert net_type in ['cnn']
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['n_hidden']   = conv_hidden
    net_args['dropout']      = drop_out

    net = (ConvDiscrim(**net_args))
    return net, net_args


def configure_training(net_type, opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    assert net_type in ['cnn']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    loss=torch.nn.BCEWithLogitsLoss(reduction='none')
    if net_type == 'cnn':
        criterion = lambda logit, target: loss(
            logit, target)

    return criterion, train_params


def main(args):
    assert args.net_type in ['cnn']
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(args.net_type, word2id,
                                                args.cuda, args.debug)

    # make net
    net, net_args = configure_net(args.net_type,
                                  len(word2id), args.emb_dim, args.conv_hidden,args.drop_out)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            {i: w for w, i in word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        args.net_type, 'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'ml_{}_dscriminator'.format(args.net_type)
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the convolutional Discriminator'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='cnn',
                        help='model type of the extractor (cnn)')
    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--drop_out', type=int, action='store', default=0.01,
                        help='drop out')
    

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
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
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
