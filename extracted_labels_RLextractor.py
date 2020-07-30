import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp
import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe
from cytoolz import curry, compose

from utils import count_data
from metric import compute_rouge_l


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')




def decode(model_dir, split, batch_size, cuda):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())

    extractor = RLExtractor(model_dir, cuda=cuda)

    start = time()
    print('start processing {} split...'.format(split))


    i = 0
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with torch.no_grad():
        for i in range(n_data):
            data_dir = join(DATA_DIR, split)
            with open(join(data_dir, '{}.json'.format(i))) as f:
                data = json.loads(f.read())
            art_sents = tokenize(None,data['article'])
            extr = extractor(art_sents)[:-1]  # exclude EOE
            if not extr:
                # use top-5 if nothing is extracted
                # in some rare cases rnn-ext does not extract at all
                extr = list(range(5))[:len(raw_art_sents)]
            else:
                extr = [i.item() for i in extr]
            data['RLextract']=extr
            with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
                json.dump(data, f, indent=4)
    print('finished in {}'.format(timedelta(seconds=time()-start)))






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='making extraction label by trained extractor')
    parser.add_argument('--model_dir', help='root of the full model RL')


    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    for split in ['val', 'train']:  
        decode( args.model_dir,split,args.batch, args.cuda)
