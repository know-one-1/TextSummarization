import argparse
import json
import re
import os
from os.path import join, exists
import pickle as pkl
from itertools import starmap

from cytoolz import curry
from cytoolz import compose


import torch
from torch.utils.data import DataLoader




from utils import PAD, UNK
from utils import make_vocab, make_embedding
from decoding import load_best_ckpt

from model.discriminate import CopyConvDiscrim
from data.data import CnnDmDataset, DiscrimDataset
from data.batcher import  copy_prepro_fn_discrim
from CopyDiscriminator import CopyGetDataset1, CopyGetDataset2, GetArticle, Concat
from data.batcher import conver2id, copy_convert_batch_discrim_conv, copy_batchify_fn_discrim_conv
from data.batcher import copy_coll_fn_discrim
    

try:
    DATA_DIR = os.environ['DATA']
    DECODED_DATA_DIR=os.environ['DECODED']
except KeyError:
    print('please use environment variable to specify data directories')




class CopyDiscriModel(object):
    def __init__(self, discrim_dir,  cuda=True):
        discrim_meta = json.load(open(join(discrim_dir, 'meta.json')))
        assert discrim_meta['net'] == 'ml_cnn_dscriminator'
        discrim_args = discrim_meta['net_args']
        discrim_ckpt = load_best_ckpt(discrim_dir)
        word2id = pkl.load(open(join(discrim_dir, 'vocab.pkl'), 'rb'))
        discriminator = CopyConvDiscrim(**discrim_args)
        discriminator.load_state_dict(discrim_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self.cuda=cuda
        self._net = discriminator.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._args = discrim_meta

    def _prepro(self, batch):
        assert len(batch)!=0
        
        sen2id=compose(copy_batchify_fn_discrim_conv(PAD, cuda=self.cuda),
                       copy_convert_batch_discrim_conv(UNK, self._word2id),copy_prepro_fn_discrim(100))
        return sen2id(batch)

    def __call__(self, processed_batch):
        self._net.eval()
        pos=0
        fw_args, res_args = self._prepro(processed_batch)
        out_args = self._net(*fw_args)
        sigmoid=torch.nn.Sigmoid()
        out_args=sigmoid(out_args)

        for x,y in zip(res_args[0], out_args):
            if torch.argmax(x)==torch.argmax(y):
                pos+=1
        return pos,out_args.size()[0]


    


def main(args):

    testset1=CopyGetDataset1('test')
    testset2=Concat([GetArticle('test'),CopyGetDataset2('test')])
    Extract_test_set=torch.utils.data.ConcatDataset([testset1,testset2])

    test_loader = DataLoader(
        Extract_test_set, batch_size=2000,
        shuffle=False,
        num_workers=4 ,
        collate_fn=copy_coll_fn_discrim
    )

    discri=CopyDiscriModel(args.path)
    correct_total=0
    total=0
    for i,batch in enumerate(test_loader):
        crct, totl = discri(batch)
        correct_total+=crct
        total+=totl
        if i%10==0:
            print("Current Accuracy {}".format(float(correct_total/total)))   
    print("Overall Accuracy {}".format(float(correct_total/total)))   




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='testing of the Discriminator'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='cnn',
                        help='model type of the extractor (cnn)')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')

    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the test batch size')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
