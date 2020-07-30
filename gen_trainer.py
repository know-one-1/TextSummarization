""" module providing basic training utilities"""
import os
from os.path import join
import json
import pickle as pkl
from time import time
from datetime import timedelta
from itertools import starmap
import copy
from cytoolz import curry, reduce
from cytoolz import compose
from collections import defaultdict
from itertools import cycle
from toolz.sandbox import unzip
import gc

import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX
from data.batcher import tokenize
from model.copy_summ import CopySumm
from model.extract import ExtractSumm, PtrExtractSumm
from model.rl import ActorCritic
from decoding import load_best_ckpt, ArticleBatcher
from utils import UNK,PAD, START ,END
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy, batchify_fn_copy
from training import get_basic_grad_fn
 








@curry
def compute_loss_gen(net, dis, criterion, fw_args, loss_args):
    net_out = net(*fw_args)
    dis._net.eval()
    prob, reward = get_reward(dis, net_out, fw_args)
    loss = criterion(torch.max(net_out,dim=2)[0], reward)
    return loss

@curry
def val_step_gen(loss_step, fw_args, loss_args):
    loss = loss_step(fw_args, loss_args)
    return loss_args[0].size(0), loss.sum().item()

@curry
def basic_validate_gen(net, dis, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step_gen(compute_loss_gen(net, dis, criterion))
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
    val_loss = tot_loss / n_data
    print(
        'validation finished in {} '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}

def get_PG_loss(prob, rewards):
    size = prob.size()
    inter = torch.zeros(size[0],size[1])
    for i in range (size[0]):
        if i==0:
            inter[i]=-prob[i]*rewards[i] 
    return inter.mean()


def id2id( unk, word2id, id2word, id_list):
    id2word = defaultdict( lambda : unk , id2word)
    words_list = [[id2word[int(i)] for i in idx] for idx in id_list]
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]



def kernel_filter(src,abst):
    def good(d):
        src,tgt=d
        if len(tgt)<5:
          tgt+=[0]*(5-len(tgt))
        if len(src)<5:
          src+=[0]*(5-len(tgt))
        return src,tgt
    data=list(zip(src,abst))
    return list(map(good,data))


def get_reward(dis, net_out, fw_args):
    enc_abs = torch.max(net_out,dim=2)
    enc_abs_prob = enc_abs[0] 
    enc_abs_tok = enc_abs[1]
    src, a, b, c, d = fw_args
    enc_src = id2id(UNK, dis._word2id, dis._id2word, src)
    abs_tok = id2id(UNK, dis._word2id, dis._id2word, enc_abs_tok)
    tensor_type = torch.cuda.LongTensor if dis.cuda else torch.LongTensor
    enc_src, abstok = unzip(kernel_filter(enc_src, abs_tok))
    net_in = tensor_type(list(enc_src)),tensor_type(list(abs_tok))
    intem_reward = dis(net_in)
    del a,b,c,d
    reward = [torch.max(x,dim=0)[0] if torch.max(x,dim=0)[1]==1 else 0 for x in intem_reward]
    return enc_abs_prob, reward

class GenPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, discrim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        self.dis = discrim

        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()

    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1



    def train_step(self):
#        forward pass of model
            # 
            self._net.train()
            fw_args, bw_args = next(self._batches)
            net_out = self._net(*fw_args)
    # 
            # 
            self.dis._net.eval()
            prob, reward = get_reward(self.dis, net_out.detach(), fw_args)
            log_dict={}
    
            loss = self._criterion(torch.max(net_out,dim=2)[0], reward).mean()
            del fw_args , bw_args, prob, reward
    
            loss.backward()
    
            log_dict['loss'] = loss.item()
            if self._grad_fn is not None:
                log_dict.update(self._grad_fn())
            self._opt.step()
            self._net.zero_grad()
            del loss 
# 
# 
            return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


class GenTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir,  GenDataset ,ckpt_freq, patience,
                 scheduler=None, val_mode='loss'):
        assert isinstance(pipeline, GenPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode
        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None
        self.discrim = dis_train(GenDataset, self._pipeline.dis, self._pipeline._net, self._pipeline._batch_size)


    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        print('train step: {}, {}: {:.4f}\r'.format(
            self._step,
            'loss' if 'loss' in log_dict else 'reward',
            self._running_loss), end='')
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        val_log = self._pipeline.validate()
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')

            while True:
                log_dict = self._pipeline.train_step()
                self._step += 1
                self.log(log_dict)
                if self._step % self._pipeline._batch_size == 0:
                    self.discrim.train_step()


                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()





class dis_train(object):
    def __init__(self,GenDataset , dis, gen,  batch_size):

        self.train_loader = cycle(DataLoader(
            GenDataset, batch_size=batch_size,
            shuffle=False,
            num_workers=4  ,
            collate_fn = coll_fn
            ))
        self.gen = gen
        self._batch_size = batch_size
        self.word2id = dis._word2id
        self.id2word  = dis._id2word
        self.cuda = dis.cuda
        self.dis = dis._net.train()
        self.opt = optim.Adam(self.dis.parameters(),lr=1e-4)
        self.loss=torch.nn.BCEWithLogitsLoss()
        self.btch = self.get_batch()



    def get_batch(self):
        for i,batch in enumerate(self.train_loader):
            yield batch

    def train_step(self):
        sample2id4gen = compose(batchify_fn_copy(PAD, START, END, cuda=self.cuda),
        convert_batch_copy(UNK,self.word2id ),prepro_fn(100, 30))
        sample2id4dis = compose(batchify_fn_copy(PAD, START, END, cuda=self.cuda),
        convert_batch_copy(UNK, self.word2id),prepro_fn(100, 30))
        data = self.btch.__next__()
        ext_art, _ = sample2id4gen(data)
        src, abstract = sample2id4dis(data)
        abstract = abstract[0] 
        with torch.no_grad():
            gen_abst = self.gen(*ext_art)
        
        enc_abs = torch.max(gen_abst,dim=2)
        enc_gen_abs_tok = enc_abs[1]
        ori_art, _, _, _, _ = src
        

        enc_ori_art = id2id(UNK,self.word2id, self.id2word, ori_art)
        enc_ori_abst = id2id(UNK, self.word2id, self.id2word, abstract)
        gen_abs_tok = id2id(UNK, self.word2id, self.id2word, enc_gen_abs_tok)

        # del ori_art, abstract, enc_gen_abs_tok, ext_art,src,data,gen_abst,enc_abs
        
        tensor_type = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        tensor_type_float = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        filt_ori_art_gen, gen_abs_tok = unzip(kernel_filter(enc_ori_art, gen_abs_tok))
        filt_ori_art, enc_ori_abst = unzip(kernel_filter(enc_ori_art, enc_ori_abst))
        net_in_ori = tensor_type(list(filt_ori_art)),tensor_type(list(enc_ori_abst))
        net_in_gen = tensor_type(list(filt_ori_art_gen)),tensor_type(list(gen_abs_tok))
        dis_out = self.dis(*net_in_ori)
        target = tensor_type_float([[0,1]]*dis_out.size()[0])
        tot_loss = self.loss(dis_out,target)
        dis_out = self.dis(*net_in_gen)
        tot_loss+=float(self.loss(dis_out,tensor_type_float([[1,0]]*dis_out.size()[0])))
        tot_loss.backward()
        self.opt.step()
        self.dis.zero_grad()

       

#####################################################
class GenAbstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        abstractor = CopySumm(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._max_len = max_len


#####################################################
class GenRLExtractor(object):
    def __init__(self, ext_dir, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        assert ext_meta['net'] == 'rnn-ext_abs_rl'
        ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, cuda))
        ext_ckpt = load_best_ckpt(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = agent.to(self._device)

    def __call__(self, raw_article_sents):
        self._net.eval()
        indices = self._net(raw_article_sents)
        return indices

####################################################