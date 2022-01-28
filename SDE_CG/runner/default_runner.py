#coding: utf-8
from time import time
from tqdm import tqdm
import os
import numpy as np
import pickle
import copy

import rdkit
from rdkit import Chem

import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add

#from confgf SDE_CG utils, dataset


class Runner(object):
    def __init__(self, train_set, val_set, test_set, model, optimizer, scheduler, config, loss_fn, device, checkpoint):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.device = device
        self.config = config
        self.batch_size = self.config.train.batch_size
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.best_loss = 100.0
        self.start_epoch = 0
        self.checkpoint = checkpoint

    def save(self, checkpoint, epoch, val_list={}):
        state = {
            **val_list,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config
        }
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        torch.save(state, checkpoint)

    def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        print("Load checkpoint from %s" % checkpoint)

        states = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(states['model'])
        try:
            self.best_loss = states['best_loss']
            self.start_epoch = states['cur_epoch'] + 1
        except:
            print('no best_loss and cur_epoch!')
        else:
            pass

        if load_scheduler:
            self.scheduler.load_state_dict(states['scheduler'])

        if load_optimizer:
            self.optimizer.load_state_dict(states['optimizer'])
            if self.device == 'cuda':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

    def evaluate(self, split, verbose=0):
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either val, or test.')
        if split == 'val':
            dataloader = DataLoader(self.val_set, batch_size=self.config.train.batch_size, \
                                    shuffle=False)
            model = self.model
            model.eval()
            eval_start = time()
            eval_losses = []
            for batch in dataloader:
                batch = batch.to(self.device)
                scores, targets, edge_sigmas = model(batch)
                loss = model(scores, targets, edge_sigmas)
                eval_losses.append(loss.item())
            average_loss = sum(eval_losses) / len(eval_losses)

            if verbose:
                print('Evaluate %s Loss: %.5f | Time: %.5f' % (split, average_loss, time() - eval_start))
            return average_loss

    def train(self, verbose=1):
        train_start = time()
        num_epochs = self.config.train.epochs
        dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                shuffle=self.config.train.shuffle,drop_last=False)
        model = self.model
        loss_fn = self.loss_fn
        train_losses = []
        val_losses = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        print('start training...')

        for epoch in range(num_epochs):
            model.train()
            epoch_start = time()
            batch_losses = []
            batch_cnt = 0
            for batch in dataloader:
                batch_cnt += 1
            batch = batch.to(self.device)
            scores, targets, edge_sigmas = model(batch)
            loss = loss_fn(scores, targets, edge_sigmas)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())

            if batch_cnt % self.config.train.log_interval == 0 or (epoch == 0 and batch_cnt <= 10):
                print('Epoch: %d | Step: %d | loss: %.5f | Lr: %.5f' % \
                      (epoch + start_epoch, batch_cnt, batch_losses[-1], self.optimizer.param_groups[0]['lr']))
            average_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(average_loss)
            if verbose:
                print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (
                epoch + start_epoch, average_loss, time() - epoch_start))
            average_eval_loss = self.evaluate('val', verbose=1)
            val_losses.append(average_eval_loss)
            if self.config.train.scheduler.type == "plateau":
                self.scheduler.step(average_eval_loss)
            else:
                self.scheduler.step()
            if val_losses[-1] < best_loss:
                if self.config.train.save:
                    val_list = {
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': best_loss,
                    }
                    self.save(self.checkpoint, epoch + start_epoch, val_list)
        self.beat_loss = self.best_loss
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))


