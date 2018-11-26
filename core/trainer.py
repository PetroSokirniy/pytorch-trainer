import gc 
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from hooks import HookList, Hook
from tqdm import tqdm_notebook as tqdm
from typing import Callable

class Trainer(object):
    def __init__(self, model:nn.Module, optimizer:torch.optim.Optimizer, loss_fn:Callable, hooks:HookList=None):
        self.model = model
        if hooks is None:
            hooks = HookList([Hook()])
        self.hooks = hooks
        self.stop = False
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def clean(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def fit(self, epochs:int, train, valid=None, batch_sz=32, batch_sz_valid=None, shuffle=True, sampler=None) -> None:

        self.hooks.on_fit_begin()

        for e in tqdm(range(epochs)):
            if self.stop:
                break

            self.hooks.on_epoch_begin(e)
            if train is not None:
                self.hooks.on_train_begin()
                self._train_loop(data=train, loss_fn=self.loss_fn, optimizer=self.optimizer, 
                    batch_sz=batch_sz, shuffle=shuffle, sampler=sampler)
                self.hooks.on_train_end()

            if valid is not None:
                self._validation(e, valid, batch_sz if batch_sz_valid is None else batch_sz_valid)

            self.hooks.on_epoch_end(e)

        self.hooks.on_fit_end()

    def _validation(self, epoch, valid, batch_sz) -> None:
        self.hooks.on_validation_begin()
        self._train_loop(data=valid, loss_fn=self.loss_fn, optimizer=None, batch_sz=batch_sz, shuffle=False)
        self.hooks.on_validation_end()

    def _train_loop(self, data, optimizer=None, loss_fn=None, batch_sz=32, shuffle=True, sampler=None) -> None:
        shuffle = shuffle and sampler is None
        self.clean()
        # for data in tqdm(DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=4),  leave=False):
        # tqdm has a problem with notebooks and a second inner loop
        for b, data in enumerate(DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=4)):
            if self.stop:
                break
            data = self.hooks.on_batch_begin(b, data) 
            data = self.model(data)
            data = self.hooks.on_output_data(data)
            if loss_fn is not None:
                loss = loss_fn(data)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                loss = None
            data['loss'] = loss
            
            
            self.hooks.on_batch_end(b, data)
            b += 1
        self.clean()