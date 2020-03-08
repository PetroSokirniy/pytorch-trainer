import gc
import sys
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook as tqdm

from .base_hook import Hook, HookList

class Trainer(object):
    def __init__(self, model:nn.Module, optimizer:torch.optim.Optimizer, loss_fn:Callable, hooks:HookList=None):
        self.model = model
        if hooks is None:
            hooks = HookList([Hook()])
        self.hooks = hooks
        self.hooks.set_trainer(self)
        self.stop = False
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def clean(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def fit(self, epochs:int, train, valid=None, batch_sz=32, batch_sz_valid=None, shuffle=True, sampler=None, num_workers=4) -> None:

        self.hooks.on_fit_begin()

        for e in tqdm(range(epochs)):
            if self.stop:
                break

            self.hooks.on_epoch_begin(e)
            if train is not None:
                self.hooks.on_train_begin()
                self._loop(data=train, loss_fn=self.loss_fn, optimizer=self.optimizer, 
                    batch_sz=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
                self.hooks.on_train_end()

            if valid is not None:
                self.hooks.on_validation_begin()
                self._loop(valid, loss_fn=self.loss_fn, batch_sz=batch_sz if batch_sz_valid is None else batch_sz_valid, 
                    shuffle=False, num_workers=num_workers)
                self.hooks.on_validation_end()

            self.hooks.on_epoch_end(e)

        self.hooks.on_fit_end()

    def _loop(self, data, optimizer=None, loss_fn=None, batch_sz=32, shuffle=True, sampler=None, num_workers=4) -> None:
        shuffle = shuffle and sampler is None
        self.clean()
        # for data in tqdm(DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=4),  leave=False):
        # tqdm has a problem with notebooks and a second inner loop
        for b, batch_data in enumerate(DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=num_workers)):
            if self.stop:
                break
            self._batch(b, batch_data, loss_fn, optimizer)
        self.clean()

    def run(self, data, batch_sz=32, shuffle=True, sampler=None, num_workers=4):
        self._loop(data, batch_sz=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=num_workers)

    def _batch(self, b, data, loss_fn, optimizer):
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

