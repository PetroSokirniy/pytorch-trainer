import gc
import sys
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm as tqdm
from fastprogress import master_bar, progress_bar

from fastprogress import force_console_behavior
import fastprogress
master_bar, progress_bar = force_console_behavior()

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
        for i in range(20):
            torch.cuda.empty_cache()
            gc.collect()

    def fit(self, epochs:int, train, valid=None, batch_sz=None, batch_sz_valid=None, shuffle=True, 
            sampler=None, batch_sampler=None, num_workers=4, accumulator_steps=1, clean=False) -> None:

        self.hooks.on_fit_begin()
        mb = master_bar(range(epochs), total_time=True)
        for e in mb:
            if self.stop:
                break

            if self.optimizer is not None:
                self.optimizer.zero_grad()

            if clean:
                self.clean()

            self.hooks.on_epoch_begin(e)
            if train is not None:
                self.hooks.on_train_begin()
                self._loop(self.model, train, self.hooks, loss_fn=self.loss_fn, optimizer=self.optimizer, 
                    batch_sz=batch_sz, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                    accumulator_steps=accumulator_steps, mb=mb)
                self.hooks.on_train_end()

            if clean:
                self.clean()

            if valid is not None:
                with torch.no_grad():
                    self.hooks.on_validation_begin()
                    self._loop(self.model, valid, self.hooks, loss_fn=self.loss_fn, batch_sz=batch_sz if batch_sz_valid is None else batch_sz_valid, 
                        shuffle=False, num_workers=num_workers, mb=mb)
                    self.hooks.on_validation_end()
            self.hooks.on_epoch_end(e)

        self.hooks.on_fit_end()


    def _loop(self, model, data, hooks, optimizer=None, loss_fn=None, batch_sz=32, shuffle=True,
            sampler=None, batch_sampler=None, num_workers=4, accumulator_steps=1, mb=None) -> None:
        shuffle = shuffle and sampler is None
        # self.clean()
        b = 0
        # tqdm has a problem with notebooks and a second inner loop
        # for b, batch_data in enumerate(DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=num_workers)):
        dataloder = DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, 
            batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True,  drop_last=False)

        for batch_data in progress_bar(dataloder, leave=False, parent=mb):
            if self.stop:
                break
            batch_data = hooks.on_batch_begin(b, batch_data) 
            batch_data = model(batch_data)
            batch_data = hooks.on_output_data(batch_data)
            
            if loss_fn is not None:
                loss = loss_fn(batch_data, optimizer, b)

            hooks.on_batch_end(b, batch_data)
            b += 1
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        # self.clean()

    def run(self, data, batch_sz=32, shuffle=True, sampler=None, num_workers=4):
        self._loop(self.model, data, self.hooks, batch_sz=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=num_workers)