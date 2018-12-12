from .core import Hook, HookList
from .utils import adjust_learning_rate

import sys
import os

import numpy as np
import pandas as pd

import torch
from torch import nn

from typing import List, Tuple, Any, Dict, Callable
from collections import defaultdict


class DataPrep(Hook):
    def __init__(self, device='cuda'):
        super().__init__(self.__class__.__name__)
        self.device = device

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        return {k:v.float().to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}

class DelTensor(Hook):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    def on_batch_end(self, b:int, data:Dict[str,Any]) -> None:
        for v in data.values():
            if isinstance(v, torch.Tensor):
                del v

class StatTracker(Hook):
    def __init__(self, collect_stats:Dict[str,Callable[[Dict], float]],  save_csv:str=None, prefix_train:str='t_', prefix_valid:str='v_', model_n:int=0):
        super().__init__(self.__class__.__name__)
        self.model_n:int = model_n
        self.collect_stats = collect_stats
        self.stats = defaultdict(list)
        self.prefix_train = prefix_train
        self.prefix_valid = prefix_valid
        self.save_csv = save_csv
        self._restart = False
        
    def on_fit_begin(self):
        if not self._restart:
            self.from_csv(self.save_csv)
    
    def on_train_begin(self) -> None:
        self.prefix = self.prefix_train
    
    def on_validation_begin(self) -> None:
        self.prefix = self.prefix_valid

    def on_epoch_begin(self, e: int) -> None:
        self.stat_runner:Dict = defaultdict(list)
        if self.stats['epoch'] == []:
            self.stats['epoch'].append(0)
        else:
            self.stats['epoch'].append(self.stats['epoch'][-1] + 1)
        self.stats['model'].append(self.model_n)

    def on_epoch_end(self, e:int) -> None:
        for stat in self.stat_runner:
            self.stats[stat].append(np.mean(self.stat_runner[stat]))
    
        if self.save_csv is not None:
            pd.DataFrame(data=self.stats).to_csv(self.save_csv, index=False)

    def from_csv(self, path:str=None) -> None:
        if path is None: path = self.save_csv
        if os.path.exists(path):
            df = pd.read_csv(path)
            dict_:Dict[str,Dict[int,float]] = df.to_dict()
            self.stats = {k:list(v.values()) for k,v in dict_.items()}

    def on_batch_end(self, b:int, data:Dict[str, Any]) -> None:
        for stat in self.collect_stats:
            self.stat_runner[self.prefix + stat].append(self.collect_stats[stat](data))

    def get_last_stats(self) -> Dict[str, float]:
        return {k:self.stats[k][-1] for k in self.stats.keys()}

    def restart(self):
        self._restart = True
        self.stats = defaultdict(list)
        return self

class PrintHook(Hook):
    def __init__(self, stat_tracker:StatTracker=None, keys=['model','epoch', 't_loss', 'v_loss', 't_acc', 'v_acc'],
          custom:Dict[str,Tuple[str,Callable]]={}):
        super().__init__(self.__class__.__name__)
        self.stat_tracker = stat_tracker
        self.keys = keys
        self.custom = custom 
    
    def on_train_begin(self):
        if self.stat_tracker is None:
           self.stat_tracker = self.hook_list.dict_hooks[StatTracker.__name__]

    def on_epoch_end(self, e:int):
        self.print(self.stat_tracker.get_last_stats(), self.keys)

    def print(self, stats:Dict[str,Any], keys=None):
        if keys is None and self.keys is not None: keys = self.keys
        if keys is None: keys = [k for k in stats.keys() if isinstance(k, int) or isinstance(k, float)]
        p_str  = ''
        for key in keys:
            if 'float' in str(type(stats[key])):
                p_str += f'{key}:{stats[key]:0.3f} '
            else:
                p_str += f'{key}:{stats[key]} '

        for key,value in self.custom.items():
            p_str += f'{key}:{value[1](self.stat_tracker.stats[value[0]]):0.3f} ' 
        print(p_str)
        sys.stdout.flush()

class CheckPointHook(Hook):
    def __init__(self, model:nn.Module, path:str):
        super().__init__(self.__class__.__name__)
        self.model = model
        self.path = path

    def save_model(self, path:str=None):
        if path is None: path = self.path
        torch.save(self.model.state_dict(), path)

    def load_model(self, path:str):
        if path is None: path = self.path
        self.model.load_state_dict(torch.load(path))

    def on_train_begin(self):
        if os.path.exists(self.path):
            self.load_model(self.path)

    def on_epoch_end(self, e:int) -> None:
        self.save_model(self.path)

    def restart(self):
        if self.path is not None and os.path.exists(self.path):
            os.remove(self.path)
        return self

class SaveBestHook(CheckPointHook):
    up   = 'up'
    down = 'down'
    zero ='zero'

    def __init__(self, model:nn.Module, path:str, key:str='v_acc',  stat_tracker:StatTracker=None, direction:str=up):
        super().__init__(model, path)
        self.name = self.__class__.__name__
        self.model = model
        self.stat_tracker = stat_tracker
        self.key = key
        self.path_b = path + '_b'
        self.direction = direction
        self.best = None

    def on_train_begin(self):
        if self.stat_tracker is None:
           self.stat_tracker = self.hook_list.dict_hooks[StatTracker.__name__]

        if self.stat_tracker.stats[self.key] != [] and self.best is None:
            if self.direction == self.up:
                self.best = np.max(self.stat_tracker.stats[self.key])
            elif self.direction == self.down:
                self.best = np.min(self.stat_tracker.stats[self.key])
            elif self.direction == self.zero:
                arr = self.stat_tracker.stats[self.key]
                idx = (np.abs(arr)).argmin()
                self.best = arr[idx]

    def on_epoch_end(self, e:int) -> None:
        super().on_epoch_end(e)
        current = self.stat_tracker.get_last_stats()[self.key]

        if self.best is None:
            self.best = current
            self.save_model(self.path_b)
        elif self.direction == self.up and current > self.best:
            self.best = current
            self.save_model(self.path_b)
        elif self.direction == self.down and current < self.best:
            self.best = current
            self.save_model(self.path_b)
        elif self.direction == self.zero and abs(current) < abs(self.best):
            self.best = current
            self.save_model(self.path_b)

    def restart(self):
        s = super().restart()
        if self.path_b is not None and os.path.exists(self.path_b):
            os.remove(self.path_b)
        return self


class EndEarlyHook(Hook):
    def __init__(self, tracker:StatTracker, trainer, key:str='v_acc', wait:int=10, best_func:Callable=np.max):
        super().__init__(self.__class__.__name__)
        self.tracker = tracker
        self.trainer = trainer
        self.key = key
        self.wait = wait
        self.best_func = best_func

    def on_epoch_end(self, e:int) -> None:
        best = self.best_func(self.tracker.stats[self.key])
        last = self.best_func(self.tracker.stats[self.key][-self.wait:])
        if (best >= self.best_func(last)).sum() > 0:
            self.trainer.stop = True

class LRChangeEpochHook(Hook):
    def __init__(self, optimizer, lr, epoch):
        super().__init__(self.__class__.__name__)
        self.optimizer = optimizer
        self.lr = lr
        self.epoch = epoch
    
    def on_epoch_end(self, e:int)-> None:
        if e >= self.epoch:
            adjust_learning_rate(self.optimizer, self.lr)
