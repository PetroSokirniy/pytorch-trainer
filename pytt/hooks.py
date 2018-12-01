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
        self.device = device

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        return {k:v.float().to(self.device) for k,v in data.items()}

class StatTracker(Hook):
    def __init__(self, collect_stats:Dict[str,Callable[[Dict], float]], prefix_train:str='t_', prefix_valid:str='v_', 
            print:List[str]=None, save_path:str=None, model_n:int=None, best:str=None, best_func:Callable=np.max, check_prev=True):
        self.model_n:int = model_n
        self.collect_stats = collect_stats
        self.stats = defaultdict(list)
        self.prefix_train = prefix_train
        self.prefix_valid = prefix_valid
        self.save_path = save_path
        self.print = print
        self.best = best
        self.best_func = best_func
        self.check_prev = check_prev
        if check_prev:
            self.from_csv(save_path)
        
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

    def on_epoch_end(self, e:int) -> None:
        for stat in self.stat_runner:
            self.stats[stat].append(np.mean(self.stat_runner[stat]))

        if self.save_path is not None:
            pd.DataFrame(data=self.stats).to_csv(self.save_path)

        if self.print is None or len(self.print) != 0:
            self._print()

    def _print(self) -> None:
        print_dict = self.get_last_stats()
        epoch = int(print_dict['epoch'])
        if self.model_n is not None:
            p_str = f'Model:{self.model_n} Epoch:{epoch} '
        else:
            p_str = f'Epoch:{epoch} '

        for key in self.collect_stats:
            if self.print is None or key in self.print and key != 'epoch':
                if self.prefix_train + key in print_dict:
                    p_str += f'{self.prefix_train + key}:{print_dict[self.prefix_train + key]:0.3f} '
                if self.prefix_valid + key in print_dict:
                    p_str += f'{self.prefix_valid + key}:{print_dict[self.prefix_valid + key]:0.3f} '

        if self.best is not None:
            p_str += f' Best:{self.best_func(self.stats[self.best]):0.3f}'

        print(p_str)
        sys.stdout.flush()

    def from_csv(self, path:str) -> None:
        if path is not None and os.path.exists(path):
            df = pd.read_csv(path)
            dict_:Dict[str,Dict[int,float]] = df.to_dict()
            self.stats = {k:list(v.values()) for k,v in dict_.items() if k != 'Unnamed: 0'}

    def on_batch_end(self, b:int, data:Dict[str, Any]) -> None:
        for stat in self.collect_stats:
            self.stat_runner[self.prefix + stat].append(self.collect_stats[stat](data))

    def get_last_stats(self) -> Dict[str, float]:
        return {k:self.stats[k][-1] for k in self.stats.keys()}

    def restart(self):
        self.stats = defaultdict(list)

class CheckPointHook(Hook):
    def __init__(self, model:nn.Module, path:str, load=True):
        self.model = model
        self.path_chk = path
        self.load = load

    def save_model(self, path:str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path:str):
        self.model.load_state_dict(torch.load(path))

    def on_train_begin(self):
        if os.path.exists(self.path_chk) and self.load:
            self.load_model(self.path_chk)

    def on_epoch_end(self, e:int) -> None:
        self.save_model(self.path_chk)

    def delete_chk_pt(self):
        if self.path_chk is not None and os.path.exists(self.path_chk):
            os.remove(self.path_chk)

class SaveBestHook(CheckPointHook):
    up   = 'up'
    down = 'down'
    zero ='zero'

    def __init__(self, model:nn.Module, tracker:StatTracker, key:str, path:str, direction:str=up, start:float=None, do_chk:bool=True):
        super().__init__(model, path + '_chk_pt')
        self.model = model
        self.tracker = tracker
        self.key = key
        self.path_b = path + '_b'
        self.direction = direction
        self.best = start
        if self.tracker.stats[key] != [] and self.best is None:
            if self.direction == self.up:
                self.best = np.max(self.tracker.stats[key])
            elif self.direction == self.down:
                self.best = np.min(self.tracker.stats[key])
            elif self.direction == self.zero:
                arr = self.tracker.stats[key]
                idx = (np.abs(arr)).argmin()
                self.best = arr[idx]


    def on_epoch_end(self, e:int) -> None:
        super().on_epoch_end(e)
        current = self.tracker.get_last_stats()[self.key]

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
        self.delete_chk_pt()
        self.load = False
        if self.path_b is not None and os.path.exists(self.path_b):
            os.remove(self.path_b)


class EndEarlyHook(Hook):
    def __init__(self, tracker:StatTracker, trainer, key:str='v_acc', wait:int=10, best_func:Callable=np.max):
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
        self.optimizer = optimizer
        self.lr = lr
        self.epoch = epoch
    
    def on_epoch_end(self, e:int)-> None:
        if e >= self.epoch:
            adjust_learning_rate(self.optimizer, self.lr)
