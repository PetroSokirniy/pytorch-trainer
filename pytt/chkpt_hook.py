
from typing import Any, Dict

from .core import *

import torch
import numpy as np


class CheckPointHook(ElasticHook):
    def __init__(self, path:str, save_last=True, save_chkpt=False, load_on_fit=True):
        super().__init__(self.__class__.__name__)
        self.path = path
        self.save_last = save_last
        self.save_chkpt = save_chkpt
        self.load_on_fit = load_on_fit
        self.count = 0

    def save_model(self, model, path:str=None, num=None):
        if path is None: path = self.path
        if num is None:
            num = ''
        else:
            num = f'_{num}'
        torch.save(model.state_dict(), path + num)

    def load_model(self, model, path:str):
        if path is None: path = self.path
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        return model

    def on_fit_begin(self, epochs, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        if self.load_on_fit:
            trainer_data[MODEL] = self.load_model(trainer_data[MODEL])

    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        if self.save_last:
            self.save_model(trainer_data[MODEL])
        if self. save_chkpt:
            self.save_model(trainer_data[MODEL], num=counter)
            counter += 1


class SaveBestHook(CheckPointHook):
    up   = 'up'
    down = 'down'
    zero ='zero'

    def __init__(self, path:str, stat_key:str, stat:str,save_last=True, save_chkpt=False, load_on_fit=True, direction:str=up):
        super().__init__(path, save_last, save_chkpt, load_on_fit)
        self.stat_key = stat_key
        self.stat = stat
        self.direction = direction
        self.best = None
        self.bast_path = f'{path}_b'

    def on_fit_begin(self, epochs, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        super().on_fit_begin(epochs, trainer_data)
        if self.stat_key not in trainer_data:
            return
        
        if len(trainer_data[self.stat_key]) == 0:
            return

        stats = [s[self.stat] for s in trainer_data[self.stat_key]]

        if self.direction == self.up:
            self.best = np.max(stats)
        elif self.direction == self.down:
            self.best = np.min(stats)
        elif self.direction == self.zero:
            idx = np.abs(stats).argmin()
            self.best = arr[idx]


    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        super().on_epoch_end(e, trainer_data)
        current =  trainer_data[self.stat_key][-1][self.stat]

        if self.best is None:
            self.best = current
            self.save_model(trainer_data[MODEL], self.bast_path)
        elif self.direction == self.up and current > self.best:
            self.best = current
            self.save_model(trainer_data[MODEL], self.bast_path)
        elif self.direction == self.down and current < self.best:
            self.best = current
            self.save_model(trainer_data[MODEL], self.bast_path)
        elif self.direction == self.zero and abs(current) < abs(self.best):
            self.best = current
            self.save_model(trainer_data[MODEL], self.bast_path)