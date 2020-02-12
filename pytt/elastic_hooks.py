from collections import defaultdict

from .core import *

import torch
import pandas as pd
import numpy as np


class TensorsToDevice(ElasticHook):
    def __init__(self, device='cuda'):
        super().__init__(self.__class__.__name__)
        self.device = device

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(self.device)
        return batch_data, step_data

class BatchDel(ElasticHook):
    def __init__(self, keys):
        super().__init__(self.__class__.__name__)
        self.keys = keys

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        for key in self.keys:
            if key in batch_data.keys():
                del batch_data[key]

class MixUpHook(ElasticHook):
    def __init__(self, key, alpha=1.0, steps=['train']):
        super().__init__(self.__class__.__name__)
        self.key = key
        self.alpha = alpha
        self.steps = steps
        self.run = False

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        self.run = step in self.steps

    def mixup_data(self, x):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]

        return mixed_x, index, lam

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        if self.run:
            mixed_x, idx, lam = self.mixup_data(batch_data[self.key])
            batch_data[self.key] = mixed_x
            batch_data['idx'] = idx
            batch_data['lam'] = lam

class StatTracker(ElasticHook):
    def __init__(self, mapping:Dict[str,List[str]],  path:str=None, save_key='stats'):
        super().__init__(self.__class__.__name__)
        self.mapping = mapping
        self.path = path
        self.save_key = save_key
        self.stats = []

    def to_df(self):
        return pd.DataFrame(self.stats)

    def load_previous(self):
        df = pd.read_csv(self.path)
        self.stats = df.to_dict('records')

    def save(self, path=None):
        if path is None:
            path = self.path

        if path is not None:
            self.to_df().to_csv(path)

    def on_fit_begin(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        trainer_data[self.save_key] = self.stats 

    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        row = {}
        for k, stats in self.mapping.items():
            for stat in stats:
                row[f'{k}_{stat}'] = trainer_data[k][stat]
        row['epoch'] = e
        self.stats.append(row)
        self.save()


class PrintHook(ElasticHook):
    def __init__(self, keys=None, key_replace={}, print_hist=True, stats_per_line=8, input_key='stats'):
        super().__init__(self.__class__.__name__)
        self.keys = keys
        self.key_replace = key_replace
        self.print_hist = print_hist
        self.stats_per_line = stats_per_line
        self.input_key = input_key 

    def _replace_key(self, key):
        for k,v in self.key_replace.items():
            key = key.replace(k,v)
        return key

    def on_fit_begin(self, epochs, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        if self.print_hist:
            self.print_hist = False
            if self.input_key in trainer_data and len(trainer_data[self.input_key]) != 0:
                self.print(trainer_data[self.input_key][-1], self.keys, self.stats_per_line)

    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        self.print(trainer_data[self.input_key][-1], self.keys, self.stats_per_line)

    def print(self, stats:Dict[str,Any], keys=None, stats_per_line=None):
        if keys is None: keys = self.keys
        if keys is None: keys = [k for k in stats.keys() if 'int' in str(type(k)) or 'float' in str(type(k))]

        if stats_per_line is None: stats_per_line = self.stats_per_line

        i = 0
        p_str  = ''
        for key in keys:
            _key = self._replace_key(key)
            if 'float' in str(type(stats[key])):
                p_str += f'{_key}:{stats[key]:0.5f} '
            else:
                p_str += f'{_key}:{stats[key]} '
            i += 1
            if i > stats_per_line:
                p_str += '\n'
                i = 0
        print(p_str)
        sys.stdout.flush()

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