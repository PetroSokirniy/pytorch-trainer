from .core import Hook, HookList
from .utils import adjust_learning_rate
from .core import Trainer

import os
import sys
import time
import itertools

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import torch
from torch import nn
import torchvision.utils as vutils

from typing import List, Tuple, Any, Dict, Callable
from collections import defaultdict

import asyncio

class DataPrep(Hook):
    def __init__(self, device='cuda'):
        super().__init__(self.__class__.__name__)
        self.device = device

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        return {k:v.float().to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}


class OptWalker(Hook):
    def __init__(self, opti, lr):
        super().__init__(self.__class__.__name__)
        self.opti = opti
        self.lr = lr
        self.i = 0
        for g in self.opti.param_groups:
            g['lr'] = 0

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        self.opti.param_groups[self.i - 1]['lr'] = 0
        self.opti.param_groups[self.i]['lr'] = self.lr
        self.i = (self.i + 1) % len(self.opti.param_groups)

        return data

class ReqGradHook(Hook):
    def __init__(self, keys=['img']):
        super().__init__(self.__class__.__name__)
        self.keys = keys

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        for k in self.keys:
            data[k].requires_grad = True
        return data

class TransposeHook(Hook):
    def __init__(self, keys=['img', 'mask'], p = 0.5, allow_key='stage'):
        super().__init__(self.__class__.__name__)
        self.keys = keys
        self.p = p
        self.allow_key = allow_key

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        if data[self.allow_key][0] == 'train' and np.random.rand() >= self.p:
            for k in self.keys:
                data[k] = data[k].transpose(-1,-2)
        return data

class MixUpHook(Hook):
    def __init__(self, key, alpha=1.0, device='cuda'):
        super().__init__(self.__class__.__name__)
        self.key = key
        self.alpha = alpha
        self.device = device
        self.is_train = False

    def on_train_begin(self):
        self.is_train = True
    
    def on_validation_begin(self):
        self.is_train = False

    def mixup_data(self, x):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]

        return mixed_x, index, lam

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        if self.is_train:
            mixed_x, idx, lam = self.mixup_data(data[self.key])
            data[self.key] = mixed_x
            data['idx'] = idx
            data['lam'] = lam
        return data

def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped

@fire_and_forget
def save_img(imgs, paths, save_path):
    for img, path in zip(imgs, paths):
        img = (img.transpose(1,2,0) * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        if save_path is not None:
            name = path.split('/')[-1]
            path = save_path + name
            cv2.imwrite(path, img * 255)

class FGSMHook(Hook):
    def __init__(self, key='img', epsilon=0.05, save_path=None):
        super().__init__(self.__class__.__name__)
        self.key = key
        self.epsilon = epsilon
        self.save_path = save_path
        # self.pool = Pool(processes=16)

    def on_batch_end(self, b, data):
        fgsm_img = self.fgsm_attack(data['img'], self.epsilon, data['img'].grad.data)
        fgsm_img = data['img'].detach().cpu().numpy()
        save_img(fgsm_img, data['path'], self.save_path) 

        return data

    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

class SaveImgHook(Hook):
    def __init__(self, keys, path, im_name, b):
        super().__init__(self.__class__.__name__)
        self.keys = keys
        self.path = path
        self.im_name = im_name
        self.b = b
        self.epoch = 0
        
    def on_batch_end(self, b:int, data:dict) -> None: 
        if b == self.b:
            i_len = int(data[self.keys[0]].shape[0] / len(self.keys))
            imgs = []
            for i in range(i_len):
                for k in self.keys:
                    imgs.append(data[k][i])
            vutils.save_image(imgs, f'{self.path}/{self.im_name}_{self.epoch}.png')
            self.epoch += 1
        return data    

class DelTensor(Hook):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    def on_batch_end(self, b:int, data:Dict[str,Any]) -> None:
        for v in data.values():
            if isinstance(v, torch.Tensor):
                del v
        return data

class GetDataHook(Hook):
    def __init__(self, keys):
        super().__init__(self.__class__.__name__)
        self.keys = keys
        self.data = defaultdict(list)

    def on_output_data(self, data:Dict[str,Any]) -> Dict[str,Any]: 
        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                value = data[key].cpu().detach().numpy()
            else:
                value = data[key]
            self.data[key].append(value)
        return data

class SplitDataHook(GetDataHook):
    def __init__(self, keys, on_train=False, on_valid=True):
        super().__init__(keys)
        self.keys = keys
        self.train = defaultdict(list)
        self.valid = defaultdict(list)
        self.data = None

    def on_train_begin(self) -> None: 
        self.data = self.train

    def on_train_end(self) -> None: 
        self.data = None

    def on_validation_begin(self) -> None: 
        self.data = self.valid

    def on_validation_end(self) -> None:
        self.data = None

class TimerHook(Hook):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.train_time = []
        self.valid_time = []
        self.batch_time = None
        self.train_batch_time = []
        self.valid_batch_time = []
        

    def on_train_begin(self) -> None: 
        self.train_begin = time.time()
        self.batch_time = self.train_batch_time

    def on_train_end(self) -> None: 
        self.train_end = time.time()
        self.train_time.append(time.time() - self.train_begin)

    def on_validation_begin(self) -> None: 
        self.valid_begin = time.time()
        self.batch_time = self.valid_batch_time

    def on_validation_end(self) -> None: 
        self.valid_time.append(time.time() - self.valid_begin)

    def on_batch_begin(self, b:int) -> None:
        self.batch_begin = time.time()

    def on_batch_end(self, b:int, data) -> None: 
        self.batch_time.append(time.time() - self.batch_begin)
        return data

class StatTracker(Hook):
    def __init__(self, collect_stats:Dict[str,Callable[[Dict], float]],  save_csv:str=None, 
            prefix_train:str='t_', prefix_valid:str='v_', model_n:int=0):
        super().__init__(self.__class__.__name__)
        self.model_n:int = model_n
        self.collect_stats = collect_stats
        self.stats = defaultdict(list)
        self.prefix_train = prefix_train
        self.prefix_valid = prefix_valid
        self.save_csv = save_csv
        self._restart = False
        
    def on_fit_begin(self):
        if not self._restart and self.save_csv is not None:
            self.from_csv(self.save_csv)
    
    def on_train_begin(self) -> None:
        self.prefix = self.prefix_train
    
    def on_validation_begin(self) -> None:
        self.prefix = self.prefix_valid

    def on_epoch_begin(self, e: int) -> None:
        if self.stats['epoch'] == []:
            self.stats['epoch'].append(0)
        else:
            self.stats['epoch'].append(self.stats['epoch'][-1] + 1)
        self.stats['model'].append(self.model_n)

        for stat in self.collect_stats:
            self.collect_stats[stat].on_epoch_begin()

    def on_epoch_end(self, e:int) -> None:
        for stat in self.collect_stats:
            self.stats[stat].append(self.collect_stats[stat].get_result())
    
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
            if self.prefix in stat: 
                self.collect_stats[stat].on_batch_result(data)
        return data

    def get_last_stats(self) -> Dict[str, float]:
        return {k:self.stats[k][-1] for k in self.stats.keys()}

    def __getitem__(self, idx:int):
        return {k:self.stats[k][idx] for k in self.stats.keys()}

    def __len__(self):
        all_lens = [len(self.stats[k]) for k in self.stats.keys()]
        return min(all_lens) if len(all_lens) > 0 else 0

    def restart(self):
        self._restart = True
        self.stats = defaultdict(list)
        return self

class ValidTrainer(Hook):
    def __init__(self, tracker:StatTracker, train, valid, tolerance:float=0.05, max_epochs:int=None, t_key='t_loss', v_key='v_loss', params:dict={}):
        super().__init__(self.__class__.__name__)
        self.tracker = tracker
        self.train = train
        self.valid = valid
        self.tolerance = tolerance
        self.max_epochs = max_epochs
        self.t_key = t_key
        self.v_key = v_key
        self.params = params
        self.start = True

    def on_fit_end(self):
        if not self.start:
            return 
        self.start = False

        target = self.tracker.get_last_stats[self.t_key]
        value = self.tracker.get_last_stats[self.v_key]

        while target < value - self.tolerance :
            self.trainer.fit(1, self.train + self.valid, self.valid, **self.params)
            value = self.tracker.get_last_stats[self.v_key]

class TTAHook(GetDataHook):
    def __init__(self, data_set, transforms, batch_size=32, keys=[]):
        super().__init__(keys)
        self.data_set = data_set
        self.transforms = transforms
        self.batch_size = batch_size 
        self.keys = keys
        self.datas = []

    def run(self):
        for t in tqdm(self.transforms):
            self.data_set.tsfrm = t
            self.data = defaultdict(list)
            self.trainer.run(self.data_set, batch_sz=self.batch_size, shuffle=False)
            self.datas.append(self.data)

    def on_fit_end(self):
        self.run()
            
    def __getitem__(self, key):
        return self.get_results(key)
    
    def get_results(self, key):
        return np.column_stack([np.concatenate(d[key]) for d in self.datas])

class PrintHook(Hook):
    def __init__(self, stat_tracker:StatTracker=None, keys=['model','epoch', 't_loss', 'v_loss', 't_acc', 'v_acc'],
          custom:Dict[str,Tuple[str,Callable]]={}, print_hist=True, stats_per_line=8):
        super().__init__(self.__class__.__name__)
        self.stat_tracker = stat_tracker
        self.keys = keys
        self.custom = custom 
        self.print_hist = print_hist
        self.stats_per_line = stats_per_line
    
    def on_fit_begin(self):
        if self.stat_tracker is None:
           self.stat_tracker = self.hook_list.dict_hooks[StatTracker.__name__]

        if self.print_hist:
            for idx in range(len(self.stat_tracker)):
                self.print(self.stat_tracker[idx])

    def on_epoch_end(self, e:int):
        self.print(self.stat_tracker.get_last_stats(), self.keys)

    def print(self, stats:Dict[str,Any], keys=None, stats_per_line=None):
        if keys is None and self.keys is not None: keys = self.keys
        if keys is None: keys = [k for k in stats.keys() if 'int' in str(type(k)) or 'float' in str(type(k))]
        if stats_per_line is None: stats_per_line = self.stats_per_line
        i = 0
        p_str  = ''
        for key in keys:
            if 'float' in str(type(stats[key])):
                p_str += f'{key}:{stats[key]:0.3f} '
            else:
                p_str += f'{key}:{stats[key]} '
            i += 1
            if i > stats_per_line:
                p_str += '\n\t'
                i = 0

        for key,value in self.custom.items():
            p_str += f'{key}:{value[1](self.stat_tracker.stats[value[0]]):0.3f} ' 
            i += 1
            if i > stats_per_line:
                p_str += '\n\t'
                i = 0
        print(p_str)
        sys.stdout.flush()

class CheckPointHook(Hook):
    def __init__(self, model:nn.Module, path:str):
        super().__init__(self.__class__.__name__)
        self.model = model
        self.path = path
        self.count = 0

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

    def checkpoint(self):
        self.save_model(self.path + f'ckpt_{self.count}')
        self.count += 1


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
        super().restart()
        if self.path_b is not None and os.path.exists(self.path_b):
            os.remove(self.path_b)
        return self



class EndEarlyHook(Hook):
    def __init__(self, key:str='v_acc', wait:int=10, best_func:Callable=np.max, stat_tracker:StatTracker=None, trainer:Trainer=None):
        super().__init__(self.__class__.__name__)
        self.stat_tracker = stat_tracker
        self.trainer = trainer
        self.key = key
        self.wait = wait
        self.best_func = best_func

    def on_train_begin(self):
        if self.stat_tracker is None:
           self.stat_tracker = self.hook_list.dict_hooks[StatTracker.__name__]

    def on_epoch_end(self, e:int) -> None:
        best = self.best_func(self.stat_tracker.stats[self.key])
        last = self.best_func(self.stat_tracker.stats[self.key][-self.wait:])
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


class EpochSchedulerHook(Hook):
    def __init__(self, optimizer):
        super().__init__(self.__class__.__name__)
        self.optimizer = optimizer
    
    def on_epoch_end(self, e:int)-> None:
        self.optimizer.step()

class StatHook(Hook):
    def get_stat_batch(self): pass

    def get_stat_epoch(self): pass

class FuncStatHook(StatHook):
    def __init__(self, key, function, agg_function):
        self.key = key
        self.function = function
        self.agg_function = agg_function
        self.current_epoch = []
        self.history = []

    def on_batch_end_data(self, data):
        self.current_epoch.append(self.function(data[self.key]))
        return data
    
    def on_batch_end(self, b, data):
        self.history.append(self.current_epoch)
        self.current_epoch = []
        return data

    def get_stat_batch(self):
        return self.current_epoch[-1]

    def get_stat_epoch(self):
        return self.agg_function(self.current_epoch)


class LossStatHook(FuncStatHook):
    def __init__(self, key='loss'):
        super(self).__init__(self, key, np.mean, np.mean)

class AccStatHook(FuncStatHook):
    def __init__(self, key='loss'):
        super(self).__init__(self, key, np.mean, np.mean)


