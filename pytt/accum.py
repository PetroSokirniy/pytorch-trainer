from typing import List, Any, Dict, Callable
from collections import namedtuple

from .core import ElasticHook
from .utils import identity

import numpy as np


class BasicAccumulator(ElasticHook):
    def __init__(self, batch_key:str, step_key:str, acc_func:Callable=identity, red_func:Callable=np.mean):
        super().__init__(self.__class__.__name__)
        self.batch_key = batch_key
        self.step_key = step_key
        self.temp_key = f'{batch_key}_{step_key}'
        self.acc_func = acc_func
        self.red_func = red_func

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        step_data[self.temp_key] = []

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        step_data[self.temp_key].append(self.acc_func(batch_data[self.batch_key]))

    def on_step_end(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]: 
        step_data[self.step_key] = self.red_func(step_data[self.temp_key])


class MultiAccumulator(ElasticHook):
    def __init__(self, batch:Dict[str,Callable], step:Dict[str,Callable]):
        super().__init__(self.__class__.__name__)
        self.batch = batch
        self.step = step

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        for bk, sk in zip(self.batch.keys(), self.step.keys()):
            step_data[f'{bk}_{sk}'] = []

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        for bk, sk in zip(self.batch.keys(), self.step.keys()):
            step_data[f'{bk}_{sk}'].append(self.batch[bk](batch_data[bk]))

    def on_step_end(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]: 
        for bk, sk in zip(self.batch.keys(), self.step.keys()):
            step_data[sk] = self.step[bk](step_data[f'{bk}_{sk}'])


class FuncAccumulator(ElasticHook):
    def __init__(self, batch:Dict[str,Callable], reducer:List[Callable]):
        super().__init__(self.__class__.__name__)
        self.batch = batch
        self.reducer = reducer

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        for k in self.batch.keys(): step_data[k] = [] 

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        for k, func in self.batch.items(): step_data[k].append(func(batch_data))

    def on_step_end(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]: 
        for func in self.reducer: func(step_data)