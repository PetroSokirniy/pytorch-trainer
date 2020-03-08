from collections import defaultdict, OrderedDict
from typing import List, Tuple, Any, Dict, Callable

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

class GradSwitch(ElasticHook):
    def on_step_begin(self, step:str, step_data:Dict[str,Any]):
        if USE_GRAD in step_data:
            torch.set_grad_enabled(step_data[USE_GRAD])



