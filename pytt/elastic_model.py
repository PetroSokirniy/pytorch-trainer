from .core import *

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from fastprogress import force_console_behavior
import fastprogress
master_bar, progress_bar = force_console_behavior()

class BasicModelLoop(ElasticHook):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    
    def on_epoch_begin(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: 
        self.model = trainer_data[MODEL]

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        self.model(batch_data)
        step_data[LOSS_FN](b, batch_data, step_data)


        


