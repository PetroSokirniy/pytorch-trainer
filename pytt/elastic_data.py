from .core import *

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from fastprogress import force_console_behavior
import fastprogress
master_bar, progress_bar = force_console_behavior()

class BasicDataIter(ElasticHook):
    def __init__(self, loader_key=LOADER, **kwargs):
        super().__init__(self.__class__.__name__)
        self.loader_key = loader_key
        self.kwargs = kwargs

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        dataloder = DataLoader(step_data[self.loader_key], **self.kwargs)
        step_data[DATE_ITER] = progress_bar(dataloder, leave=False)

        


