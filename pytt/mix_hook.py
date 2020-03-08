from .core import ElasticHook
from typing import List, Tuple, Any, Dict, Callable

import torch
import numpy as np

class MixUpHook(ElasticHook):
    def __init__(self, keys, alpha=1.0, beta=1.0, steps=['train']):
        super().__init__(self.__class__.__name__)
        self.keys = keys

        if isinstance(alpha, float):
            self.alpha = (alpha, alpha)
        else:
            self.alpha = alpha

        if isinstance(beta, float):
            self.beta = (beta, beta)
        else:
            self.beta = beta

        self.steps = steps
        self.run = False

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        self.run = step in self.steps

    def _get_lam_(self):
        alpha = np.random.uniform(*self.alpha)
        beta = np.random.uniform(*self.beta)
        lam = np.random.beta(alpha, beta)
        return lam

    def _get_perm_like_(self, x):
        return torch.randperm(x.shape[0]).to(x.device)
        
    def _mixup_data_(self, x, lam, index):
        return lam * x + (1 - lam) * x[index, :]

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        if self.run:
            lam = self._get_lam_()
            idx = self._get_perm_like_(batch_data[self.keys[0]])
            for k in self.keys:
                batch_data[k] =  self._mixup_data_(batch_data[k], lam, idx)
            batch_data['idx'] = idx
            batch_data['lam'] = lam