from collections import OrderedDict

from .core import ElasticHook
from typing import List, Tuple, Any, Dict, Callable


class RunArgAug(ElasticHook):
    def __init__(self, in_keys, out_keys, func):
        super().__init__(self.__class__.__name__)
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.func = func

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        for ik, ok in zip(self.in_keys, self.out_keys):
            if isinstance(ok, list):
                batch_data[ok] = self.func(*[batch_data[k] for k in ok])
            else:
                batch_data[ok] = self.func(batch_data[ik])
        return batch_data, step_data


class RunBatchAug(ElasticHook):
    def __init__(self, keys, funcs=None):
        super().__init__(self.__class__.__name__)
        if isinstance(keys, OrderedDict):
            self.keys = keys.keys()
            self.funcs = keys.values()
        else:
            self.keys = keys
            self.funcs = funcs

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        for key, func in zip(self.keys, self.funcs):
            batch_data[k] = func(batch_data)
        return batch_data, step_data

