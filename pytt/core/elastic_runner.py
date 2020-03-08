from typing import List, Tuple, Any, Dict, Callable

from .elastic_hook import ElasticHook
from .type import *

class _ElasticRunner(ElasticHook):
    def __init__(self, hooks:List[ElasticHook]):
        self.hooks = hooks
        self.dict_hooks = {h.name:h for h in hooks if h.name is not None}
        for h in hooks: h.set_runner(self)

    def set_trainer(self, trainer): 
        self.trainer = trainer
        for h in self.hooks: h.set_trainer(trainer)

    def _run_dict_(key:str, dicts:List[Dict[str,Callable]], *args, **kwargs):
        for d in dicts:
            if key in d:
                for h in d[key]:
                    h(*args, **kwargs)

    def on_fit_begin(self, epochs, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        self._run_dict_(ON_FIT_BEGIN, [self.trainer.trainer_dict], 
            epochs=epochs, trainer_data=trainer_data)
        for h in self.hooks: h.on_fit_begin(epochs, trainer_data)  

    def on_fit_end(self, epochs, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        self._run_dict_(ON_FIT_END, [self.trainer.trainer_dict], 
            epochs=epochs, trainer_data=trainer_data)
        for h in self.hooks: h.on_fit_end(epochs, trainer_data)  

    def on_epoch_begin(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: 
        self._run_dict_(ON_EPOCH_BEGIN, [self.trainer.trainer_dict], 
            epochs=epochs, trainer_data=trainer_data)
        for h in self.hooks: h.on_epoch_begin(e, trainer_data)

    def on_epoch_end(self, e:int, trainer:Dict[str,Any]) -> Dict[str,Any]: 
        self._run_dict_(ON_EPOCH_END, [self.trainer.trainer_dict], 
            epochs=epochs, trainer_data=trainer_data)
        for h in self.hooks: h.on_epoch_end(e, trainer)

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        self._run_dict_(ON_STEP_BEGIN, [self.trainer.trainer_dict, step_data], 
            step=step, step_data=step_data)
        for h in self.hooks: h.on_step_begin(step, step_data)

    def on_step_end(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        self._run_dict_(ON_STEP_END, [self.trainer.trainer_dict, step_data], 
            step=step, step_data=step_data)
        for h in self.hooks: h.on_step_end(step, step_data)

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        self._run_dict_(ON_BATCH, [self.trainer.trainer_dict, step_data], 
            b=b, batch_data=batch_data, step_data=step_data)
        for h in self.hooks: h.on_batch(b, batch_data, step_data)

    def __len__(self) -> int:
        return len(self.hooks)

    def __getitem__(self, idx) -> ElasticHook:
        if isinstance(idx, int):
            return self.hooks[idx]
        elif isinstance(idx, str):
            return self.dict_hooks[idx]
        raise TypeError("idx has to be either a string or int")


class ElasticRunner(_ElasticRunner):
    def __add__(self, other) -> Any:
        if isinstance(other, _ElasticRunner):
            return HookList(self.hooks + other.hooks)
        elif isinstance(other, ElasticHook):
            return HookList(self.hooks + [other])
        raise TypeError("other has to be either a Hook or HookList")

    def __iadd__(self, other) -> None:
        if isinstance(other, _ElasticRunner):
            self.hooks += other.hooks
            self.dict_hooks.update(other.dict_hooks)
            return self
        elif isinstance(other, ElasticHook):
            self.hooks += [other]
            if other.name != None:
                self.dict_hooks[other.name] = other
            return self
        else:
            raise TypeError("other has to be either a Hook or HookList")