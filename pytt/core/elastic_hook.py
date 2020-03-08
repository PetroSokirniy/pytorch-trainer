from typing import List, Tuple, Any, Dict, Callable

class ElasticHook(object):
    def __init__(self, name:str=None, trainer=None, runner=None):
        self.name = name
        self.trainer = trainer
        self.runner = runner
        
    def set_trainer(self, trainer): self.trainer = trainer
    def set_runner(self, runner): self.runner = runner

    def on_fit_begin(self, epochs:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: return trainer_data
    def on_fit_end(self, epochs:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: return trainer_data

    def on_epoch_begin(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: return trainer_data
    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: return trainer_data

    def on_step_begin(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]: return step_data
    def on_step_end(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]: return step_data

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]): return batch_data, step_data
      
    def hook(self, args=None) -> Any: pass
