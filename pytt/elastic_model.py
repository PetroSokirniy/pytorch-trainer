from .core import *

import torch

from fastprogress import force_console_behavior
import fastprogress
master_bar, progress_bar = force_console_behavior()

class BasicModelLoop(ElasticHook):
    def __init__(self, add_final=False):
        super().__init__(self.__class__.__name__)
        self.add_final = add_final
    
    def on_epoch_begin(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]: 
        self.model = trainer_data[MODEL]

    def on_batch(self, b:int, batch_data:Dict[str,Any], step_data:Dict[str,Any]):
        if step_data[step_data]:
            self.model(batch_data)
            step_data[LOSS_FN](b, batch_data, step_data)
        else:
            with torch.no_grad():
                self.model(batch_data)
                step_data[LOSS_FN](b, batch_data, step_data)


    def on_step_end(self, step:str, step_data:Dict[str,Any]) -> Dict[str,Any]:
        if self.add_final:
            step_data[LOSS_FN](-1, None, step_data)
