from typing import List, Tuple, Any, Dict, Callable

from .elastic_runner import ElasticRunner
from .types import *


class ElasticTrainer(object):
    def __init__(self, runner:ElasticRunner, trainer_dict=Dict[str,Any], order:List[str]=None):
        self.runner = runner 
        self.runner.set_trainer(self)
        self.trainer_dict = trainer_dict
        if order is None:
            self.order = self.trainer_dict[ORDER]

    def fit(self, epochs=None, runner=None):
        if epochs is None:
            epochs = self.trainer_dict[EPOCHS]

        if runner is None:
            runner = self.runner
        
        runner.on_fit_begin(epochs, self.trainer_dict)
        for e in range(epochs):
            runner.on_epoch_begin(e, self.trainer_dict)
            for step in self.order:
                step_data = self.trainer_dict[step]
                step_data[NAME] = step
                runner.on_step_begin(step, step_data)
                b = 0
                for batch_data in step_data[DATE_ITER]:
                    runner.on_batch(b, batch_data, step_data)
                    b += 1
                runner.on_step_end(step, step_data)
            runner.on_epoch_end(e, self.trainer_dict)
        runner.on_fit_end(epochs, self.trainer_dict)