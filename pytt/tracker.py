from typing import List, Tuple, Any, Dict, Callable

from .core import ElasticHook

import pandas as pd


class StatTracker(ElasticHook):
    def __init__(self, mapping:Dict[str,List[str]],  path:str=None, save_key='stats'):
        super().__init__(self.__class__.__name__)
        self.mapping = mapping
        self.path = path
        self.save_key = save_key
        self.stats = []
        self.epoch_counter = 0

    def to_df(self):
        return pd.DataFrame(self.stats)

    def load_previous(self):
        df = pd.read_csv(self.path)
        self.stats = df.to_dict('records')

    def save(self, path=None):
        if path is None:
            path = self.path

        if path is not None:
            self.to_df().to_csv(path, index=False)

    def on_fit_begin(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        trainer_data[self.save_key] = self.stats 

    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        row = {}
        for k, stats in self.mapping.items():
            for stat in stats:
                row[f'{k}_{stat}'] = trainer_data[k][stat]
        row['epoch'] = self.epoch_counter
        self.epoch_counter += 1
        self.stats.append(row)
        self.save()


class PrintHook(ElasticHook):
    def __init__(self, keys=None, key_replace={}, print_hist=True, stats_per_line=8, input_key='stats'):
        super().__init__(self.__class__.__name__)
        self.keys = keys
        self.key_replace = key_replace
        self.print_hist = print_hist
        self.stats_per_line = stats_per_line
        self.input_key = input_key 

    def _replace_key(self, key):
        for k,v in self.key_replace.items():
            key = key.replace(k,v)
        return key

    def on_fit_begin(self, epochs, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        if self.print_hist:
            self.print_hist = False
            if self.input_key in trainer_data and len(trainer_data[self.input_key]) != 0:
                self.print(trainer_data[self.input_key][-1], self.keys, self.stats_per_line)

    def on_epoch_end(self, e:int, trainer_data:Dict[str,Any]) -> Dict[str,Any]:
        self.print(trainer_data[self.input_key][-1], self.keys, self.stats_per_line)

    def print(self, stats:Dict[str,Any], keys=None, stats_per_line=None):
        if keys is None: keys = self.keys
        if keys is None: keys = [k for k in stats.keys() if 'int' in str(type(k)) or 'float' in str(type(k))]

        if stats_per_line is None: stats_per_line = self.stats_per_line

        i = 0
        p_str  = ''
        for key in keys:
            _key = self._replace_key(key)
            if 'float' in str(type(stats[key])):
                p_str += f'{_key}:{stats[key]:0.5f} '
            else:
                p_str += f'{_key}:{stats[key]} '
            i += 1
            if i > stats_per_line:
                p_str += '\n\t'
                i = 0
        print(p_str)
        sys.stdout.flush()
