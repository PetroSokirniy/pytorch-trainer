from typing import List, Tuple, Any, Dict, Callable

class Hook(object):
    def __init__(self, name:str=None, trainer=None, hook_list=None):
        self.name = name
        self.trainer = trainer
        self.hook_list = hook_list
        
    def set_trainer(self, trainer): self.trainer = trainer
    def set_hook_list(self, hook_list): self.hook_list = hook_list

    def on_fit_begin(self) -> None: pass
    def on_fit_end(self) -> None: pass

    def on_epoch_begin(self, e:int) -> None: pass
    def on_epoch_end(self, e:int) -> None: pass

    def on_train_begin(self) -> None: pass
    def on_train_end(self) -> None: pass

    def on_validation_begin(self) -> None: pass
    def on_validation_end(self) -> None: pass

    def on_batch_begin(self, b:int, data:Dict[str,Any]): return data
    def on_output_data(self, data:Dict[str,Any]) -> Dict[str,Any]: return data
    def on_batch_end(self, b:int, data:Dict[str,Any]): return data

    def hook(self, args=None) -> Any: pass

class _HookList(Hook):
    def __init__(self, hooks:List[Hook]):
        self.hooks = hooks
        self.dict_hooks = {h.name:h for h in hooks if h.name is not None}
        self.hook_list = self
        for h in hooks: h.set_hook_list(self)

    def set_trainer(self, trainer): 
        self.trainer = trainer
        for h in self.hooks: h.set_trainer(trainer)

    def on_fit_begin(self) -> None:
        for h in self.hooks: h.on_fit_begin()  
    def on_fit_end(self) -> None:
        for h in self.hooks: h.on_fit_end()

    def on_train_begin(self) -> None: 
        for h in self.hooks: h.on_train_begin()
    def on_train_end(self) -> None:
        for h in self.hooks: h.on_train_end()
    
    def on_validation_begin(self) -> None:
        for h in self.hooks: h.on_validation_begin()
    def on_validation_end(self) -> None: 
        for h in self.hooks: h.on_validation_end()

    def on_epoch_begin(self, e:int) -> None:
        for h in self.hooks: h.on_epoch_begin(e)
    def on_epoch_end(self, e:int) -> None:
        for h in self.hooks: h.on_epoch_end(e)

    def on_batch_begin(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        for h in self.hooks: data = h.on_batch_begin(b, data)
        return data

    def on_batch_end(self, b:int, data:Dict[str,Any]) -> Dict[str,Any]:
        for h in self.hooks: data = h.on_batch_end(b, data)
        return data
    
    def on_output_data(self, data:Dict[str,Any]) -> Dict[str,Any]: 
        for h in self.hooks: data = h.on_output_data(data)
        return data

    def __len__(self) -> int:
        return len(self.hooks)

    def __getitem__(self, idx) -> Hook:
        if isinstance(idx, int):
            return self.hooks[idx]
        elif isinstance(idx, str):
            return self.dict_hooks[idx]
        raise TypeError("idx has to be either a string or int")
            

class HookList(_HookList):
    def __add__(self, other) -> Any:
        if isinstance(other, _HookList):
            return HookList(self.hooks + other.hooks)
        elif isinstance(other, Hook):
            return HookList(self.hooks + [other])
        raise TypeError("other has to be either a Hook or HookList")

    def __iadd__(self, other) -> None:
        if isinstance(other, _HookList):
            self.hooks += other.hooks
            self.dict_hooks.update(other.dict_hooks)
            return self
        elif isinstance(other, Hook):
            self.hooks += [other]
            if other.name != None:
                self.dict_hooks[other.name] = other
            return self
        else:
            raise TypeError("other has to be either a Hook or HookList")