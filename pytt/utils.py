def adjust_learning_rate(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def identity(x):
	return x

def to_numpy(t):
	return t.detach().cpu().numpy()

def key_to_numpy(key):
    def _func(data:dict): return data[key].detach().cpu().numpy()
    return _func
    