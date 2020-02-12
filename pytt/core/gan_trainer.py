from .trainer import *


class GANTrainer(Trainer):
    def __init__(self, dis_model:nn.Module, gen_model:nn.Module, 
            dis_optimizer:torch.optim.Optimizer, gen_optimizer:torch.optim.Optimizer, 
            dis_loss_fn:Callable, gen_loss_fn:Callable, hooks:HookList=None):

        self.dis_model = dis_model
        self.gen_model = gen_model

        self.dis_optimizer = dis_optimizer
        self.gen_optimizer = gen_optimizer

        self.dis_loss_fn = dis_loss_fn
        self.gen_loss_fn = gen_loss_fn

        if hooks is None:
            hooks = HookList([Hook()])

        self.hooks = hooks
        self.hooks.set_trainer(self)

    

    def fit(self, epochs:int, data, batch_sz=32, shuffle=True, sampler=None, num_workers=4) -> None:

        self.hooks.on_fit_begin()

        for e in tqdm(range(epochs)):
            if self.stop:
                break

            self.hooks.on_epoch_begin(e)

            for b, real_data in enumerate(DataLoader(data, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, num_workers=num_workers)):
                if self.stop:
                    break

                self.hooks.on_batch_begin(b) 

                real_data = self.hooks.on_batch_data_real(real_data) 
                real_data = self.dis_model(real_data)
                real_data = self.hooks.on_output_data_real(real_data)
                real_data = self._loss(real_data, self.dis_loss_fn, self.dis_optimizer)
                self.hooks.on_batch_end_data_real(real_data)

                fake_data = self.gen_model(batch_sz)
                fake_data = self.hooks.on_output_data_gen(fake_data)

                fake_data = self.dis_model(fake_data)
                fake_data = self.hooks.on_output_data_gen_dis(fake_data)
                fake_data = self._loss(fake_data, self.dis_loss_fn, self.dis_optimizer)
                fake_data = self._loss(fake_data, self.gen_loss_fn, self.gen_optimizer)

                self.hooks.on_batch_end(b) 

                self.hooks.on_batch_end_data(fake_data)

            self.hooks.on_epoch_end(e)

        self.hooks.on_fit_end()
