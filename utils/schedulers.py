import torch
import logging

def get_scheduler(config, optimizer, loader):
    if 'scheduler' not in config['model']:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif config['model']['scheduler'] == 'Cyclic01':
        logging.info('Use Cyclic-01 Scheduler')
        base_lr = config['train']['learning_rate']
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[base_lr, base_lr, base_lr * 10], 
                                                 max_lr=[base_lr * 1.6, base_lr * 1.6, base_lr * 16], 
                                                 step_size_up=len(loader) * 5, mode='triangular', cycle_momentum=False)
    elif config['model']['scheduler'] == 'WarmupPoly01':
        class WarmupPolyLR:
            def __init__(self, power, max_iter, warmup_iter=500, warmup_ratio=5e-4, warmup='exp'):
                self.power = power
                self.max_iter = max_iter
                self.warmup_iter = warmup_iter
                self.warmup_ratio = warmup_ratio
                self.warmup = warmup
            
            def get_lr_ratio(self, last_epoch):
                return self.get_warmup_ratio(last_epoch) if last_epoch < self.warmup_iter else self.get_main_ratio(last_epoch)
            
            def get_warmup_ratio(self, last_epoch):
                assert self.warmup in ['linear', 'exp']
                alpha = last_epoch / self.warmup_iter
                return self.warmup_ratio + (1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)
            
            def get_main_ratio(self, last_epoch):
                real_iter = last_epoch - self.warmup_iter
                real_max_iter = self.max_iter - self.warmup_iter
                alpha = real_iter / real_max_iter
                return (1 - alpha) ** self.power

        logging.info('Use Warmup-Poly-01 Scheduler')
        warmup_poly = WarmupPolyLR(power=0.9, max_iter=len(loader) * config['train']['num_epochs'], warmup_iter=len(loader) * 10, warmup_ratio=0.1, warmup='linear')
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter: warmup_poly.get_lr_ratio(iter))