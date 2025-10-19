import torch
import logging

def get_optimizer(config, param_groups):
    if 'optimizer' in config['model'] and config['model']['optimizer'] == 'SGD':
        logging.info('Use SGD Optimizer')
        return torch.optim.SGD(
            params=[
                {'params': param_groups[0], 'lr': config['train']['learning_rate'], 'weight_decay': 0.01},
                {'params': param_groups[1], 'lr': config['train']['learning_rate'], 'weight_decay': 0.00},
                {'params': param_groups[2], 'lr': config['train']['learning_rate'] * 10, 'weight_decay': 0.01},
            ], momentum=0.9
        )
    logging.info('Use AdamW Optimizer')
    return torch.optim.AdamW(
        params=[
            {'params': param_groups[0], 'lr': config['train']['learning_rate'], 'weight_decay': 0.01},
            {'params': param_groups[1], 'lr': config['train']['learning_rate'], 'weight_decay': 0.00},
            {'params': param_groups[2], 'lr': config['train']['learning_rate'] * 10, 'weight_decay': 0.01},
        ]
    )