import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from tqdm import tqdm
import logging
import yaml
import time
import pdb
import os

import models
import datasets
from utils import criterions
from utils import AverageMeter, StreamSegMetrics, CkptManager, Visualizer
from utils import load_config, set_logger, set_random_seed, cal_flops, setup_ddp, cleanup_ddp
from utils import get_train_augmentation, get_val_augmentation
from utils import get_optimizer, get_scheduler


def train(config, model, loader, sampler, optimizer, criterion, epoch, scheduler):
    model.train()
    if config['train']['freeze_bn'] is True:
        for module in model.modules():
            module.eval() if isinstance(module, nn.BatchNorm2d) else None
    if config['distributed_data_parallel']:
        sampler.set_epoch(epoch)
    scaler = GradScaler(enabled=True)
    total_loss, batch_time = AverageMeter(), AverageMeter()
    for i, sample in (tbar := tqdm(enumerate(loader), total=len(loader)) 
                      if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']) 
                      else enumerate(loader)):
        start = time.time()
        inputs, targets = ((sample['RGB'].cuda(config['gpu']).float(), 
                            sample['DVS'].cuda(config['gpu']).float() if 'DVS' in sample else None), 
                           (sample['SEG'].cuda(config['gpu']).long(),
                            None, #sample['SEG_D4'].cuda(config['gpu']).float(), 
                            sample['SEG_BY'].cuda(config['gpu']).float(), epoch))
        with autocast(enabled=True):
            outputs = model(inputs, label_size=sample['SEG'].size()[-2:])
            loss, loss_items = criterion(outputs, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.synchronize()
        total_loss.update(loss.item())
        batch_time.update(time.time() - start)
        if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
            tbar.set_description('Train: epoch {}, loss: {:.8f}, lr0: {:.8f}, lr1: {:.8f}, time: {:.2f}s'.format(
                epoch, loss.item(), scheduler.get_last_lr()[0], scheduler.get_last_lr()[-1], batch_time.val))
    if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
        logging.info('epoch {}, average loss is {:.6f}, batch time is {:.2f}s.'.format(epoch, total_loss.avg, batch_time.sum))
    torch.cuda.empty_cache()


def validate(config, model, loader, epoch, visualizer):
    model.eval()
    with torch.no_grad():
        metric, batch_time = StreamSegMetrics(config['dataset']['num_classes']), AverageMeter()
        for i, sample in (tbar := tqdm(enumerate(loader), total=len(loader)) 
                          if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']) 
                          else enumerate(loader)):
            start = time.time()
            inputs, targets = ((sample['RGB'].cuda(config['gpu']).float(), 
                                sample['DVS'].cuda(config['gpu']).float() if 'DVS' in sample else None), 
                               (sample['SEG'].long(), ))
            outputs = model(inputs, label_size=sample['SEG'].size()[-2:])
            visualizer.visualize(sample, outputs)
            metric.update(targets[0].numpy(), outputs[0].detach().max(dim=1)[1].cpu().numpy())
            batch_time.update(time.time() - start)
            if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
                tbar.set_description('Validation: epoch {}, time: {:.2f}s'.format(epoch, batch_time.val))
    score = metric.get_results()
    if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
        logging.info('epoch {}, global accuracy is {:.6f}, mean accuracy is {:.6f}, mean IoU is {:.6f}, batch time is {:.2f}s.'
                    .format(epoch, score['Overall Acc'], score['Mean Acc'], score['Mean IoU'], batch_time.sum))
    return score


def create_model(config):
    model = getattr(models, config['model']['arch'])(config=config['model'],
                                                     backbone=config['model']['backbone'], 
                                                     num_classes=config['dataset']['num_classes'], 
                                                     pretrained_weights=config['model']['pretrained_weights'])
    model = model.cuda(config['gpu'])
    return model, model.get_param_groups()


def create_loaders(config):
    if config['eval']:
        if config.get('test') is None:
            config['test'] = config['val']
        test_transform = get_val_augmentation(config['test']['size'] if 'size' in config['test'] else None)
        test_set = getattr(datasets, config['dataset']['dataset'])(config['dataset'], config['test'], test_transform)
        test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'], num_workers=config['test']['num_workers'],
                                pin_memory=True, sampler=None)
        return None, test_loader, None
    train_transform = get_train_augmentation(config['train']['crop_size'], seg_fill=255)
    val_transform = get_val_augmentation(config['val']['size'] if 'size' in config['val'] else None)
    train_set = getattr(datasets, config['dataset']['dataset'])(config['dataset'], config['train'], train_transform)
    val_set = getattr(datasets, config['dataset']['dataset'])(config['dataset'], config['val'], val_transform)
    if config['distributed_data_parallel']:
        train_sampler = DistributedSampler(train_set, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'],
                              pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=config['val']['batch_size'], num_workers=config['val']['num_workers'],
                            pin_memory=True, sampler=None)
    return train_loader, val_loader, train_sampler


def main(config):
    logging.info('Load Config from {}'.format(config['config_file']))
    logging.info('Configuration:\n' + yaml.dump(config))
    model, param_groups = create_model(config)
    if config['distributed_data_parallel']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['gpu']], output_device=0, find_unused_parameters=True)
    train_loader, val_loader, train_sampler = create_loaders(config)
    if not config['eval']:
        optimizer = get_optimizer(config, param_groups)
        scheduler = get_scheduler(config, optimizer, train_loader)
    ckpt_manager = CkptManager(config, config['distributed_data_parallel'])
    visualizer = Visualizer(config)
    cal_flops(model)
    if config['eval']:
        state_dict = ckpt_manager.load(config['model_path'])
        model.load_state_dict(state_dict['state_dict'])
        score = validate(config, model, val_loader, state_dict['epoch'], visualizer)
        if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
            logging.info('Score: \n' + StreamSegMetrics.to_str(score))
        exit()
    criterion = getattr(criterions, config['model']['criterion'])(config['model']).cuda(config['gpu'])
    start_epoch = 0
    if config['resume']:
        state_dict = ckpt_manager.load_resume()
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    if 'finetune_weights' in config['model']:
        state_dict = ckpt_manager.load(config['model']['finetune_weights'])
        model.load_state_dict(state_dict['state_dict'])
        logging.info('Use Finetune Model: {}'.format(config['model']['finetune_weights']))
        validate(config, model, val_loader, -1, visualizer)
    for epoch in range(start_epoch, config['train']['num_epochs']):
        train(config, model, train_loader, train_sampler, optimizer, criterion, epoch, scheduler)
        score = validate(config, model, val_loader, epoch, visualizer)
        if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
            ckpt_manager.save(score['Mean IoU'], epoch, model, optimizer, score)


if __name__ == '__main__':
    config = load_config()
    config['gpu'] = setup_ddp()
    set_random_seed(config['random_seed'])
    if (config['distributed_data_parallel'] and dist.get_rank() == 0) or (not config['distributed_data_parallel']):
        set_logger(config['logger'])
    main(config)
    cleanup_ddp()