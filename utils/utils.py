import torch
import numpy as np
import logging
import random
import yaml
import cv2
import os
import torchvision.transforms.functional as TF
from models.vqvae import VQVAEEncoder, VQVAEDecoder, PureVQVAE, VQVAEConfig
import seaborn as sns
import re
import shutil
from fvcore.nn import flop_count_table, FlopCountAnalysis
from .meter import StreamSegMetrics


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.do_visualize = 'visualization_outputs' in config['test']
        if self.do_visualize and (not os.path.exists(config['test']['visualization_outputs'])):
            os.makedirs(config['test']['visualization_outputs'])
        self.set_vqvae(config)
        self.count = 0

    def set_vqvae(self, config):
        self.vqvae = PureVQVAE(model_config=VQVAEConfig(use_ema=False, latent_dim=256, num_embeddings=config['model']['num_embeddings'])).cuda(self.config['gpu'])
        self.vqvae_encoder = VQVAEEncoder(latent_dim=256, downsample_ratio=4).cuda(self.config['gpu'])
        self.vqvae_decoder = VQVAEDecoder(latent_dim=256, downsample_ratio=4).cuda(self.config['gpu'])
        state_dict_path = self.config['model']['vqvae_weights']
        state_dict = torch.load(state_dict_path, map_location='cuda:{}'.format(self.config['gpu']))
        vqvae_dict = {'quantizer.embeddings.weight': state_dict['state_dict']['vqvae.quantizer.embeddings.weight']}
        self.vqvae.load_state_dict(vqvae_dict)
        encoder_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict['state_dict'].items() if k.startswith('encoder')}
        self.vqvae_encoder.load_state_dict(encoder_dict)
        decoder_dict = {k.replace('decoder.', '', 1): v for k, v in state_dict['state_dict'].items() if k.startswith('decoder')} 
        self.vqvae_decoder.load_state_dict(decoder_dict)
        self.key = 'SEG_BY'


    def visualize_vqvae(self, sample, outputs):
        if not self.do_visualize:
            return      
        output_path = os.path.join(self.config['test']['visualization_outputs'], '{}'.format(self.config['task_name']))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with torch.no_grad():
            vqvae_outputs = self.vqvae(self.vqvae_encoder(sample[self.key].cuda(self.config['gpu']).float()))
        rgb_selections = outputs[1].detach().max(dim=1)[1].cpu().numpy()
        dvs_selections = outputs[2].detach().max(dim=1)[1].cpu().numpy()
        gt_selections = vqvae_outputs['quantized_indices'].squeeze(1).cpu().numpy()
        cmap = np.array(sns.color_palette("blend:#2e371a,#476205,#B5C09B", n_colors=self.config['model']['num_embeddings']))[::-1]
        cmap[76], cmap[127] = cmap[127].copy(), cmap[76].copy()
        cmap = (cmap * 255).astype(np.uint8)
        rgb_heatmap = np.array([cmap[i.astype(np.uint8)] for i in rgb_selections])
        dvs_heatmap = np.array([cmap[i.astype(np.uint8)] for i in dvs_selections])
        gt_heatmap = np.array([cmap[i.astype(np.uint8)] for i in gt_selections])
        return rgb_heatmap, dvs_heatmap, gt_heatmap

    def visualize(self, sample, outputs):
        if not self.do_visualize:
            return
        output_path = os.path.join(self.config['test']['visualization_outputs'], self.config['task_name'])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # import pdb; pdb.set_trace()
        if sample['RGB'].shape[2:] != sample['SEG'].shape[2:]:
            # sample['RGB'] = TF.resize(sample['RGB'], sample['SEG'].size()[2:], TF.InterpolationMode.BILINEAR, antialias=True)
            # sample['DVS'] = TF.resize(sample['DVS'], sample['SEG'].size()[2:], TF.InterpolationMode.BILINEAR, antialias=True)
            sample['SEG'] = TF.resize(sample['SEG'], sample['RGB'].size()[2:], TF.InterpolationMode.NEAREST, antialias=True)
            sample['SEG_BY'] = TF.resize(sample['SEG_BY'], sample['RGB'].size()[2:], TF.InterpolationMode.NEAREST, antialias=True)
            predict = TF.resize(outputs[0], sample['RGB'].size()[2:], TF.InterpolationMode.NEAREST, antialias=True)   
            gt = sample['SEG']
            edge_gt = sample['SEG_BY']
        else:
            predict = outputs[0]
            gt = sample['SEG']
            edge_gt = sample['SEG_BY']

        rgb_heatmap, dvs_heatmap, gt_heatmap = self.visualize_vqvae(sample, outputs)
        
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        rgb = sample['RGB'].cpu()
        rgb = TF.normalize(rgb, mean=[-m / s for m, s in zip(mean, std)], std=[1.0 / s for s in std])
        rgb = (rgb * 255).type(torch.uint8).numpy()
        gt = gt.squeeze(1).cpu().numpy()
        predict = predict.detach().max(dim=1)[1].cpu().numpy()
        rgb_edge_predict = self.vqvae_decoder(self.vqvae.quantizer.embeddings(outputs[1].detach().max(dim=1)[1]).permute(0, 3, 1, 2)).clamp(0, 1).squeeze(1).cpu().numpy()
        dvs_edge_predict = self.vqvae_decoder(self.vqvae.quantizer.embeddings(outputs[2].detach().max(dim=1)[1]).permute(0, 3, 1, 2)).clamp(0, 1).squeeze(1).cpu().numpy()
        edge_gt = edge_gt.squeeze(1).cpu().numpy()
        similarity, rgb_confidence, dvs_confidence = outputs[3][0].detach().cpu().numpy(), outputs[3][1].detach().cpu().numpy(), outputs[3][2].detach().cpu().numpy()

        cmap = np.zeros((256, 3), np.uint8)
        cmap[:11] = np.load('./utils/cmap11.npy')
        cmap[255] = (255, 255, 255)
        gt = np.array([cmap[i.astype(np.uint8)] for i in gt])
        predict = np.array([cmap[i.astype(np.uint8)] for i in predict])
        rgb = rgb.transpose(0, 2, 3, 1)
        rgb_edge_predict = (np.stack([rgb_edge_predict] * 3, axis=3) * 255).astype(np.uint8)
        dvs_edge_predict = (np.stack([dvs_edge_predict] * 3, axis=3) * 255).astype(np.uint8)
        edge_gt = np.stack([edge_gt] * 3, axis=3) * 255
        similarity, rgb_confidence, dvs_confidence = (similarity * 255).astype(np.uint8), (rgb_confidence * 255).astype(np.uint8), (dvs_confidence * 255).astype(np.uint8)

        dvs_voxel = sample['DVS'].cpu().numpy().sum(axis=1)
        dvs_positive = dvs_voxel.clip(min=0)
        dvs_negative = (- dvs_voxel).clip(min=0)
        max_value = max(dvs_positive.max(), dvs_negative.max())
        dvs = np.zeros_like(rgb)
        if max_value > 0:
            dvs[:, :, :, 0] = (dvs_negative / max_value) * 63
            dvs[:, :, :, 2] = (dvs_positive / max_value) * 63
            dvs[dvs > 0] += 192

        for i in range(rgb.shape[0]):
            cv2.imwrite(os.path.join(output_path, '{:04d}-{}.png'.format(self.count, sample['ID'][i])), 
                        cv2.cvtColor(np.concatenate([rgb[i], dvs[i], predict[i], gt[i]], axis=1), cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(output_path, '{:04d}-{}_edge.png'.format(self.count, sample['ID'][i])), 
                        cv2.cvtColor(np.concatenate([predict[i], rgb_edge_predict[i], dvs_edge_predict[i], edge_gt[i]], axis=1), cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(output_path, '{:04d}-{}_confilarity.png'.format(self.count, sample['ID'][i])), 
                        cv2.cvtColor(np.concatenate([similarity[i], rgb_confidence[i], dvs_confidence[i]], axis=1), cv2.COLOR_GRAY2RGB))
            cv2.imwrite(os.path.join(output_path, '{:04d}-{}_seg.png'.format(self.count, sample['ID'][i])), cv2.cvtColor(predict[i], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(output_path, '{:04d}-{}_gt.png'.format(self.count, sample['ID'][i])), cv2.cvtColor(gt[i], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(output_path, '{:04d}-{}_heatmap.png'.format(self.count, sample['ID'][i])), 
                        cv2.cvtColor(np.concatenate([rgb_heatmap[i], dvs_heatmap[i], gt_heatmap[i]], axis=1), cv2.COLOR_BGR2RGB))
            self.count += 1


class CkptManager:
    def __init__(self, config, data_parallel, initial_val=0, condition=lambda x, y: x > y):
        self.config = config
        self.ckpt_path = os.path.join(config['ckpt']['ckpt_base_path'], config['task_name'])
        # if not os.path.exists(self.ckpt_path):
        os.makedirs(self.ckpt_path, exist_ok=True)
        if not os.path.exists(os.path.join(self.ckpt_path, 'config.yaml')):
            with open(os.path.join(self.ckpt_path, 'config.yaml'), 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        self.data_parallel = data_parallel
        self.best_val = initial_val
        self.condition = condition
        self.save_checkpoints_at = config['ckpt']['save_checkpoints_at'] if 'save_checkpoints_at' in config['ckpt'] else None
        self.save_begin = config['ckpt']['save_begin'] if 'save_begin' in config['ckpt'] else 0
        self.last_epoch = config['train']['num_epochs']

    def save(self, new_val, epoch, model, optimizer, score):
        state_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'score': score
        }
        if epoch >= self.save_begin and self.condition(new_val, self.best_val):
            logging.info('New best val {:.6f} at epoch {}, was {:.6f}'.format(new_val, epoch, self.best_val))
            self.best_val = new_val
            torch.save(state_dict, os.path.join(self.ckpt_path, 'model-best.pth.tar'))
        # torch.save(state_dict, os.path.join(self.ckpt_path, 'checkpoint.pth.tar'))
        if self.save_checkpoints_at is not None and epoch in self.save_checkpoints_at:
            torch.save(state_dict, os.path.join(self.ckpt_path, 'checkpoint_epoch_{}.pth.tar'.format(epoch)))
            logging.info('Save checkpoint at epoch {}, the val is {:.6f}'.format(epoch, new_val))
            shutil.copy(os.path.join(self.ckpt_path, 'model-best.pth.tar'), os.path.join(self.ckpt_path, 'model-best_epoch_{}.pth.tar'.format(epoch)))
        if epoch == self.last_epoch - 1:
            logging.info('Score: \n' + StreamSegMetrics.to_str(score))
            torch.save(state_dict, os.path.join(self.ckpt_path, 'model-last.pth.tar'))

    def load_resume(self):
        state_dict = torch.load(os.path.join(self.ckpt_path, 'checkpoint.pth.tar'), map_location='cuda:{}'.format(self.config['gpu']))
        logging.info('Load checkpoint from {}, current epoch is {}'.format(
            os.path.join(self.ckpt_path, 'checkpoint.pth.tar'), state_dict['epoch']))
        # self.best_val = state_dict['score']['Mean IoU']
        if not self.data_parallel:
            state_dict['state_dict'] = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
            return state_dict
        return state_dict
    
    def load_eval(self):
        state_dict = torch.load(os.path.join(self.ckpt_path, 'model-best.pth.tar'), map_location='cuda:{}'.format(self.config['gpu']))
        logging.info('Load best model from {}'.format(os.path.join(self.ckpt_path, 'model-best.pth.tar')))
        if not self.data_parallel:
            state_dict['state_dict'] = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
            return state_dict
        return state_dict

    def load(self, model_path):
        state_dict = torch.load(model_path, map_location='cuda:{}'.format(self.config['gpu']))
        logging.info('Load model from {}'.format(model_path))
        if not self.data_parallel:
            state_dict['state_dict'] = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
            return state_dict
        return state_dict

def cal_flops(model):
    try:
        rgb, dvs = torch.zeros(1, 3, 512, 512), torch.zeros(1, 5, 512, 512)

        if torch.cuda.is_available:
            rgb, dvs = rgb.cuda(), dvs.cuda()
            model = model.cuda()
        logging.info(flop_count_table(FlopCountAnalysis(model, [rgb, dvs])))
    except:
        pass

    try:
        rgb, dvs = torch.zeros(1, 3, 512, 512), torch.zeros(1, 6, 512, 512)

        if torch.cuda.is_available:
            rgb, dvs = rgb.cuda(), dvs.cuda()
            model = model.cuda()
        logging.info(flop_count_table(FlopCountAnalysis(model, [rgb, dvs])))
    except:
        pass

    try:
        rgb, dvs = torch.zeros(1, 3, 512, 512), torch.zeros(1, 3, 512, 512)

        if torch.cuda.is_available:
            rgb, dvs = rgb.cuda(), dvs.cuda()
            model = model.cuda()
        logging.info(flop_count_table(FlopCountAnalysis(model, [rgb, dvs])))
    except:
        pass

    try:
        rgb = torch.zeros(1, 1, 512, 512)

        if torch.cuda.is_available:
            rgb = rgb.cuda()
            model = model.cuda()
        logging.info(flop_count_table(FlopCountAnalysis(model, rgb)))
    except:
        pass