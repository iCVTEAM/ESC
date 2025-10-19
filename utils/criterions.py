import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

import logging

from pythae.models import VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.vq_vae.vq_vae_utils import Quantizer, QuantizerEMA
from models.vqvae import VQVAEEncoder, VQVAEDecoder, PureVQVAE


class StandardCrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, predictions, targets):
        loss = self.criterion(predictions[0], targets[0].squeeze(1))
        return loss, {'loss': loss.item()}


class LossForESC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.edge_criterion = nn.CrossEntropyLoss()
        self.loss_weight = config['loss_weight']
        logging.info('The Weighted Loss is {} * standard_loss + {} * edge_loss'.format(self.loss_weight[0], self.loss_weight[1]))
        self.set_vqvae()

    def set_vqvae(self):
        self.vqvae = PureVQVAE(model_config=VQVAEConfig(use_ema=False, latent_dim=256, num_embeddings=self.config['num_embeddings']))
        self.vqvae_encoder = VQVAEEncoder(latent_dim=256, downsample_ratio=4)
        self.vqvae_decoder = VQVAEDecoder(latent_dim=256, downsample_ratio=4)
        if 'vqvae_weights' in self.config:
            state_dict_path = self.config['vqvae_weights']
        else:
            state_dict_path = './pretrained/vqvae/240419_vqvae_w{}/model-best.pth.tar'.format(self.config['num_embeddings'])
        state_dict = torch.load(state_dict_path, map_location='cpu')
        logging.info('Load VQVAE state dict from path: {}'.format(state_dict_path))
        vqvae_dict = {'quantizer.embeddings.weight': state_dict['state_dict']['vqvae.quantizer.embeddings.weight']}
        self.vqvae.load_state_dict(vqvae_dict)
        encoder_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict['state_dict'].items() if k.startswith('encoder')}
        self.vqvae_encoder.load_state_dict(encoder_dict)
        decoder_dict = {k.replace('decoder.', '', 1): v for k, v in state_dict['state_dict'].items() if k.startswith('decoder')}
        self.vqvae_decoder.load_state_dict(decoder_dict)

    def forward(self, predictions, targets):
        standard_loss = self.criterion(predictions[0], targets[0].squeeze(1))
        with torch.no_grad():
            vqvae_outputs = self.vqvae(self.vqvae_encoder(targets[2]))
        rgb_edge_loss = self.edge_criterion(predictions[1], vqvae_outputs['quantized_indices'].squeeze(1))
        dvs_edge_loss = self.edge_criterion(predictions[2], vqvae_outputs['quantized_indices'].squeeze(1))
        edge_loss = rgb_edge_loss + dvs_edge_loss
        loss = self.loss_weight[0] * standard_loss + self.loss_weight[1] * edge_loss
        similarity, rgb_confidence, dvs_confidence = predictions[3][0].mean(), predictions[3][1].mean(), predictions[3][2].mean()

        return loss, {
            'loss': loss.item(),
            'standard_loss': self.loss_weight[0] * standard_loss.item(),
            'edge_loss': self.loss_weight[1] * edge_loss.item(),
            'rgb_edge_loss': rgb_edge_loss.item(),
            'dvs_edge_loss': dvs_edge_loss.item(), 
            'similarity': similarity.item(),
            'rgb_confidence': rgb_confidence.item(),
            'dvs_confidence': dvs_confidence.item(),
        }
    

class LossForVQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, predictions, targets):
        recon_loss = self.criterion(predictions['recon_x'], targets)
        vqvae_loss = predictions['loss'].mean()
        loss = vqvae_loss + recon_loss
        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'vqvae_loss': vqvae_loss.item()
        }