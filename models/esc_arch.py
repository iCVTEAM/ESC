import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from functools import partial
from math import log2
from pythae.models import VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.vq_vae.vq_vae_utils import Quantizer, QuantizerEMA

from . import mix_transformer
from .segformer_head import Head_A, Head_B
from .vqvae import PureVQVAE


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class EdgeEncoder(nn.Module):
    def __init__(self, embed_dims, latent_dim=256, num_resnet_blocks=4):
        super().__init__()
        self.embed_dims = embed_dims
        enc_layers = [[ResBlock(embed_dims[i]) for j in range(num_resnet_blocks)] for i in range(4)]
        for i in range(4):
            enc_layers[i].append(nn.Conv2d(embed_dims[i], latent_dim, 1, 1))
        self.encoders = nn.ModuleList([nn.Sequential(*enc_layers[i]) for i in range(4)])
    def forward(self, x, index=0):
        output = self.encoders[index](x)
        return output


class ESC(nn.Module):
    def __init__(self, config, backbone, num_classes=20, embedding_dim=256, pretrained_weights=None):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.rgb_encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.rgb_encoder.embed_dims
        self.rgb_encoder.init_weights(pretrained_weights)
        self.dvs_encoder = mit_b1_dvs()
        self.dvs_encoder.init_weights('./pretrained/mit_b1.pth')
        self.decoder = Head_B(in_channels=self.in_channels, 
                              embedding_dim=self.embedding_dim,
                              num_classes=self.num_classes)
        self.resolvers = nn.ModuleList([
            ResolveBodyEdge(self.in_channels[i], partial(nn.BatchNorm2d, eps=1e-6)) for i in range(4)])
        self.rgb_edge_decoder = Head_A(in_channels=[self.embedding_dim] * 4, 
                                          embedding_dim=self.embedding_dim, 
                                          num_classes=config['num_embeddings']) # The number of VQ-VAE Codewords K=512
        self.dvs_edge_decoder = Head_A(in_channels=[self.embedding_dim] * 4, 
                                          embedding_dim=self.embedding_dim, 
                                          num_classes=config['num_embeddings']) # The number of VQ-VAE Codewords K=512
        self.rgb_edge_encoder = EdgeEncoder(embed_dims=self.in_channels, latent_dim=self.embedding_dim, num_resnet_blocks=2)
        self.dvs_edge_encoder = EdgeEncoder(embed_dims=self.in_channels, latent_dim=self.embedding_dim, num_resnet_blocks=2)
        self.softmax = nn.Softmax(dim=1)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.set_vqvae(config)

    def set_vqvae(self, config):
        self.vqvae = PureVQVAE(model_config=VQVAEConfig(use_ema=False, latent_dim=256, num_embeddings=config['num_embeddings']))
        state_dict_path = self.config['vqvae_weights']
        state_dict = torch.load(state_dict_path, map_location='cpu')
        logging.info('Load VQVAE Embeddings from path: {}'.format(state_dict_path))
        vqvae_dict = {'quantizer.embeddings.weight': state_dict['state_dict']['vqvae.quantizer.embeddings.weight']}
        self.vqvae.load_state_dict(vqvae_dict)
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def embeddings_requires_grad(self, requires_grad):
        for param in self.vqvae.parameters():
            param.requires_grad = requires_grad

    def get_param_groups(self):
        param_groups = [[], [], []]
        for name, param in list(self.rgb_encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for name, param in list(self.dvs_encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for name, param in list(self.resolvers.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for name, param in list(self.rgb_edge_encoder.named_parameters()):
            param_groups[0].append(param)
        for name, param in list(self.dvs_edge_encoder.named_parameters()):
            param_groups[0].append(param)
        for name, param in list(self.vqvae.named_parameters()):
            param_groups[2].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)
        for param in list(self.rgb_edge_decoder.parameters()):
            param_groups[2].append(param)
        for param in list(self.dvs_edge_decoder.parameters()):
            param_groups[2].append(param)
        return param_groups
        
    def forward(self, inputs, label_size=None):
        rgb = inputs[0]
        dvs = inputs[1]

        rgb_features = self.rgb_encoder(rgb)
        dvs_features = self.dvs_encoder(dvs)

        resolved = [self.resolvers[i](rgb_features[i]) for i in range(4)]
        edge_features = [resolved[i][1] for i in range(4)]

        rgb_edge_encoded_features = [self.rgb_edge_encoder(edge_features[i], index=i) for i in range(4)]
        dvs_edge_encoded_features = [self.dvs_edge_encoder(dvs_features[i], index=i) for i in range(4)]
        
        rgb_edge_outputs = self.rgb_edge_decoder(rgb_edge_encoded_features)
        dvs_edge_outputs = self.dvs_edge_decoder(dvs_edge_encoded_features)

        rgb_score = self.softmax(rgb_edge_outputs)
        dvs_score = self.softmax(dvs_edge_outputs)
        similarity = self.cosine_similarity(rgb_score, dvs_score)
        rgb_confidence, rgb_indices = torch.max(rgb_score, dim=1)
        dvs_confidence, dvs_indices = torch.max(dvs_score, dim=1)
        
        rgb_embeddings = self.vqvae.quantizer.embeddings(rgb_indices).permute(0, 3, 1, 2)
        dvs_embeddings = self.vqvae.quantizer.embeddings(dvs_indices).permute(0, 3, 1, 2)

        outputs = self.decoder(rgb_features, (rgb_edge_encoded_features, dvs_edge_encoded_features),
                               (rgb_embeddings, dvs_embeddings), (rgb_confidence, dvs_confidence), similarity)
        if label_size is not None:
            outputs = F.interpolate(outputs, size=label_size, mode='bilinear', align_corners=False)
        else:
            outputs = F.interpolate(outputs, size=rgb.size()[2:], mode='bilinear', align_corners=False)
        
        return outputs, rgb_edge_outputs, dvs_edge_outputs, (similarity, rgb_confidence, dvs_confidence) #(vqvae_losses, recon_losses, vq_losses)
    

class ResolveBodyEdge(nn.Module):
    def __init__(self, embed_dim, norm_layer):
        super(ResolveBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, groups=embed_dim, stride=2),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, groups=embed_dim, stride=2),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(embed_dim *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        feature_down = self.down(x)
        feature_down = F.interpolate(feature_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, feature_down], dim=1))
        feature_flow_warp = self.flow_warp(x, flow, size)
        feature_edge = x - feature_flow_warp
        return feature_flow_warp, feature_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output
    

class mit_b1_dvs(mix_transformer.MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1_dvs, self).__init__(
            patch_size=4, in_chans=5, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        
    
    def init_weights(self, pretrained_weights=None):
        if isinstance(pretrained_weights, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            state_dict = torch.load(pretrained_weights)
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict.pop('patch_embed1.proj.weight')
            self.load_state_dict(state_dict, strict=False)


class mit_b2_dvs(mix_transformer.MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2_dvs, self).__init__(
            patch_size=4, in_chans=5, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        
    
    def init_weights(self, pretrained_weights=None):
        if isinstance(pretrained_weights, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            state_dict = torch.load(pretrained_weights)
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict.pop('patch_embed1.proj.weight')
            self.load_state_dict(state_dict, strict=False)