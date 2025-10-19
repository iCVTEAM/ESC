import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from pythae.models import VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.vq_vae.vq_vae_utils import Quantizer, QuantizerEMA

class ResBlock_v1(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x
    

class ResBlock_v2(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)
    

class VQVAEEncoder(nn.Module):
    def __init__(self, embed_dim=1, latent_dim=512, hidden_dim=16, num_resnet_blocks=2, residul_type='v1', downsample_ratio=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = int(log2(downsample_ratio) - 1)
        self.residul_type = residul_type
        if self.residul_type == 'v1':
            ResBlock = ResBlock_v1
        elif self.residul_type == 'v2':
            ResBlock = ResBlock_v2

        dim = self.hidden_dim
        enc_layers = [
            nn.Conv2d(embed_dim, dim, 4, stride=2, padding=1), nn.ReLU()]

        for i in range(self.layer_num):
            enc_layers.append(nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            dim = dim * 2

        for i in range(num_resnet_blocks):
            enc_layers.append(ResBlock(dim))
        enc_layers.append(nn.Conv2d(dim, latent_dim, 1))
        self.encoder = nn.Sequential(*enc_layers)

    def forward(self, x):
        output = self.encoder(x)
        return output
    

class VQVAEDecoder(nn.Module):
    def __init__(self, embed_dim=1, latent_dim=512, hidden_dim=16, num_resnet_blocks=2, residul_type='v1', downsample_ratio=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.downsample_ratio = downsample_ratio
        self.layer_num = int(log2(downsample_ratio) - 1)
        self.residul_type = residul_type
        if self.residul_type == 'v1':
            ResBlock = ResBlock_v1
        elif self.residul_type == 'v2':
            ResBlock = ResBlock_v2

        dim = self.hidden_dim * self.downsample_ratio // 2

        dec_layers = [nn.Conv2d(latent_dim, dim, 1), nn.ReLU()]

        for i in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dim))

        dec_layers.append(nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1))
        dec_layers.append(nn.ReLU())
        for i in range(self.layer_num):
            dec_layers.append(nn.ConvTranspose2d(
                dim, dim // 2, 4, stride=2, padding=1))
            dec_layers.append(nn.ReLU())
            dim = dim // 2
        dec_layers.append(nn.Conv2d(dim, embed_dim, 1))

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, z):
        output = self.decoder(z)
        return output
    

class PureVQVAE(nn.Module):
    def __init__(self, model_config: VQVAEConfig):
        super().__init__()
        self.latent_dim = model_config.latent_dim
        self.model_config = model_config
        self._set_quantizer(model_config)
        self.model_name = "VQVAE"

    def _set_quantizer(self, model_config):
        self.model_config.embedding_dim = self.latent_dim
        if model_config.use_ema:
            self.quantizer = QuantizerEMA(model_config=model_config)
        else:
            self.quantizer = Quantizer(model_config=model_config)

    def forward(self, inputs, **kwargs):
        embeddings = inputs.permute(0, 2, 3, 1)

        quantizer_output = self.quantizer(embeddings)

        quantized_embed = quantizer_output.quantized_vector
        quantized_indices = quantizer_output.quantized_indices
        loss = quantizer_output.loss

        output = ModelOutput(
            loss=loss,
            z=quantized_embed,
            quantized_indices=quantized_indices,
        )

        return output


class VQVAE(nn.Module):
    def __init__(self, vqvae, encoder, decoder):
        super().__init__()
        self.vqvae = vqvae
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        x = self.encoder(inputs)
        outputs = self.vqvae(x)
        recon_x = self.decoder(outputs.z)

        outputs['recon_x'] = recon_x
        return outputs