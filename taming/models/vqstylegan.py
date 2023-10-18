import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
from main import instantiate_from_config
import taming.modules.diffusionmodules.stylegan as StyleGAN
from taming.modules.diffusionmodules.model import StyleGANDecoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.models.vqgan import VQModel

def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

class StyleGANVQModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 w_dim=512,
                 ckpt_path=None,
                 image_key="image",
                 ):
        super().__init__(ddconfig, 
                         lossconfig, 
                         n_embed, 
                         embed_dim, 
                         ckpt_path, 
                         image_key=image_key,
                         decoder_model=StyleGANDecoder)
        self.encoder_fc = StyleGAN.FullyConnectedLayer(16*16*ddconfig["z_channels"], w_dim, activation='relu')

    def encode(self, x, mask=None):
        if mask is not None:
          x = x * mask

        h = self.encoder(x)
        ws = self.encoder_fc(h.reshape(h.shape[0], -1))
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        if mask is not None:
          # a naive approach to produce downsampled mask
          H1 = x.shape[-1]
          H2 = quant.shape[-1]
          mask = torch.nn.functional.interpolate(mask.float(), scale_factor=H2/H1)
          return quant, ws, emb_loss, info, mask
        else:
          return quant, ws, emb_loss, info

    def decode(self, quant, ws):
        quant = self.post_quant_conv(quant)
        dec, _ = self.decoder(quant, ws)
        return dec

    def forward(self, input):
        quant, ws, diff, _ = self.encode(input)
        dec = self.decode(quant, ws)
        return dec, diff
