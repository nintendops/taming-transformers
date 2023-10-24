import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import random
import torch.distributions as Dist
import copy
from main import instantiate_from_config
from taming.models.decoder.refinementAE import RefinementAE
from taming.modules.util import scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
from taming.modules.diffusionmodules.model import ResnetBlock
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.diffusionmodules.mat import Conv2dLayerPartial, Conv2dLayerPartialRestrictive


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))



class RefinementAE_v2(RefinementAE):
    '''
        Refinement model (v2.0) for maskgit transformer:
            refine given a recomposition of the inferred masked region and the original image
        Added a latent mapping network which reformulate how the encoded features and quantized features are processed
    '''
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 first_stage_config = None,
                 first_stage_model_type='vae', # vae | transformer
                 mask_lower = 0.25,
                 mask_upper = 0.75,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 restriction=False        # whether a partial encoder is used
                 ):

        super().__init__(ddconfig, 
                         lossconfig, 
                         n_embed, 
                         embed_dim, 
                         first_stage_config=first_stage_config, 
                         first_stage_model_type=first_stage_model_type,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         restriction=restriction)

        self.temb_ch = 0
        modules = []
        for i in range(8):
            if i == 0:
                modules.append(ResnetBlock(in_channels=2*embed_dim, out_channels=embed_dim, temb_channels=self.temb_ch, dropout=0.0))
            else:                
                modules.append(ResnetBlock(in_channels=embed_dim, out_channels=embed_dim, temb_channels=self.temb_ch, dropout=0.0))

        self.mapping = torch.nn.ModuleList(modules)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def forward(self, batch, quant=None, mask_in=None, mask_out=None, return_fstg=True, use_noise=True, debug=False):

        input_raw = self.get_input(batch, self.image_key)

        # first, get a composition of quantized reconstruction and the original image
        if mask_in is None:
            mask = self.get_mask([input_raw.shape[0], 1, input_raw.shape[2], input_raw.shape[3]], input_raw.device)
        else:
            mask = mask_in

        input = input_raw * mask

        ###### for comparison only ################
        if self.first_stage_model_type == 'transformer':
            x_raw, quant_fstg = self.first_stage_model.forward_to_recon(batch, 
                                                                        mask=mask_out, 
                                                                        det=False, 
                                                                        return_quant=True)    
        if self.first_stage_model_type == 'vae':
            x_raw, _ = self.first_stage_model(input_raw)
        
        if return_fstg:
            x_comp = mask * input_raw + (1 - mask) * x_raw
        ############################################

        # quant_gt, _, info = self.first_stage_model.encode(input_raw)       
        # quant = quant_gt

        if quant is None:
            if self.first_stage_model_type == 'vae':
                quant_gt, _, info = self.first_stage_model.encode(input_raw)
                if use_noise:
                    B, C, H, W = quant_gt.shape
                    # randomly replace indices from gt info
                    gt_indices = info[2]
                    prob = 0.1
                    rand_indices = torch.rand(gt_indices.shape) * (self.n_embed - 1)
                    rand_indices = rand_indices.int().to(input.device)
                    rand_mask = (torch.rand(gt_indices.shape) < prob).int().to(input.device)
                    gt_indices = rand_mask * rand_indices + (1-rand_mask) * gt_indices
                    quant = self.first_stage_model.quantize.get_codebook_entry(gt_indices.reshape(-1).int(), shape=(B, H, W, C))
                else:
                    quant = quant_gt
            else:
                quant = quant_fstg

        # _, _, codes = info
        B, C, H, W = quant.shape
        
        if mask_out is None:
            h, mask_out = self.encode(input, mask)
        else:
            h, _ = self.encode(input, mask)

        # h = mask_out * h + quant * (1 - mask_out) * 0.5 + h * (1 - mask_out) * 0.5
        h = torch.cat([h, quant], dim=1)
        for block in self.mapping:
            h = block(h, None)
        dec = self.decode(h)
        dec = input + (1 - mask) * dec

        # Additional U-Net to refine output
        if self.use_refinement:
            x_fstg = dec
            dec = self.second_stage(dec, mask)
            dec = mask * input + (1 - mask) * dec
            if return_fstg:
                return dec, mask, x_comp, x_fstg
            else:
                return dec, mask
        else:
            if debug:
                return dec, mask, mask_out, quant * (1 - mask_out), h * (1 - mask_out)
            elif return_fstg:                
                return dec, mask, x_comp
            else:
                return dec, mask
