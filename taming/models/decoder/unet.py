import os
import torch
import torch.nn.functional as F
import torch.distributions as Dist
import pytorch_lightning as pl
import cv2
import numpy as np
import random
import copy
import taming.modules.diffusionmodules.stylegan as StyleGAN
from main import instantiate_from_config
from taming.modules.util import scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
from taming.modules.diffusionmodules.model import PartialEncoder, Encoder, Decoder, StyleGANDecoder, MatEncoder, MaskEncoder
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

def to_categorical(code, n_label):
    '''
        input: L length vector
        return: (L, N) onehot vectors
    '''
    assert code.max() < n_label
    onehot = torch.zeros([code.size(0), n_label], dtype=bool)
    onehot[np.arange(code.size(0)), code] = True
    return onehot.float().to(code.device)


class RefinementUNet(pl.LightningModule):
    '''
        Refinement model for maskgit transformer:
            refine a recomposed image
    '''
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 lossconfig = None,
                 first_stage_config = None,
                 first_stage_model_type='vae', # vae | transformer
                 mask_lower = 0.25,
                 mask_upper = 0.75,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 second_stage_refinement=False,
                 ):
        super().__init__()
        self.image_key = image_key

        self.encoder = Encoder(**ddconfig)       
        self.decoder = Decoder(**ddconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None

        self.first_stage_model_type = first_stage_model_type

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        if first_stage_config is not None:
            # initialize the U-Net with vq-vae if not resumed from a checkpoint
            self.init_first_stage_from_ckpt(first_stage_config, initialize_current=ckpt_path is None)

        self.image_key = image_key

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.mask_function = box_mask
        self.mask_lower = mask_lower
        self.mask_upper = mask_upper


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config, initialize_current=False):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

        if self.first_stage_model_type == 'vae':
            target_model = model
        elif self.first_stage_model_type == 'transformer':
            target_model = model.first_stage_model
        elif self.first_stage_model_type == 'decoder':
            target_model = model.first_stage_model
        else:
            raise Exception(f"Unrecognized model type {self.first_stage_model_type}")

        if initialize_current:
            self.encoder = copy.deepcopy(target_model.encoder)
            self.bottleneck_conv = copy.deepcopy(target_model.quant_conv)
            self.post_bottleneck_conv = copy.deepcopy(target_model.post_quant_conv)
            self.decoder = copy.deepcopy(target_model.decoder)

    def set_first_stage_model(self, model):
        self.first_stage_model = model

    def encode(self, x):
        # encode the composited image
        h = self.encoder(x)
        h = self.bottleneck_conv(h)
        return h

    def decode(self, h):
        h = self.post_bottleneck_conv(h)
        dec, _ = self.decoder(h)
        return dec

    def decode_at_layer(self, quant, i):
        quant = self.post_bottleneck_conv(quant)
        _, feat = self.decoder(quant, target_i_level = i)
        return feat

    @torch.no_grad()
    def refine(self, img, mask):
        h = self.encode(img)
        dec = self.decode(h)
        dec = mask * img + (1 - mask) * dec
        return dec

    def forward(self, batch, quant=None, mask_in=None, mask_out=None, return_fstg=True, debug=False):

        input_raw = self.get_input(batch, self.image_key)

        # first, get a composition of quantized reconstruction and the original image
        if mask_in is None:
            mask = self.get_mask([input.shape[0], 1, input.shape[2], input.shape[3]], input.device)
        else:
            mask = mask_in

        input = input_raw * mask

        if self.first_stage_model_type == 'transformer':
            x_raw, quant_fstg = self.first_stage_model.forward_to_recon(batch, 
                                                                        mask=mask, 
                                                                        det=False, 
                                                                        return_quant=True)    
            x_comp = input + (1 - mask) * x_raw
        elif self.first_stage_model_type == 'vae':
            x_raw, _ = self.first_stage_model(input_raw)
            x_comp = input + (1 - mask) * x_raw
        elif self.first_stage_model_type == 'decoder':
            x_raw, mask = self.first_stage_model.generate(batch)
            input = input_raw * mask
            x_comp = x_raw
        else:
            raise Exception(f"Unrecognized model type {self.first_stage_model_type}")

        # forward pass with the recomposed image
        h = self.encode(x_comp)
        dec = self.decode(h)
        dec = input + (1 - mask) * dec
        return dec, mask, x_comp

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

    def get_mask(self, shape, device):
        return torch.from_numpy(BatchRandomMask(shape[0], shape[-1])).to(device)
        
    def get_mask_eval(self, shape, device):
        return box_mask(shape, device, 0.5, det=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        # mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask, _ = self(batch, mask_in=mask_in, mask_out=None)
        xfstg = None

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask = self(batch, mask_in=mask_in, mask_out=mask_out, return_fstg=False)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        from PIL import Image

        # We are always making assumption that the latent block is 16x16 here
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))

        x = self.get_input(batch, self.image_key)
        xrec, mask = self(batch, mask_in=mask_in, mask_out=mask_out, return_fstg=False)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.debug_log_image(x, batch_idx, tag='gt')
        self.debug_log_image(xrec, batch_idx)
        return self.log_dict

    def configure_optimizers_with_lr(self, lr):
        params = list(list(self.encoder.parameters()) + 
                 list(self.decoder.parameters()) + 
                 list(self.bottleneck_conv.parameters()) + 
                 list(self.post_bottleneck_conv.parameters()))

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


    def configure_optimizers(self):
        lr = self.learning_rate
        return self.configure_optimizers_with_lr(lr)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        # We are always making assumption that the latent block is 16x16 here
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask, xrec_fstg = self(batch, mask_in=mask_in, mask_out=mask_out)
        
        # log["inputs"] = x
        log["reconstructions"] = xrec
        log["masked_input"] = x * mask
        log['recon_fstg'] = xrec_fstg

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
