import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import random
from main import instantiate_from_config
from taming.modules.util import scatter_mask, box_mask, mixed_mask
from taming.modules.diffusionmodules.model import Encoder, Decoder, MatEncoder, MaskEncoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

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

class RefinementAE(pl.LightningModule):
    '''
        Refinement model for maskgit transformer:
            refine given a recomposition of the inferred masked region and the original image
    '''
    def __init__(self,
                 first_stage_config,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 mask_lower = 0.25,
                 mask_upper = 0.75,
                 first_stage_model_type='vae',
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 restriction=False,
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        decoder_model = Decoder
        # self.encoder = MatEncoder(**ddconfig)
        self.encoder = MaskEncoder(**ddconfig)
        self.decoder = decoder_model(**ddconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.loss = instantiate_from_config(lossconfig)
        self.first_stage_model_type = first_stage_model_type
        self.init_first_stage_from_ckpt(first_stage_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
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

    def init_first_stage_from_ckpt(self, config, initialize_decoder=False):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        if initialize_decoder:
            self.post_bottleneck_conv = copy.deepcopy(model.post_quant_conv)
            self.decoder = copy.deepcopy(model.decoder)

    def encode(self, x, mask):
        # encode the composited image
        h, mask_out = self.encoder(x, mask)
        h = self.bottleneck_conv(h)
        return h, mask_out

    def decode(self, h):
        h = self.post_bottleneck_conv(h)
        dec, _ = self.decoder(h)
        return dec

    def decode_at_layer(self, quant, i):
        quant = self.post_bottleneck_conv(quant)
        _, feat = self.decoder(quant, target_i_level = i)
        return feat

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, batch, mask=None, composition=True):

        input = self.get_input(batch, self.image_key)

        # first, get a composition of quantized reconstruction and the original image
        if mask is None:
            mask = self.get_mask_eval([input.shape[0], 1, input.shape[2], input.shape[3]], input.device)
        
        # enable this if first stage model is an VQ autoencoder
        # x_fstg, _ = self.first_stage_model(input)

        if self.first_stage_model_type == 'vae':
            quant, _, info = self.first_stage_model.encode(input)
        else:
            x_fstg, quant = self.first_stage_model.forward_with_recon(batch, return_quant=True)

        # _, _, codes = info
        B, C, H, W = quant.shape
        
        # enable this if first stage model is a maskGIT transformer
        # x_fstg = self.first_stage_model.forward_with_recon(batch, mask=mask)
            
        # x_comp = mask * input + (1 - mask) * x_fstg
        # x_in = torch.cat([mask - 0.5, x_comp], dim=1)

        h, mask_out = self.encode(input, mask)
        h = h + quant * ( 1 - mask_out)
        dec = self.decode(h)       
        dec = mask * input + (1 - mask) * dec

        # for comparison only
        if self.first_stage_model_type == 'vae':
            x_fstg, _ = self.first_stage_model(input)
        
        x_comp = mask * input + (1 - mask) * x_fstg

        return dec, mask, x_comp

    # def refine(self, x, mask=None):
    #     dec = self.decode(self.encode(x))
    #     if mask is not None:
    #         return mask * x + (1-mask) * dec
    #     else:
    #         return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

    def get_mask(self, shape, device):
        p = random.uniform(self.mask_lower, self.mask_upper)
        return box_mask(shape, device, p)

    def get_mask_eval(self, shape, device):
        return box_mask(shape, device, 0.5, det=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, mask, _ = self(batch)
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
        x = self.get_input(batch, self.image_key)
        xrec, mask, _ = self(batch)
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
        x = self.get_input(batch, self.image_key)
        xrec, mask, _ = self(batch)
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

    def debug_log_image(self, rec, idx, tag="recon"):
        nb = rec.shape[0]
        rec = torch.clamp(rec, min=-1.0, max=1.0)
        rec = 2 * (rec - rec.min()) / (rec.max() - rec.min()) - 1
        for i in range(nb):
            img = rec[i].cpu().detach().numpy().transpose(1,2,0)
            write_images(os.path.join("logs/eval", f"{tag}_{idx}_{i}.png"), img)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.bottleneck_conv.parameters())+
                                  list(self.post_bottleneck_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, mask, xrec_fstg = self(batch)
        
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
            xrec_fstg = self.to_rgb(xrec_fstg)

        log["inputs"] = x
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


import torch.distributions as Dist
import copy

class RefinementDecoder(pl.LightningModule):
    '''
        Refinement model for the VQGAN model:
            noise modeling based on vector quantized latent codes from a pretrained VQGAN model
    '''

    def __init__(self,
                 first_stage_config,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_noise_sampler()     
        ddconfig = first_stage_config.params.ddconfig
        self.decoder = Decoder(**ddconfig)
        self.post_quant_conv = torch.nn.Conv2d(2 * embed_dim, ddconfig["z_channels"], 1)
        self.loss = instantiate_from_config(lossconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

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

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        # self.post_quant_conv = copy.deepcopy(model.post_quant_conv)
        # self.decoder = copy.deepcopy(model.decoder)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_noise_sampler(self):
        ###### very experimental here ###################
        embeddings = self.first_stage_model.quantize.embedding.weight.data
        L, D = embeddings.shape
        # k = 1.5
        # bias = (embeddings.max(0)[0] - embeddings.min(0)[0]) / (L-1) # D-vector for uniform range
        # self.noise_dist = Dist.uniform.Uniform(-k*bias, k*bias)
        ##################################################
        self.noise_dist = Dist.uniform.Uniform(torch.full([D],-1.0), torch.full([D],1.0))

    def forward(self, x):
        quant, _, info = self.first_stage_model.encode(x)
        _, _, codes = info
        B, C, H, W = quant.shape
        N = self.first_stage_model.quantize.n_e
        # codes_onehot = to_categorical(codes, N).reshape(B, H, W, N).permute(0,3,1,2).contiguous()
        # code_grid = codes.reshape(quant.shape[0],quant.shape[2],quant.shape[3])
        quant_embeddings = quant
        noise = self.noise_dist.sample([B,H,W]).permute(0,3,1,2).to(quant.device)
        # quant = quant + noise.to(quant.device)
        quant = torch.cat([noise, quant], dim=1)
        quant = self.post_quant_conv(quant)
        dec, _ = self.decoder(quant)
        return dec, quant_embeddings, noise

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, labels, _ = self(x)

        # upsampling label
        cH, cW = labels.shape[-2:]
        H, W = xrec.shape[-2:]
        labels = torch.nn.functional.interpolate(labels, scale_factor=H//cH)
        
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            cond=labels, last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            cond=labels, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, labels, _ = self(x)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            cond=labels, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            cond=labels, last_layer=self.get_last_layer(), split="val")
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
        x = self.get_input(batch, self.image_key)
        xrec, labels, _ = self(x)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            cond=labels, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            cond=labels, last_layer=self.get_last_layer(), split="val")
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

    def debug_log_image(self, rec, idx, tag="recon"):
        nb = rec.shape[0]
        rec = torch.clamp(rec, min=-1.0, max=1.0)
        rec = 2 * (rec - rec.min()) / (rec.max() - rec.min()) - 1
        for i in range(nb):
            img = rec[i].cpu().detach().numpy().transpose(1,2,0)
            write_images(os.path.join("logs/eval", f"{tag}_{idx}_{i}.png"), img)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, labels, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
