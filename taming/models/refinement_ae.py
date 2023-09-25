import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import random
from main import instantiate_from_config
from taming.modules.util import scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
from taming.modules.diffusionmodules.model import PartialEncoder, Encoder, Decoder, MatEncoder, MaskEncoder
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

# ENCODER MODEL
class MaskPartialEncoderModel(pl.LightningModule):
    def __init__(self,    
                 ddconfig,
                 n_embed,
                 embed_dim,
                 first_stage_config=None,            
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
        self.encoder = PartialEncoder(**ddconfig, simple_conv=False)
        self.cls_head = Conv2dLayerPartialRestrictive(ddconfig["z_channels"], n_embed, kernel_size=1, simple_conv=False)

        # self.proj = torch.nn.Linear(n_embed, n_embed)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if first_stage_config is not None:
            self.init_first_stage_from_ckpt(first_stage_config)
    
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

    def init_first_stage_from_ckpt(self, config, initialize_encoder=False):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        self.quantize = self.first_stage_model.quantize
        self.quantize.train = disabled_train

    def set_first_stage_model(self, model):
        self.first_stage_model = model
        self.quantize = model.quantize

    def encode_logits(self, x, mask):
        h, mask_out = self.encoder(x, mask)
        logits, mask_out = self.cls_head(h, mask_out)
        return logits, mask_out

    @torch.no_grad()
    def encode(self, x, mask=None, return_ref=False):       
        temperature = 1.0
        # quant_z_gt, _, info_gt = self.first_stage_model.encode(x)        

        if mask is None:
            mask_in = torch.full([x.shape[0],1,x.shape[2],x.shape[3]], 1.0, dtype=torch.int32).to(x.device)
        else:
            mask_in = mask

        x = x * mask_in 
        quant_z_ref, _, info = self.first_stage_model.encode(x)       
        indices_ref = info[2].reshape(-1)
        logits, mask_out = self.encode_logits(x, mask_in)     
        B, L, H, W = logits.shape
        logits = logits.permute(0,2,3,1).reshape(B, -1, L)
        probs = F.softmax(logits / temperature, dim=-1)
        _, indices = torch.topk(probs, k=1, dim=-1)
        indices = indices.reshape(-1).int()
        bhwc = (quant_z_ref.shape[0],
                quant_z_ref.shape[2],
                quant_z_ref.shape[3],
                quant_z_ref.shape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            indices, shape=bhwc)

        info = (None, None, indices)

        if return_ref:
            return quant_z, None, info, mask_out, quant_z_ref, indices_ref
        elif mask is not None:
            return quant_z, None, info, mask_out
        else:
            return quant_z, None, info

    @torch.no_grad()
    def encode_to_z_first_stage(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        return quant_z, info[2].view(quant_z.shape[0], -1)

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def decode(self, quant_z):
        return self.first_stage_model.decode(quant_z)

    def forward(self, x, mask=None):

        if mask is not None:
            x = mask * x
        else:
            mask = torch.full([x.shape[0],1,x.shape[2],x.shape[3]], 1.0, dtype=torch.int32).to(x.device)

        logits, mask_out = self.encode_logits(x, mask)
        return logits, mask_out

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        # obtain target quantized vectors 
        _, target_z_indices = self.encode_to_z_first_stage(x)
        # random mask function given by MAT paper
        mask_in = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)
        logits, mask_out = self(x, mask_in)
        B, L, H, W = logits.shape
        logits = logits.permute(0,2,3,1).reshape(-1, L)
        logits_select = torch.masked_select(logits, mask_out.reshape(-1,1).bool()).reshape(-1, L)
        target_select = torch.masked_select(target_z_indices.reshape(-1), mask_out.reshape(-1).bool())
        loss = F.cross_entropy(logits_select, target_select)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.cls_head.parameters()),
                                  lr=lr, betas=(0.9, 0.95))
        return optimizer


    def configure_optimizers_with_lr(self, lr):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.cls_head.parameters()),
                                  lr=lr, betas=(0.9, 0.95))
        return optimizer


    def log_images(self, batch, **kwargs):

        log = dict()

        temperature = 1.0
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        quant_z, gt_z_indices = self.encode_to_z_first_stage(x)
        # mask_in = box_mask(x.shape, x.device, 0.5, det=True)
        mask_in = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)
        logits, mask_out = self(x, mask_in)

        # upsampling
        B, _, H, W = x.shape
        _, _, gh, hw = mask_out.shape
        mask_out_reference = torch.round(torch.nn.functional.interpolate(mask_out.reshape(B,1,gh,hw).float(), scale_factor=H//gh))

        B, L, H, W = logits.shape
        logits = logits.permute(0,2,3,1).reshape(B, -1, L)

        # recomposing logits with gt logits
        probs = F.softmax(logits / temperature, dim=-1)
        _, indices_pred = torch.topk(probs, k=1, dim=-1)

        ######################################
        # quant_z, _, info, mask_out = self.encode(x, mask_in)
        # indices_pred = info[2]
        ######################################

        mask_out = mask_out.reshape(B, -1).int()
        indices_combined = mask_out * indices_pred.reshape(B, -1) + (1 - mask_out) * gt_z_indices
        xrec = self.decode_to_img(indices_combined.int(), quant_z.shape)

        # for comparison: encoding masked image with original encoder
        quant_z_ref, ref_z_indices = self.encode_to_z_first_stage(x * mask_in)
        indices_combined_ref = mask_out * ref_z_indices.reshape(B, -1) + (1 - mask_out) * gt_z_indices
        xrec_ref = self.decode_to_img(indices_combined_ref.int(), quant_z.shape)


        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)

        log["inputs"] = x
        log["inputs_masked"] = x * mask_in
        log["inputs_masked_reference"] = x * torch.round(mask_out_reference)
        log["reconstructions"] = xrec
        log["reconstructions_ref"] = xrec_ref

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


# DECODER MODEL
class RefinementAE(pl.LightningModule):
    '''
        Refinement model for maskgit transformer:
            refine given a recomposition of the inferred masked region and the original image
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
                 restriction=False,        # whether a partial encoder is used
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 second_stage_refinement=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.use_refinement = second_stage_refinement   

        # First Stage
        ##########################################
        # self.encoder = MatEncoder(**ddconfig)
        # self.encoder = MaskEncoder(**ddconfig)
        if restriction:
            ddconfig = dict(**ddconfig)
            ddconfig['conv_choice'] = Conv2dLayerPartial
            self.encoder = PartialEncoder(**ddconfig)
        else:
            self.encoder = MaskEncoder(**ddconfig)
        ##########################################
       
        self.decoder = Decoder(**ddconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # Second Stage
        if self.use_refinement:
            self.encoder_2 = MaskEncoder(**ddconfig)
            self.decoder_2 = Decoder(**ddconfig)
            self.bottleneck_conv_2 = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
            self.post_bottleneck_conv_2 = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss = instantiate_from_config(lossconfig)
        self.first_stage_model_type = first_stage_model_type

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        if first_stage_config is not None:
            self.init_first_stage_from_ckpt(first_stage_config, initialize_decoder=ckpt_path is None)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.mask_function = box_mask
        self.mask_lower = mask_lower
        self.mask_upper = mask_upper

        self.att = torch.nn.Conv2d(embed_dim, 1, 1)

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
        # if initialize_decoder:
        #     self.post_bottleneck_conv = copy.deepcopy(model.post_quant_conv)
        #     self.decoder = copy.deepcopy(model.decoder)

    def set_first_stage_model(self, model):
        self.first_stage_model = model

    def encode(self, x, mask):
        # encode the composited image
        h, mask_out = self.encoder(x, mask)
        h = self.bottleneck_conv(h)
        return h, mask_out

    def decode(self, h):
        h = self.post_bottleneck_conv(h)
        dec, _ = self.decoder(h)
        return dec

    def second_stage(self, x, mask):
        h, _ = self.encoder_2(x, mask)
        h = self.bottleneck_conv_2(h)
        h = self.post_bottleneck_conv_2(h)
        dec, _ = self.decoder_2(h)
        return dec

    def decode_at_layer(self, quant, i):
        quant = self.post_bottleneck_conv(quant)
        _, feat = self.decoder(quant, target_i_level = i)
        return feat

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, batch, quant=None, mask_in=None, mask_out=None, return_fstg=True, debug=False):

        input_raw = self.get_input(batch, self.image_key)
        input = input_raw * mask_in

        # first, get a composition of quantized reconstruction and the original image
        if mask_in is None:
            mask = self.get_mask([input.shape[0], 1, input.shape[2], input.shape[3]], input.device)
        else:
            mask = mask_in

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

        if quant is None:
            if self.first_stage_model_type == 'vae':
                quant, _, info = self.first_stage_model.encode(input_raw)
            else:
                quant = quant_fstg

        # _, _, codes = info
        B, C, H, W = quant.shape
        
        # x_in = torch.cat([mask - 0.5, x_comp], dim=1)

        if mask_out is None:
            h, mask_out = self.encode(input, mask)
        else:
            h, _ = self.encode(input, mask)

        w1 = torch.sigmoid(self.att(h))

        # TODO: this operation might be worth investigating (IMPORTANT!)
        # mask_out_interpolate = F.interpolate(mask, (16,16))
        
        h = mask_out * h + h * (1 - mask_out) * w1 + quant * (1 - mask_out) * (1 - w1)
        # h = mask_out * h + quant * (1 - mask_out) * 0.5 + h * (1 - mask_out) * 0.5

        dec = self.decode(h)

        # linear blending
        # k = 3
        # kernel = torch.ones(1,1,k,k) / k**2
        # pad = k // 2
        # smoothed_mask = F.conv2d(F.pad(mask,(pad,pad,pad,pad),value=1), kernel.to(mask.device), bias=None, padding=0)
        # dec = smoothed_mask * input + (1 - smoothed_mask) * dec
        dec = input + (1 - mask_in) * dec

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
        # p = random.uniform(self.mask_lower, self.mask_upper)
        # return box_mask(shape, device, p)
        return torch.from_numpy(BatchRandomMask(shape[0], shape[-1], hole_range=[0.1,0.4])).to(device)
        # return torch.from_numpy(BatchRandomMask(shape[0], shape[-1])).to(device)

    def get_mask_eval(self, shape, device):
        return box_mask(shape, device, 0.5, det=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))

        if self.use_refinement:
            xrec, mask, _, xfstg = self(batch, mask_in=mask_in, mask_out=mask_out)
        else:
            xrec, mask, _ = self(batch, mask_in=mask_in, mask_out=mask_out)
            xfstg = None

        if optimizer_idx == 0:
            # autoencode

            if self.use_refinement:
                aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step, reconstructions_fstg=xfstg,
                                                mask=mask, last_layer=self.get_last_layer(), split="train")
            else:
                aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            if self.use_refinement:
                discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step, reconstructions_fstg=xfstg,
                                                    mask=mask, last_layer=self.get_last_layer(), split="train")
            else:
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

        if self.use_refinement:
            params = list(params + 
                     list(self.encoder_2.parameters()) + 
                     list(self.decoder_2.parameters()) + 
                     list(self.bottleneck_conv_2.parameters()) + 
                     list(self.post_bottleneck_conv_2.parameters()))

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        if self.use_refinement:
            opt_disc = torch.optim.Adam(list(self.loss.discriminator.parameters()) + 
                                        list(self.loss.discriminator_2.parameters()),
                                        lr=lr, betas=(0.5, 0.9))
        else:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


    def configure_optimizers(self):
        lr = self.learning_rate
        return self.configure_optimizers_with_lr(lr)

    def get_last_layer(self):
        if self.use_refinement:
            return self.decoder_2.conv_out.weight
        else:
            return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        # We are always making assumption that the latent block is 16x16 here
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))

        if self.use_refinement:
            xrec, mask, xraw, xrec_fstg = self(batch, mask_in=mask_in, mask_out=mask_out)
        else:
            xrec, mask, xrec_fstg = self(batch, mask_in=mask_in, mask_out=mask_out)
        
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["masked_input"] = x * mask
        log['recon_fstg'] = xrec_fstg

        if self.use_refinement:
            log['recon_raw'] = xraw

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

class RefinementUNet(pl.LightningModule):
    '''
        Refinement model for maskgit transformer:
            refine a recomposed image
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
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 second_stage_refinement=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.use_refinement = second_stage_refinement   

        self.encoder = Encoder(**ddconfig)       
        self.decoder = Decoder(**ddconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss = instantiate_from_config(lossconfig)
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
        if initialize_current:
            self.encoder = copy.deepcopy(model.encoder)
            self.bottleneck_conv = copy.deepcopy(model.bottleneck_conv)
            self.post_bottleneck_conv = copy.deepcopy(model.post_quant_conv)
            self.decoder = copy.deepcopy(model.decoder)

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
                                                                        mask=mask_out, 
                                                                        det=False, 
                                                                        return_quant=True)    
        if self.first_stage_model_type == 'vae':
            x_raw, _ = self.first_stage_model(input_raw)

        x_comp = input + (1 - mask) * x_raw

        # forward pass with the recomposed image
        h = self.encode(x_comp)
        dec = self.decoder(h)
        dec = input + (1 - mask) * dec
        return dec, mask

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

    def get_mask(self, shape, device):
        # p = random.uniform(self.mask_lower, self.mask_upper)
        # return box_mask(shape, device, p)
        return torch.from_numpy(BatchRandomMask(shape[0], shape[-1], hole_range=[0.1,0.4])).to(device)
        # return torch.from_numpy(BatchRandomMask(shape[0], shape[-1])).to(device)

    def get_mask_eval(self, shape, device):
        return box_mask(shape, device, 0.5, det=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask, _ = self(batch, mask_in=mask_in, mask_out=mask_out)
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

        if self.use_refinement:
            params = list(params + 
                     list(self.encoder_2.parameters()) + 
                     list(self.decoder_2.parameters()) + 
                     list(self.bottleneck_conv_2.parameters()) + 
                     list(self.post_bottleneck_conv_2.parameters()))

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        if self.use_refinement:
            opt_disc = torch.optim.Adam(list(self.loss.discriminator.parameters()) + 
                                        list(self.loss.discriminator_2.parameters()),
                                        lr=lr, betas=(0.5, 0.9))
        else:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


    def configure_optimizers(self):
        lr = self.learning_rate
        return self.configure_optimizers_with_lr(lr)

    def get_last_layer(self):
        if self.use_refinement:
            return self.decoder_2.conv_out.weight
        else:
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
        
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["masked_input"] = x * mask
        log['recon_fstg'] = xrec_fstg

        if self.use_refinement:
            log['recon_raw'] = xraw

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
