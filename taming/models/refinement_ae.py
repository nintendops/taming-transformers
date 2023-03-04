import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, RestrictedDecoder
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

# mask functions
def box_mask(r, mr, nb, device):
    assert mr <= r
    mask = np.ones([nb, 1, r, r]).astype(np.int32)
    for i in range(nb):
        h, w = np.random.randint(0, r-mr, 2)
        mask[i, :, h:h+mr, w:w+mr] = 0
    return torch.from_numpy(mask).to(device)

class RefinementAE(pl.LightningModule):
    def __init__(self,
                 first_stage_config,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
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
        decoder_model = RestrictedDecoder if restriction else Decoder
        self.init_first_stage_from_ckpt(first_stage_config)
        self.encoder = Encoder(**ddconfig)
        self.decoder = decoder_model(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        mask = None

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
        model = model.eval()
        model.train = disabled_train
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

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, batch, mask=None, composition=True):

        input = self.get_input(batch, self.image_key)

        # first, get a composition of quantized reconstruction and the original image
        if mask is None:
            mask = self.get_mask([input.shape[0], 1, input.shape[2], input.shape[3]], input.device)
        
        # enable this if first stage model is an VQ autoencoder
        # x_fstg, _ = self.first_stage_model(input)
        
        # enable this if first stage model is a maskGIT transformer
        x_fstg = self.first_stage_model.forward_with_recon(batch, mask=mask)
        x_comp = mask * input + (1 - mask) * x_fstg

        h = self.encode(x_comp)
        dec = self.decode(h)
        
        if composition:
            dec = mask * input + (1 - mask) * dec

        return dec, mask, x_comp

    def refine(self, x, mask=None):
        dec = self.decode(self.encode(x))
        if mask is not None:
            return mask * x + (1-mask) * dec
        else:
            return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

    def get_mask(self, shape, device):
        B, _, H, W = shape
        return box_mask(H, H//2, B, device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, mask, _ = self(batch)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, mask, _ = self(batch)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
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
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
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

