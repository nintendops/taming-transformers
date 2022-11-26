import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
from main import instantiate_from_config
from taming.modules.transformer.bert import Transformer
from taming.modules.diffusionmodules.model import Encoder, Decoder, RestrictedDecoder

def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

def to_2d(tensor):
    assert len(tensor.shape) == 4
    B, C, H, W = tensor.shape
    return tensor.reshape(B, C, H*W).permute(0,2,1).contiguous()

def to_4d(tensor, image_dim=None):
    assert len(tensor.shape) == 3
    B, L, C = tensor.shape
    if image_dim is None:
        image_dim = [int(L**0.5)] * 2
    return tensor.permute(0,2,1).reshape(B,C,*image_dim).contiguous()


class ATTVQModel(pl.LightningModule):
    def __init__(self,
                 bertconfig,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 finetune=False,
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
        self.encoder = Encoder(**ddconfig)
        self.decoder = decoder_model(**ddconfig)

        # embeddings (to_tensor) for the simplex attention 
        self.embeddings = torch.nn.Embedding(n_embed, embed_dim)

        # BERT transformer with simplex attention
        self.transformer = Transformer(**bertconfig, finetune=finetune)

        # Perception loss only 
        self.loss = instantiate_from_config(lossconfig)

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.finetune = finetune
        if self.finetune:
            self.encoder.eval()
            self.quant_conv.eval()


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

    def encode(self, x, use_topk=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        image_dim = h.shape[-2:]
        quant, att_probs, qloss = self.transformer(from_tensor=to_2d(h),
                                            to_tensor=self.embeddings.weight,
                                            from_pos=None,
                                            to_pos=None,
                                            use_topk=use_topk)
        quant = to_4d(quant, image_dim)
        return quant, att_probs, qloss

    def decode(self, quant, cast_to_4d=False):
        if cast_to_4d or len(quant.shape) == 3:
            quant = to_4d(quant)
        quant = self.post_quant_conv(quant)
        dec, _ = self.decoder(quant)
        return dec

    def decode_at_layer(self, quant, i):
        quant = self.post_quant_conv(quant)
        _, feat = self.decoder(quant, target_i_level = i)
        return feat

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def query(self, probs):
        probs = probs.unsqueeze(1) # assuming n_heads = 1
        quant = self.transformer.query(self.embeddings.weight, probs)
        return quant

    def forward(self, input):
        quant, probs, loss = self.encode(input)
        dec = self.decode(quant)
        return dec, probs, loss

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, probs, qloss = self(x)

        # self.transformer.freeze()

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/qloss", qloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, probs, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
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
        xrec, probs, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
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
        if self.finetune:
            opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                      list(self.embeddings.parameters())+
                                      list(self.transformer.parameters())+
                                      list(self.post_quant_conv.parameters()),
                                      lr=lr, betas=(0.5, 0.9))
        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                      list(self.decoder.parameters())+
                                      list(self.embeddings.parameters())+
                                      list(self.transformer.parameters())+
                                      list(self.quant_conv.parameters())+
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
        xrec, _, _ = self(x)
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
