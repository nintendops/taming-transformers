import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import random
from main import instantiate_from_config
from taming.modules.util import write_images, scatter_mask, box_mask, mixed_mask, BatchRandomMask

training_stages = {'vq', 
                   'encoder_refinement', 
                   'decoder_refinement', 
                   'maskgit', 
                   'final',}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def make_eval(model):
    model.eval()
    model.train = disabled_train

def InpaintingMaster(pl.LightningModule):
    def __init__(self,
                 stage,
                 vqmodel_config,
                 encoder_config=None,
                 decoder_config=None,
                 transformer_config=None,
                 encoder_choice='vq',
                 image_key="image",
                 ):
    '''
        vq stage: only need the VQGAN model
        encoder_refinement stage: need both a pretrained VQGAN model and an encoder model
        decoder_refinement stage: need an encoder model (VQGAN or refined) and a decoder model
        maskgit stage: need an encoder model (VQGAN or refined) and a maskgit transformer model
        final stage: need the encoder, decoder and transformer models (end-to-end finetuning)
    '''
        super().__init__()
        assert stage in training_stages

        self.stage = stage
        self.image_key = image_key
        self.encoder_choice = encoder_choice

        self.Encoder = None
        self.Decoder = None
        self.Transformer = None
        self.current_model = None

        # We always need a vq model in every stage
        # Initialize the VQ model
        self.VQModel = instantiate_from_config(vqmodel_config)
        if stage != 'vq':
            make_eval(self.VQModel)

        # Next, instantaite the other models as necessary
        if self.stage == 'encoder_refinement' or encoder_choice == 'refined':
            assert encoder_config is not None
            self.Encoder = instantiate_from_config(encoder_config)
            self.Encoder.set_first_stage_model(self.VQModel)
        if self.stage != 'encoder_refinement':
            if encoder_choice == 'vq':
                self.Encoder = self.VQModel
            else:
                make_eval(self.Encoder)
        if self.stage == 'decoder_refinement' or self.stage == 'final':
            assert decoder_config is not None
            self.Decoder = instantiate_from_config(decoder_config)
            self.Decoder.set_first_stage_model(self.Encoder)
        if self.stage == 'maskgit' or self.stage == 'final':
            assert transformer_config is not None
            self.Transformer = instantiate_from_config(transformer_config)
            self.Transformer.set_first_stage_model(self.Encoder)

        # Finally, set the current training model
        if self.stage == 'vq':
            self.current_model = self.VQModel
        if self.stage == 'encoder_refinement':
            self.current_model = self.Encoder
        if self.stage == 'decoder_refinement':
            self.current_model = self.Decoder
        if self.stage == 'maskgit':
            self.current_model = self.Transformer

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def forward(self, batch, mask=None, use_vq_decoder=False, simple_return=True):
        '''
            forward with the complete model
        '''

        x = self.get_input(self.image_key, batch)       

        if mask is None:
            # large-hole random mask following MAT
            mask = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)

        # masking image first
        x = mask * x

        # encoding image
        quant_z, _, info, mask_out = self.Encoder.encode(x, mask)
        mask_out = mask_out.reshape(x.shape[0], -1)
        z_indices = info[2].reshape(x.shape[0], -1)

        # inferring missing codes given the downsampled mask
        z_indices_complete = self.Transformer.forward_to_indices(batch, z_indices, mask_out)

        # getting the features from the codebook with the indices
        B, C, H, W = quant_z.shape
        quant_z_complete = self.VQModel.quantize.get_codebook_entry(z_indices_complete.reshape(-1), shape=(B, H, W, C))

        # decoding the features 
        if use_vq_decoder:
            return self.VQModel.decode(quant_z_complete)
        else:
            dec, _ = self.Decoder(batch, quant=quant_z_complete, mask_in=mask, mask_out=mask_out.reshape(B, 1, H, W))

        if simple_return:
            return dec, mask
        else:
            return dec, z_indices, z_indices_complete, mask, mask_out

    # Note: for now, the final stage only trains the refined decoder assuming we have everything else pretrained
    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.stage == 'vq' or self.stage == 'decoder_refinement':
            return self.current_model.training_step(batch, batch_idx, optimizer_idx)
        elif self.stage == 'encoder_refinement' or self.stage == 'maskgit':
            return self.current_model.training_step(batch, batch_idx)
        else:
            x = self.get_input(self.image_key, batch)
            xrec, mask = self(batch)

            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.Decoder.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.Decoder.get_last_layer(), split="train")
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.Decoder.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.Decoder.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss



    def validation_step(self, batch, batch_idx):
        if self.stage != 'final':
            return self.current_model.validation_step(batch, batch_idx)
        else:
            x = self.get_input(batch, self.image_key)
            xrec, mask = self(batch)
            aeloss, log_dict_ae = self.Decoder.loss(x, xrec, 0, self.global_step,
                                                mask=mask, last_layer=self.Decoder.get_last_layer(), split="val")
            discloss, log_dict_disc = self.Decoder.loss(x, xrec, 1, self.global_step,
                                                mask=mask, last_layer=self.Decoder.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]
            self.log("val/rec_loss", rec_loss,
                       prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/aeloss", aeloss,
                       prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict

    def configure_optimizers(self):
        if self.stage != 'final':
            return self.current_model.configure_optimizers()
        else:
            return self.Decoder.configure_optimizers()

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        if self.stage != 'final':
            return self.current_model.log_images(batch, **kwargs)
        else:
            # todo
            pass



