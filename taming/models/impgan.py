import os, math
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from taming.modules.diffusionmodules.model import mlpDecoder, Decoder
from taming.modules.util import SOSProvider
from main import instantiate_from_config

import kornia
import torch.distributions as dists

def flip_positions(positions):
    positions = positions.transpose(2,3)
    positions = torch.cat([positions[:,0][:,None],-positions[:,1][:,None]],1)
    return positions.contiguous()


def get_position(size, dim, device, batch_size):
    height, width = size
    aspect_ratio = width / height
    position = kornia.utils.create_meshgrid(width, height, device=torch.device(device)).permute(0, 3, 1, 2)
    position[:, 1] = -position[:, 1] * aspect_ratio  # flip y axis

    if dim == 1:
        x, y = torch.split(position, 1, dim=1)
        position = x
    if dim == 3:
        x, y = torch.split(position, 1, dim=1)
        z = torch.ones_like(x) * torch.rand(1, device=device) * 2 - 1
        a = torch.randint(0, 3, (1,)).item()
        if a == 0:
            xyz = [x, y, z]
        elif a == 1:
            xyz = [z, x, y]
        else:
            xyz = [x, z, y]
        # xyz =  [x,z,y]
        position = torch.cat(xyz, dim=1)
    position = position.expand(batch_size, dim, width, height)
    return position if dim == 3 else flip_positions(position)

def get_distribution_type(shape, type='uniform'):
    if type == 'normal':
        return dists.Normal(torch.zeros(shape), torch.ones(shape))
    elif type == 'uniform':
        return dists.Uniform(torch.zeros(shape) - 1, torch.ones(shape))
    else:
        raise NotImplementedError

def get_cropped_coord_grid(device, 
                           dist_shift, 
                           nb,  
                           res=256, 
                           crop_res=128, 
                           scale=2.0,
                           det=False):

    coord_grid = scale * get_position([res, res], 2, device, nb)
    # dist_shift = get_distribution_type([1,2], type='uniform')
    coord_grid_sample = torch.zeros([nb, 2, crop_res, crop_res]).to(device)
    shift = 0.5 * (dist_shift.sample([nb]) + 1) * (res - crop_res)
    shift = shift.int().reshape(nb,2)

    if det:
        shift = shift * 0

    for idx, s in enumerate(shift):
        dx,dy = s
        dx = int(dx.item())
        dy = int(dy.item())
        coord_grid_sample[idx] = coord_grid[idx, :, dx:dx+crop_res, dy:dy+crop_res]

    return coord_grid_sample, shift

def crop_input(device, x, shift, nb, crop_res=128):
    new_x = torch.zeros([nb, 3, crop_res, crop_res]).to(device)
    for idx, s in enumerate(shift):
        dx, dy = s
        dx = int(dx.item())
        dy = int(dy.item())       
        new_x[idx] = x[idx,:,dx:dx+crop_res,dy:dy+crop_res]
    return new_x

def batched_index_select_2d(index, feature):
    batch_size, image_dim, res1, res2 = index.shape
    _, f_dim, grid_dim, _ = feature.shape
    indexf = index.reshape(batch_size,image_dim,-1)
    indexf2 = indexf[:,0] * grid_dim + indexf[:,1]
    def select(x, idx):
        x_select = torch.index_select(x.reshape(f_dim, -1),1,idx)
        x_select = x_select.reshape(f_dim, res1, res2)
        return x_select[None]
    xyz_select = torch.cat([select(x,idx) for x,idx in zip(feature, indexf2)], dim=0)
    return xyz_select

def stationary_noise(positions, feature, scale=2.0, sigma=0.2, mode='gaussian'):
    '''
    positions: nb,2,w,h
    codebook feature: nb,nc,c_w,c_h (assume to be defining the feature space in [-scale, scale])
    '''
    # cgs_q: coordinate_grid_sampled_quantized (according to codebook resolution)
    c_res = feature.shape[2]
    cgs_q = (positions + scale) * c_res / (2 * scale)
    
    # index-select from features    
    idx_1 = torch.clamp(torch.floor(cgs_q).long(), 0, c_res - 1)
    idx_2 = torch.clamp(idx_1 + 1, 0, c_res - 1)
    idx_3 = torch.clamp(torch.cat([idx_1[:,0].unsqueeze(1) + 1, idx_1[:,1].unsqueeze(1)],1), 0, c_res-1)
    idx_4 = torch.clamp(torch.cat([idx_1[:,0].unsqueeze(1), idx_1[:,1].unsqueeze(1)+1],1), 0, c_res-1)

    # distance to corners
    px = cgs_q[:,0]
    py = cgs_q[:,1]
    x1 = (px - torch.floor(px))
    x2 = (torch.floor(px) + 1 - px)
    y1 = (py - torch.floor(py))
    y2 = (torch.floor(py) + 1 - py)  
    
    if mode == 'gaussian':
        f_grouped = torch.cat([batched_index_select_2d(index, feature)[...,None] \
                    for index in [idx_1,idx_2,idx_3,idx_4]],dim=-1)
        dist1 = torch.sqrt(x1**2 + y1**2)[...,None]
        dist2 = torch.sqrt(x2**2 + y2**2)[...,None]
        dist3 = torch.sqrt(x2**2 + y1**2)[...,None]
        dist4 = torch.sqrt(x1**2 + y2**2)[...,None]
        dists = torch.cat([dist1,dist2,dist3,dist4], 3)
        dists = torch.nn.functional.softmax(-dists/sigma, -1).unsqueeze(1)
        return torch.sum(dists * f_grouped, -1)
    elif mode == 'bilinear' or mode == 'linear':
        tr,tl,br,bl = [batched_index_select_2d(index, feature)\
                    for index in [index_1,index_3,index_4,index_2]]
        bx = x1.unsqueeze(1) * bl + x2.unsqueeze(1) * br
        tx = x1.unsqueeze(1) * tl + x2.unsqueeze(1) * tr        
        return y1.unsqueeze(1) * bx + y2.unsqueeze(1) * tx        
    else:
        raise NotImplementedError(f"type of interpolation {mode} is not recognized!")

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# x -> [cos(2^0 pi x), ..., sin(2^9) pi x]
# b,dim,... -> b,2*dim*l,...
def positional_encoding(x, l=5, beta=None):
    bs,dim = x.shape[:2]
    res = x.shape[2:]
    x = x.unsqueeze(2).expand(bs,dim,l,*res)
    octaves = 2**(torch.arange(l)).to(x.device)
    if beta is not None:        
        octaves = octaves * beta
    for r in res:
        octaves = octaves.unsqueeze(-1)
    x = x * octaves[None,None,...] * np.pi
    x = torch.cat((torch.sin(x).unsqueeze(2), torch.cos(x).unsqueeze(2)),2)
    return x.reshape(bs,-1,*res)


class ImplicitDecoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 mlp_config,
                 first_stage_config,
                 lossconfig,
                 crop_res,
                 shift_scale = 2.0,
                 ckpt_path=None,
                 image_key="image",
                 ignore_keys=[],
                 ):
        super().__init__()

        self.crop_res = crop_res
        self.shift_scale = shift_scale
        self.dist_shift = get_distribution_type([1,2], type='uniform')
        
        self.mlp = mlpDecoder(**mlp_config)

        self.loss = instantiate_from_config(lossconfig)
        self.image_key = image_key

        self.init_first_stage_from_ckpt(first_stage_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    @torch.no_grad()
    def encode_to_z(self, x):
        layer_i = 3
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], quant_z.shape[2], quant_z.shape[3])
        quant_z = self.first_stage_model.decode_at_layer(quant_z, layer_i)
        return quant_z, indices

    def decode(self, quant, res, expansion=2.0, det=False):
        # prepare input coordinates
        cgs = self.shift_scale * expansion * get_position([res, res], 2, quant.device, quant.shape[0])
        # feature sampling
        layer_i = 3
        quant_z = self.first_stage_model.decode_at_layer(quant, layer_i)
        feat = stationary_noise(cgs, quant_z, scale=scale)  
        fourier = positional_encoding(cgs)
        feat = torch.cat([fourier, feat], 1)
        dec = self.mlp(feat)
        return dec

    def decode_with_shift(self, quant, x_size, det=False):
        # prepare input coordinates
        scale = self.shift_scale
        # random cropping to model stationary shift
        cgs, shift = get_cropped_coord_grid(quant.device, 
                                            self.dist_shift, 
                                            quant.shape[0], 
                                            x_size, 
                                            self.crop_res, 
                                            scale,
                                            det=det)

        ######### DEBUG SETTING ###########################
        # cgs = scale * get_position([256, 256], 2, x.device, x.shape[0])
        # cropped_x = x
        ###################################################

        # # feature sampling
        feat = stationary_noise(cgs, quant, scale=scale)  
        fourier = positional_encoding(cgs)
        feat = torch.cat([fourier, feat], 1)
        dec = self.mlp(feat)
        return dec, shift

    def forward(self, x):
        quant, _ = self.encode_to_z(x)
        # decoding
        dec, shift = self.decode_with_shift(quant, x.shape[2])
        cropped_x = crop_input(quant.device, x, shift, quant.shape[0], crop_res=self.crop_res)
        return dec, cropped_x

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, x = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def get_input(self, batch, k):
        return self.first_stage_model.get_input(batch, k)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, x = self(x)
        qloss = torch.tensor([0.0]).to(x.device)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
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
        xrec, x = self(x)
        qloss = torch.tensor([0.0]).to(x.device)
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
        xrec, x = self(x)
        qloss = torch.tensor([0.0]).to(x.device)

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

    def get_last_layer(self):
        return self.mlp.last_conv.layer.weight
        # return self.mlp.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, x = self(x)
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

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.mlp.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
