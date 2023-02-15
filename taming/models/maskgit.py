import os, math
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config
from taming.modules.util import SOSProvider


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MaskGIT(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 cond_stage_config,
                 first_stage_config,
                 refinement_stage_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 mask_function="random_mask",
                 downsample_cond_size=-1,
                 pkeep=0.5,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.mask_token = -1 # this needs to be hard-coded due to embedding implementation in BERT
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_cond_stage_from_ckpt(cond_stage_config)

        if refinement_stage_config is None:
            self.init_first_stage_from_ckpt(first_stage_config)
        else:
            self.init_full_stages_from_ckpt(refinement_stage_config)

        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        self.use_condGPT = self.transformer.__class__.__name__ == 'CondGPT'
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

        # todo: remove hard-coded mapping
        self.mask_function = self.box_mask

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

    def init_full_stages_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.second_stage_model = model
        self.first_stage_model = model.first_stage_model


    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def scatter_mask(self, z_indices, p=None):

        p = self.pkeep if p is None else p
        assert p <= 1 and p >= 0

        mask = torch.bernoulli( p *torch.ones(z_indices.shape,
                                                     device=z_indices.device))
        mask = mask.round().to(dtype=torch.int64) 
        return mask

    def box_mask(self, z_indices, p=None):

        p = self.pkeep if p is None else p
        assert p <= 1 and p >= 0

        nb = z_indices.shape[0]
        r = int(z_indices.shape[-1]**0.5)
        mr = int((1-p) * r)
        mask = np.ones([nb, r, r]).astype(np.int32)
        for i in range(nb):
            h, w = np.random.randint(0, r-mr, 2)
            mask[i, h:h+mr, w:w+mr] = 0
        mask = torch.from_numpy(mask).reshape(nb, -1).to(z_indices.device)
        return mask

    def forward(self, x, c, mask=None):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x.float())
        _, c_indices = self.encode_to_c(c.float())

        # if during training, we'll mask out some tokens (0.15 by default)
        assert not self.training or self.pkeep < 1.0 

        if self.training and self.pkeep < 1.0:
            # mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
            #                                              device=z_indices.device))
            # mask = mask.round().to(dtype=torch.int64) 
            mask = self.mask_function(z_indices)
            # !!! replacing masked indices with the [MASK] token (a.k.a -1)
            r_indices = torch.full_like(z_indices, self.mask_token)
            a_indices = mask*z_indices+(1-mask)*r_indices           
        else:
            # during inference, mask is provided as input
            assert mask is not None
            a_indices = z_indices

        a_indices = a_indices + 1 # adding one to the indices as MASK token is set to -1

        # from this point we are making predictions on tokens where MASK==0 
        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        logits, _ = self.transformer(cz_indices)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # cut off conditioning outputs (as well as the mask label)
        logits = logits[:, c_indices.shape[1]:, 1:]
        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):

        # indices padding
        x = x + 1
        mask = (x == 0) # true: needs update, false: already updated

        if not self.use_condGPT:
            x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning

            # break if we have no candidate left
            if mask.sum() == 0:
                # print(f"sampling complete at step {k}/{steps}")
                break

            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            if self.use_condGPT:
                logits, _ = self.transformer(x_cond, c)
            else:
                logits, _ = self.transformer(x_cond)

            # pluck the logits at the final step and scale by temperature              
            logits = logits[:, c.shape[1]:, 1:] / temperature

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # prune low confidence prediction (we are taking the top% confidence score ones and remove the rest)
            B, L, D = probs.shape

            k = min(max(L // steps, 4), mask.sum()//B + 1)

            # gathering k most confident samples
            max_probs, _ = torch.max(probs * mask[...,None], -1)
            # B, k
            k_candidates = torch.argsort(max_probs, 1, descending=True)[:,:k]
            # B, L, D -> B, k, D
            gathered_probs = torch.gather(probs, 1, k_candidates[...,None].expand(-1,-1,D)) 

            if sample:
                gathered_idx = torch.multinomial(gathered_probs.reshape(-1,D),num_samples=1).reshape(B,k)
            else:
                _, gathered_idx = torch.topk(gathered_probs, k=1, dim=-1)
                gathered_idx = gathered_idx[..., -1]

            # updating mask and indices
            for i in range(B):
                mask[i, k_candidates[i]] = False
                x[i, (k_candidates[i]+1)] = gathered_idx[i] + 1

            # append to the sequence and continue
            # x = torch.cat((x, ix), dim=1)

        # cut off conditioning
        if not self.use_condGPT:
            x = x[:, c.shape[1]:]
        return x - 1

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if self.use_condGPT or len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)           
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape, use_softmax=False):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)

        x = self.first_stage_model.decode(quant_z)

        if use_softmax or self.first_stage_key != 'image':
            x = F.softmax(x, dim=1)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, composition=True, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)

        x = x.to(device=self.device).float()
        c = c.to(device=self.device).float()

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        B, C, H, W = x.shape
        gH, gW = quant_z.shape[2:]

        # create a "half"" sample
        mask = torch.ones(z_indices.shape, device=z_indices.device)
        mask[:, z_indices.shape[1]//2:] = 0
        mask = mask.to(dtype=torch.int64)
        image_mask = torch.nn.functional.interpolate(mask.reshape(B,1,gH,gW).float(), scale_factor=H//gH)

        r_indices = torch.full_like(z_indices, self.mask_token)
        z_start_indices = mask*z_indices+(1-mask)*r_indices      

        index_sample = self.sample(z_start_indices, c_indices,
                                   steps= z_indices.shape[1]//2,
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_half = self.decode_to_img(index_sample, quant_z.shape)

        # composition
        if composition:
            x_sample_half = image_mask * x + (1 - image_mask) * x_sample_half
            if self.second_stage_model is not None:
                # x_sample_half = self.second_stage_model.refine(x_sample_half, image_mask)
                x_sample_half, _, _ = self.second_stage_model(x, mask=image_mask)

        # inpainting sample
        # pkeep = 0.85
        # mask = torch.bernoulli(pkeep*torch.ones(z_indices.shape, device=z_indices.device))
        # mask = mask.round().to(dtype=torch.int64) 

        mask = self.mask_function(z_indices, p=0.5)
        image_mask = torch.nn.functional.interpolate(mask.reshape(B,1,gH,gW).float(), scale_factor=H//gH)

        r_indices = torch.full_like(z_indices, self.mask_token)
        z_start_indices = mask*z_indices+(1-mask)*r_indices      
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # composition
        if composition:
            x_sample_nopix = image_mask * x + (1 - image_mask) * x_sample_nopix
            if self.second_stage_model is not None:
                x_sample_nopix = self.second_stage_model.refine(x_sample_nopix, image_mask)

        # det inpainting sample
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # composition
        if composition:
            x_sample_det = image_mask * x + (1 - image_mask) * x_sample_det
            if self.second_stage_model is not None:
                x_sample_det = self.second_stage_model.refine(x_sample_det, image_mask)

        x_masked = image_mask * x

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.be_unconditional:
            pass
        elif self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample_half
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        log["inputs_masked"] = x_masked

        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    # loss function defined here
    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
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
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer