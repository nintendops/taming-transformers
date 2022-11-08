import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config
from taming.modules.util import SOSProvider
from taming.models.cond_transformer import Net2NetTransformer


def cross_entropy(logits, target):
    return torch.mean(-torch.sum(target * F.log_softmax(logits, dim=1), 1))

class AttTransformer(Net2NetTransformer):

    # B, C, H, W -> B, L, C
    def prepare_input(self, embeddings):
        return embeddings.reshape(*embeddings.shape[:2], -1).permute(0,2,1).contiguous()

    # ignore c for now
    def forward(self, x, c):
        '''
        quant_c: B, C, H, W
        probs: B, H*W (L), K
        '''
        
        # one step to produce the logits
        quant, z_probs = self.encode_to_z(x.float())
        # _, c_indices = self.encode_to_c(c.float())
        quant = self.prepare_input(quant) # B, L, C
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones([quant.shape[0],quant.shape[1],1],
                                                         device=z_probs.device))
            mask = mask.round().float()
            a_quant = mask*quant
        else:
            a_quant = quant

        # make the prediction: B, L, C -> B, L, K
        logits, _ = self.transformer(a_quant[:, :-1])
        target = z_probs[:,1:]

        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        # logits = logits[:, c_indices.shape[1]-1:]
        
        return logits, target

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, probs, _ = self.first_stage_model.encode(x)
        B, C, H, W = quant_z.shape
        probs = probs.reshape(B, H*W, -1)
        return quant_z, probs

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if self.use_condGPT or len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)           
        return quant_c, indices


    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
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

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)

        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

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

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log


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
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            if self.use_condGPT:
                noise = c.clone()[:,:-1]
                x = torch.cat((x,noise),dim=1)
                logits, _ = self.transformer(x, c)
            else:
                noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
                x = torch.cat((x,noise),dim=1)
                logits, _ = self.transformer(x)

            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            if not self.use_condGPT:
                x = ix[:, c.shape[1]-1:]

        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            logits, _ = self.transformer(x_cond)

            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        # cut off conditioning
        if not self.use_condGPT:
            x = x[:, c.shape[1]:]

        return x


    # loss function defined here
    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1, target.size(-1)))
        return loss
