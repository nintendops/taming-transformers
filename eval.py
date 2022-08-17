# eval testing

# load config for transformer
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import time
import cv2
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from taming.models.cond_transformer import Net2NetTransformer


def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


if __name__ == '__main__':
    device = torch.device('cuda')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # config_path = "configs/owt_nc_transformer.yaml"
    config_path = "configs/owt_transformer.yaml"
    save_path = os.path.join("logs/eval", now)
    os.makedirs(save_path, exist_ok=True)
    config = OmegaConf.load(config_path)
    config ['data']['params']['batch_size'] = 1
    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"
    
    # instantiate model
    print("Instantiating model...")
    model = Net2NetTransformer(**config.model.params).to(device)
    print("Done!")
    
    # loading checkpoint
    ckpt_path = "logs/2022-08-02T07-32-14_usc_512_transformer/checkpoints/last.ckpt"
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    # missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loading checkpoint from %s..."%ckpt_path)
    model.load_state_dict(sd)
    print("Done!")
    
    print("instantiating Dataset...")
    config.data.params.train.params.split = "validation"
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("Done!")
    dataset = data.datasets['train']
    dataset_iter = iter(data._train_dataloader())

    def eval(batch, image_idx, do_not_generate=False):
        print(f"Generating samples for batch data {image_idx}")
        image = batch['image']
        if unconditional:
            segmentation = image
        else:
            segmentation = batch['segmentation']
            write_images(os.path.join(save_path, f'image_{image_idx}_segmentation.png'), segmentation)
        write_images(os.path.join(save_path, f'image_{image_idx}_src.png'), image)

        if do_not_generate:
            return

        tensify = lambda x: torch.from_numpy(x[None]).permute(0,3,1,2).contiguous().float().to(device)
        tensor_to_numpy = lambda x:x.detach().cpu().numpy()[0].transpose(1,2,0)

        seg_tensor = tensify(segmentation)
        c_code, c_indices = model.encode_to_c(seg_tensor)
        
        if not unconditional:
            seg_rec = model.cond_stage_model.decode(c_code)
            seg_rec = F.softmax(seg_rec,dim=1)

        nb = c_code.shape[0]
        codebook_size = config.model.params.first_stage_config.params.embed_dim
        res = 256
        c_code_res = int(res**0.5)
        z_indices_shape = [nb,res]
        z_code_shape = [nb, codebook_size,c_code_res, c_code_res]

        # using random codebook entries
        z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)

        # debug testing: replace random z indices with that of an encoded batch image
        # x = tensify(batch['image'])
        # c = x
        # _, z_indices = model.encode_to_z(x)
        # _, c_indices = model.encode_to_c(c)


        if not unconditional:
            cidx = c_indices
            cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

        temperature = 1.0
        top_k = 100
        update_every = 100000
        start_t = time.time()
        start_i = 0
        start_j = 0
        n_sample = 4
        for sample in range(n_sample):        
            idx = z_indices
            idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])
            for i in range(start_i, z_code_shape[2]-0):
              if i <= 8:
                local_i = i
              elif z_code_shape[2]-i < 8:
                local_i = 16-(z_code_shape[2]-i)
              else:
                local_i = 8
              for j in range(start_j, z_code_shape[3]-0):
                if j <= 8:
                  local_j = j
                elif z_code_shape[3]-j < 8:
                  local_j = 16-(z_code_shape[3]-j)
                else:
                  local_j = 8
                
                i_start = i-local_i
                i_end = i_start+16
                j_start = j-local_j
                j_end = j_start+16
                # print(i_start, i_end, j_start, j_end)
                patch = idx[:,i_start:i_end,j_start:j_end]
                patch = patch.reshape(patch.shape[0],-1)
                if unconditional:
                    patch = torch.cat((c_indices, patch), dim=1)
                    logits,_ = model.transformer(patch[:,:-1])
                else:
                    cpatch = cidx[:, i_start:i_end, j_start:j_end]
                    cpatch = cpatch.reshape(cpatch.shape[0], -1)
                    # patch = torch.cat((cpatch, patch), dim=1)
                    logits,_ = model.transformer(patch[:,:-1], cpatch)

                logits = logits[:, -256:, :]
                logits = logits.reshape(z_code_shape[0],16,16,-1)
                logits = logits[:,local_i,local_j,:]
                logits = logits/temperature
                
                if top_k is not None:
                  logits = model.top_k_logits(logits, top_k)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx[:,i,j] = torch.multinomial(probs, num_samples=1)
                step = i*z_code_shape[3]+j
                if step==0 or step==z_code_shape[2]*z_code_shape[3]-1:
                  x_sample = model.decode_to_img(idx, z_code_shape)
                  print(f"Time: {time.time() - start_t} seconds")
                  print(f"Sample: {sample} | Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
                  write_images(os.path.join(save_path, f"image_{image_idx}_sample_{sample}.png"), tensor_to_numpy(x_sample))

    data_select = [5,13,16,20,23,26,28] # range(len(dataset))

    for i in data_select:
        batch_data = dataset[i]
        eval(batch_data, i, False)

