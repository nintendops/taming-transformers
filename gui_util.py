import io
import os
import numpy as np
import albumentations
import yaml
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from omegaconf import OmegaConf

GPU_index = 2
DISABLE_MODEL_LOADING = False
device = torch.device(f'cuda:{GPU_index}')

def draw_at(window, image, idx="0_0"):
    if window is None:
        return
    # write_images(f"gen_{idx}.png", image)
    image = np.uint8(127.5 * (image + 1.0))
    graph = window["-GRAPH-"]  
    image = Image.fromarray(np.uint8(image))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    graph.draw_image(data=bio.getvalue(), location=(0, 800))
    graph.update()
    window.refresh()

def tensify(x, device=device):
    return torch.from_numpy(x[None]).to(device).permute(0,3,1,2).contiguous().float()

def tensor_to_numpy(x):
    return x.detach().cpu().numpy()[0].transpose(1,2,0)

def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

def convert_to_onehot(segmentation, n_labels=3):
    flatseg = np.ravel(segmentation)
    onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
    onehot[np.arange(flatseg.size), flatseg] = True
    onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
    return onehot.astype(np.float32)

def process_image_expansion(coords, image, seg):
    pass


def process_image(coords, image, seg, filter_label, replace_label, expansion=False):
    if image is None or seg is None:
        return
   
    image = np.array(image).astype(np.uint8)

    if expansion:
        mask = mask_2 = None      
        seg = seg.astype(np.uint8)
        rescaler = albumentations.Compose(
                   [albumentations.SmallestMaxSize(max_size=512)],
                   additional_targets={'segmentation':'image'})
        processed = rescaler(image=image, segmentation=seg)
        image, seg = processed["image"], processed["segmentation"]
        seg = seg.astype(np.int8)
        w, h = image.shape[:2]
        sy, sx, ey, ex = coords
        sx_i = int(w - max(sx,ex) * w)
        ex_i = min(sx_i + 256, w)
        sy_i = int(min(sy,ey) * h)
        ey_i = min(sy_i + 256, h)
        ex_i_seg = min(sx_i + 512, w)
        ey_i_seg = min(sy_i + 512, h)
        masked_image = np.zeros([512,512,3], dtype=np.uint8)
        masked_image[:ex_i-sx_i, :ey_i-sy_i] = image[sx_i:ex_i, sy_i:ey_i]
        new_seg = np.zeros([512,512], dtype=np.int8)
        new_seg[:ex_i_seg-sx_i, :ey_i_seg-sy_i] = seg[sx_i:ex_i_seg, sy_i:ey_i_seg]
    else:
        w, h = image.shape[:2]
        sy, sx, ey, ex = coords
        sx_i = int(w - max(sx,ex) * w)
        ex_i = int(w - min(sx,ex) * w)
        sy_i = int(min(sy,ey) * h)
        ey_i = int(max(sy,ey) * h)
        mask = (seg != filter_label)
        mask_2 = np.ones_like(mask).astype(bool)
        mask_2[sx_i:ex_i, sy_i:ey_i] = False
        mask = np.logical_or(mask, mask_2)
        new_seg = seg * mask + np.logical_not(mask) * replace_label 

        ####################################
        # mask erosion
        # kernel = np.ones((10, 10), np.uint8)
        # mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=8)
        ###################################

        masked_image = image * mask_2[...,None]

    return masked_image, new_seg, mask_2

def prepare_input_expansion(image, seg):
    image = np.array(image).astype(np.uint8)
    segmentation = seg.astype(np.uint8)
    w, h = image.shape[:2] 
    image = (image / 127.5 - 1.0).astype(np.float32)
    segmentation = convert_to_onehot(segmentation)
    return [image, segmentation]

def prepare_input(coords, image, seg, mask):
    image = np.array(image).astype(np.uint8)
    seg = seg.astype(np.uint8)
    mask = mask.astype(np.uint8)
    w, h = image.shape[:2] 
    rescaler = 512/min(w, h)
    sy, sx, ey, ex = coords
    sx_i = int((w - max(sx,ex) * w) * rescaler)
    ex_i = int((w - min(sx,ex) * w) * rescaler)
    sy_i = int(min(sy,ey) * h * rescaler)
    ey_i = int(max(sy,ey) * h * rescaler)

    res = 512 # 256
    sx_i_end = min(int(w*rescaler), sx_i + res)
    sy_i_end = min(int(h*rescaler), sy_i + res)
    new_sy = int(800 * (sy_i_end - res) / (rescaler * h))
    new_sx = int(800 * (1-(sx_i_end - res) / (rescaler * w)))

    preprocessor = albumentations.Compose(
        [albumentations.SmallestMaxSize(max_size=512),
         albumentations.Crop(sy_i_end - res, sx_i_end - res, sy_i_end, sx_i_end)
        ],
        additional_targets={'segmentation':'image',
                            'mask':'image'})
    processed = preprocessor(image=image, segmentation=seg, mask=mask)
    image, segmentation = processed["image"], processed["segmentation"]
    image = (image / 127.5 - 1.0).astype(np.float32)
    segmentation = convert_to_onehot(segmentation)

    return [image, segmentation, processed['mask']], (new_sy, new_sx)

def load_config():
    config_path = "logs/2022-08-18T17-17-56_usc_512_pf_transformer/configs/2022-08-18T17-17-56-project.yaml"
    # config_path = "logs/2022-08-02T07-32-14_usc_512_transformer/configs/2022-08-02T07-32-14-project.yaml"
    config = OmegaConf.load(config_path)
    config ['data']['params']['batch_size'] = 1
    return config

def load_model(config):
    from taming.models.cond_transformer import Net2NetTransformer
    # instantiate model
    model = Net2NetTransformer(**config.model.params).to(device)
    # loading checkpoint
    ckpt_path = "logs/2022-08-18T17-17-56_usc_512_pf_transformer/checkpoints/last.ckpt"
    # ckpt_path = "logs/2022-08-02T07-32-14_usc_512_transformer/checkpoints/last.ckpt"
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model

def initialize():
    config = load_config() 
    model = None if DISABLE_MODEL_LOADING else load_model(config) 
    return model, config

def run(model, config, data, window=None, expansion=False):
    config = load_config() if config is None else config
    
    if not DISABLE_MODEL_LOADING:
        model = load_model(config) if model is None else model
    
    if expansion:
        processed_data = prepare_input_expansion(*data)
        new_loc = None
        processed_data = [processed_data[0],processed_data[1],new_loc]
    else:
        processed_data, new_loc = prepare_input(*data)

    if not DISABLE_MODEL_LOADING:
        target_image = run_network(model, config, *processed_data, window=window)
    else:
        target_image = processed_data[0]

    # renormalize output image
    target_image = np.uint8(127.5 * (target_image + 1.0))
    return model, config, target_image, new_loc

def run_network(model, config, image, segmentation, mask, window=None):
    nb = 1 
    step_size = 8
    c_code_res = 16
    temperature = 1.0
    top_k = 100
    det = False
    multiplier = 2 # 1 if mask is not None else 2
    res = 512 #256 if mask is not None else 512

    codebook_size = config.model.params.first_stage_config.params.embed_dim
    z_code_shape = [nb, codebook_size,c_code_res*multiplier,c_code_res*multiplier]
    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"

    # get codes from model
    x = tensify(image) 
    c = x if unconditional else tensify(segmentation)
    quant_c, c_indices = model.encode_to_c(c)
    quant_z, z_indices = model.encode_to_z(x)

    z_indices = z_indices.reshape(quant_z.shape[0],quant_z.shape[2],quant_z.shape[3])

    if not unconditional:
        cidx = c_indices
        cidx = cidx.reshape(quant_c.shape[0],quant_c.shape[2],quant_c.shape[3])

    nnx = c_code_res * multiplier 
    target_image = np.zeros([res, res, 3])

    # TODO
    # z_indices = torch.randint(codebook_size, [nb,rreconstructed_imagees], device=model.device)
    # z_indices = z_indices.reshape(nb, nnx, nnx)
    # input_z_indices = z_indices

    # Fill occupancy based on editted contents
    ################################################
    if mask is not None:
        occupancy = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        occupancy = F.interpolate(occupancy, scale_factor=1/16)
        occupancy = occupancy[0].numpy().astype(bool)
        occupancy = occupancy
    else:
        # the first quarter of the image is set
        occupancy = np.zeros(z_indices.shape).astype(bool)
        occupancy[:,:16,:16] = True
    ################################################

    # inferring codes
    for i in range(0, nnx - step_size, step_size):
        for j in range(0, nnx - step_size, step_size):
            idx = z_indices[:, i:i+c_code_res, j:j+c_code_res].reshape(nb, -1)
            occ = occupancy[:, i:i+c_code_res, j:j+c_code_res].reshape(nb, -1)       
            # only update a block in idx if it is not occupied
            for ii in range(idx.shape[1]):
                if not occ[0,ii]:
                    if unconditional:
                        patch = torch.cat((c_indices, idx[:,:ii]), dim=1)
                        logits,_ = model.transformer(patch)
                    else:
                        cpatch = cidx[:, i:i+c_code_res, j:j+c_code_res].reshape(nb, -1)
                        logits,_ = model.transformer(idx[:,:ii], cpatch)
                    logits = logits[:, -1, :]
                    logits = logits/temperature
                    if top_k is not None:
                      logits = model.top_k_logits(logits, top_k)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    if det:
                        _, idx[:,ii] = torch.topk(probs, k=1, dim=-1)
                    else:
                        idx[:,ii] = torch.multinomial(probs, num_samples=1)
                    occ[:,ii] = True
            idx = idx.reshape(nb, c_code_res, c_code_res)
            occ = occ.reshape(nb, c_code_res, c_code_res)
            z_indices[:, i:i+c_code_res, j:j+c_code_res] = idx
            occupancy[:, i:i+c_code_res, j:j+c_code_res] = occ


            # reconstructed image will be of size 256x256x3
            if mask is None:
                # new_z_code_shape = [nb, codebook_size, multiplier*16, multiplier*16]
                x_sample = model.decode_to_img(z_indices, z_code_shape)
                reconstructed_image = tensor_to_numpy(x_sample)
                print(f"Drawing at step {i}_{j}")
                draw_at(window, reconstructed_image, idx=f"{i}_{j}")

    x_sample = model.decode_to_img(z_indices, z_code_shape)
    reconstructed_image = tensor_to_numpy(x_sample)

    return reconstructed_image