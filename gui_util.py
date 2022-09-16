import io
import os
import numpy as np
import albumentations
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

tensify = lambda x: torch.from_numpy(x[None]).to(device).permute(0,3,1,2).contiguous().float()
tensor_to_numpy = lambda x:x.detach().cpu().numpy()[0].transpose(1,2,0)

def convert_to_onehot(segmentation, n_labels=3):
    flatseg = np.ravel(segmentation)
    onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
    onehot[np.arange(flatseg.size), flatseg] = True
    onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
    return onehot.astype(np.float32)

def process_image(coords, image, seg, filter_label, replace_label):
    if image is None or seg is None:
        return
    image = np.array(image)
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
    masked_image = image * mask[...,None]
    return masked_image, new_seg, mask

def prepare_input(coords, image, seg, mask):
    import ipdb; ipdb.set_trace()
    image = np.array(image).astype(np.uint8)
    seg = seg.astype(np.uint8)
    w, h = image.shape[:2]
    rescaler = 512/max(w, h)
    sy, sx, ey, ex = coords
    sx_i = int((w - max(sx,ex) * w) * rescaler)
    ex_i = int((w - min(sx,ex) * w) * rescaler)
    sy_i = int(min(sy,ey) * h * rescaler)
    ey_i = int(max(sy,ey) * h * rescaler)
 
    preprocessor = albumentations.Compose(
        [albumentations.SmallestMaxSize(max_size=512),
         albumentations.Crop(sx_i, sy_i, sx_i + 256, sy_i + 256)
        ],
        additional_targets={'segmentation':'image',
                            'mask':'mask'})

    processed = preprocessor(image=image, segmentation=seg, mask=mask)
    image, segmentation = processed["image"], processed["segmentation"]
    image = (image / 127.5 - 1.0).astype(np.float32)
    segmentation = convert_to_onehot(segmentation)
    return [image, segmentation, processed['mask']]

def load_config():
    config_path = "logs/2022-08-02T07-32-14_usc_512_transformer/configs/2022-08-02T07-32-14-project.yaml"
    config = OmegaConf.load(config_path)
    config ['data']['params']['batch_size'] = 1
    return config

def load_model(config):
    from taming.models.cond_transformer import Net2NetTransformer
    device = torch.device('cuda:0')
    # instantiate model
    model = Net2NetTransformer(**config.model.params).to(device)
    # loading checkpoint
    ckpt_path = "logs/2022-08-02T07-32-14_usc_512_transformer/checkpoints/last.ckpt"
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model, device


def run(model, config, data):
    config = load_config() if config is None else config
    model = load_model(config) if model is None else model
    processed_data = prepare_input(*data)
    target_image = run_network(model, config, *processed_data)
    return model, config, target_image

def run_network(model, config, image, segmentation, mask):
    nb = 1 
    res = 256
    step_size = 8
    c_code_res = 16
    codebook_size = config.model.params.first_stage_config.params.embed_dim
    z_indices_shape = [nb,res]
    z_code_shape = [nb, codebook_size,c_code_res,c_code_res]
    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"
    temperature = 1.0
    top_k = 100
    det = False

    # get codes from model
    x = tensify(image) 

    if unconditional:
        c = x
        quant_c, c_indices = model.encode_to_c(c)
    else:
        c = tensify(segmentation)
        quant_c, c_indices = model.encode_to_c(seg_tensor)

    quant_z, z_indices = model.encode_to_z(x)
    gt_z_indices = z_indices

    idx = z_indices
    idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])
    if not unconditional:
        cidx = c_indices
        cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

    nnx = c_code_res # * multiplier 
    target_image = np.zeros([256, 256, 3])

    # TODO
    # z_indices = torch.randint(codebook_size, [nb,res], device=model.device)
    # z_indices = z_indices.reshape(nb, nnx, nnx)
    # input_z_indices = gt_z_indices

    z_indices = gt_z_indices.reshape(nb, c_code_res, c_code_res)
    occupancy = np.zeros(z_indices.shape).astype(bool)


    # TODO: Fill occupancy based on editted contents
    ################################################

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

    # new_z_code_shape = [nb, codebook_size, multiplier*16, multiplier*16]
    x_sample = model.decode_to_img(z_indices, z_code_shape)

    # reconstructed image will be of size 256x256x3
    reconstructed_image = tensor_to_numpy(x_sample)

    return reconstructed_image