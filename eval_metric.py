# eval testing

# load config for transformer
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import functools
import yaml
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import time
import cv2
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from taming.models.cond_transformer import Net2NetTransformer
from taming.metric import metric_main
from taming.metric import metric_utils
from eval_util import *

# script for evaluating inpainting performance

# fid2993_full, fid36k5_full, ids_places, kid50k_full, pr50k3_full, ppl2_wend, is50k
metrics = ['fid50k_full']

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-m",
        "--metrics",
        nargs='+',
        type=str,
        default="fid50k_full",
        help="Choose among [fid2993_full, fid36k5_full, ids_places, kid50k_full, pr50k3_full, ppl2_wend, is50k]",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="",
        help="path to configuration",
    )
    parser.add_argument(
        "-r",
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoints",
    )
    parser.add_argument(
        "--sample-freq",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="eval",
        help="optional name for the save folder",
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    device = torch.device('cuda')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # key configuration: config and ckpt path
    config_path = opt.base
    ckpt_path = opt.ckpt
    expname = opt.name + '_' + os.path.basename(config_path).split('.yaml')[0]

    # loading config
    save_path = os.path.join("logs/eval", f"{expname}_{now}")
    os.makedirs(save_path, exist_ok=True)
    config = load_config(config_path)

    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"
    if unconditional:
        print("Using an unconditional model!")
    
    # instantiate model
    print("Instantiating model...")
    model = instantiate_from_config(config.model).to(device).eval()
    print("Done!")
   
    # loading checkpoint
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    print("Loading checkpoint from %s..."%ckpt_path)
    model.load_state_dict(sd, strict=False)
    print("Done!")
    
    # loading dataset
    print("instantiating Dataset...")
    config.data.params.train.params.split = "validation"
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("Done!")

    # if 'test' in data.datasets.keys():
    #     dataset = data.datasets['test']
    # else:
    #     dataset = data.datasets['train']

    dataset = data.datasets['train']

    # callback function for the generative model
    def generate_results(G, batch):
        return G(batch)[0]

    kwargs = dict(G=model, dataset=dataset, G_callback=generate_results, device=device, num_gpus=opt.num_gpus)

    for metric in metrics:
        results = metric_main.calc_metric(metric, **kwargs)
        metric_main.report_metric(results, run_dir=save_path)

