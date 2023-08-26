# eval testing

# load config for transformer
import warnings
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
from eval_util import *

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
        "--mode",
        type=str,
        default="transformer",
        help="vqgan | attgan | transformer | transformer_half ",
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
    # parser.add_argument(
    #     "-d",
    #     "--datadir",
    #     type=str,
    #     default="",
    #     help="path to datasets",
    # )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--split",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
    )
    parser.add_argument(
        "--inpainting",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
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
    warnings.filterwarnings("ignore")
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

    # key configuration: whether the model is conditional or not
    ###################################################
    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"
    if unconditional:
        print("Using an unconditional model!")
    ###################################################
    
    # instantiate model
    print("Instantiating model...")
    model = instantiate_from_config(config.model).to(device).eval()
    # model = Net2NetTransformer(**config.model.params).to(device).eval()
    print("Done!")
   
    # loading checkpoint
    if len(ckpt_path) > 0:
        sd = torch.load(ckpt_path, map_location=device)["state_dict"]
        print("Loading checkpoint from %s..."%ckpt_path)
        model.load_state_dict(sd, strict=False)
        print("Done!")
        
    # loading dataset
    print("instantiating Dataset...")
    # config.data.params.train.params.split = "validation"
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("Done!")

    if 'test' in data.datasets.keys():
        dataset = data.datasets['test']
    else:
        dataset = data.datasets['train']

    dataset_iter = iter(dataset)
    length = min(200, len(dataset))
    data_select = range(length) # range(len(dataset))

    if opt.mode == 'transformer':
        eval_method = eval_transformer_log
    elif opt.mode == "transformer_half":
        eval_method = eval_half      
    elif opt.mode == 'vqgan':
        eval_method = eval_vqgan
    elif opt.mode == 'attgan':
        eval_method = eval_attgan
    else:
        print(f"No mode found for {opt.mode}!")
        exit()

    import time

    eval_method = functools.partial(eval_method, model=model, opt=opt, config=config, save_path=save_path)
    with torch.no_grad():
        for i in data_select:
            batch_data = dataset[i]
            start = time.time()
            eval_method(data=batch_data, idx=i)
            print("Time spent on inference is %.2f seconds"%(time.time() - start))
            # eval_mult(batch_data, i, model, None, config, None, multiplier=scale)

