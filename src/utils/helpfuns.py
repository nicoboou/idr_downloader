import os
import sys
import cv2
import pdb
import json
import math
import h5py
import time
import torch
import pickle
import shutil
import random
import inspect
import warnings
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
from tabulate import tabulate
import matplotlib.pylab as plt
from collections import OrderedDict
from easydict import EasyDict as edict

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def check_dir(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
        
def dir_path(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return string
    else:
        raise NotAFileError(path)        
        
def get_parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))
    
def save_json(data, fname):
    fname = os.path.abspath(fname)
    if not fname.endswith('.json'):
        fname += '.json'
    with open(fname, 'w') as wfile:  
        json.dump(data, wfile)
        
def load_json(fname):
    fname = os.path.abspath(fname)
    with open(fname, "r") as rfile:
        data = json.load(rfile)
    return data

def save_pickle(data, fname):
    fname = os.path.abspath(fname)
    if not fname.endswith('.pickle'):
        fname += '.pickle'    
    with open(fname, 'wb') as wfile:
        pickle.dump(data, wfile, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(fname):
    fname = os.path.abspath(fname)
    with open(fname, 'rb') as rfile:
        data = pickle.load(rfile)
    return data      

def get_saved_model_path(checkpoint_name):
    path = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(path):
        raise IOError("Checkpoint path {} does not exist".format(path))
    else:
        return os.path.join(path, checkpoint_name) 
    
def load_params(args):
    if args.checkpoint:
        checkpoint_path = get_saved_model_path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        checkpoint['parameters']['training_params']['model_name'] = args.checkpoint
        return checkpoint['parameters']    
    elif args.params_path:
        return load_json(args.params_path)

    raise IOError("Please define the training paramenters")
        
def isnan(x):
    return x != x

def iszero(x):
    return x == 0    

def get_rand_image_from_dataloader(dataloader):
    batch_imgs, batch_targets = next(iter(dataloader['testloader']))
    rand_idx = np.random.randint(0, batch_imgs.shape[0])
    rand_img_tensor = batch_imgs[rand_idx]
    rand_target_tensor = batch_targets[rand_idx]
    rand_img = rand_img_tensor.permute(1, 2, 0).numpy()
    normalized_img = (rand_img - np.min(rand_img)) / (np.max(rand_img) - np.min(rand_img))
    normalized_img_float = normalized_img.astype(np.float32)
    rand_target = rand_target_tensor.item()
    return batch_imgs, batch_targets, rand_img_tensor, normalized_img_float, rand_target

def overlap_imgs(
        img: np.ndarray,
        mask: np.ndarray,
        use_rgb: bool = False,
        colormap: int = cv2.COLORMAP_JET,
        image_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Description:
        -----------
            Overlap a mask on top of an image.

        Arguments:
        -----------
            img (np.ndarray): Original input image
            cam (np.ndarray): Mask
            use_rgb (bool): Whether to use RGB or BGR
            image_weight (float): Weight of the image

        Returns:
        -----------
            np.ndarray: Original input mage with mask overlay.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

        resized_mask = cv2.resize(heatmap, (img.shape[0], img.shape[1]))#, interpolation=cv2.INTER_NEAREST)

        normalized_mask = resized_mask / np.max(resized_mask)

        if np.max(img) > 1:
            raise ValueError(
                "Bad input shape: {type(img)}; input image should be of type np.float32 in the range [0, 1]"
            )

        if image_weight < 0 or image_weight > 1:
            raise ValueError(
                f"image_weight should be in the range 0 <= x <= 1 but got {image_weight}"
            )

        overlay = (1 - image_weight) * normalized_mask + image_weight * img
        #overlay = overlay / np.max(overlay)

        return overlay
