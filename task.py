import os
import shutil
import random
import math
from itertools import permutations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as F
from torchvision import datasets, transforms, models

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', '-r', type=str, required=False)
## parser.add_argument("--seq_len", '-sl', type = int, required = False, default = 6,)
args = parser.parse_args()

preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class WM_task(Dataset):
    """Delayed Recall task: with/without temporal order."""

    def __init__(self, image_path, seq_len = 2, delay_period = 1, grid_size = 2, mode = "train", order_type = 1):
        """
        Info: 
            Delay Recall task: the subject is required to report the spatial location of the sequence 
            of objects presented during sample phase, in the given order or random order.
        Args:
            image_path: path to the image folder
            seq_len: lenght of the sample sequence
            delay_period: duration of the delay frames, if 0, then random delay [need to set an upper boudn]
            grid_size: discretize the spatial locations
            mode: if "train", draw images from train dataset; otherwiser from validation dataset
            order_type: if report the spatial locations in random or given order; if 0, random order; if 1, followe the temporal order
        """
        self.seq_len = seq_len
        self.delay_period = delay_period
        self.grid_size = grid_size
        self.order_type = order_type
        self.mode = mode
        self.img_dir = os.path.join(image_path, "%s"%self.mode) # mode: train, val, test
        self.target_size = (224 // self.grid_size, 224 // self.grid_size)

        if self.order_type == 0:
            self.n_permutations = math.factorial(self.seq_len)

    def __len__(self):
        return 1280 # just some random number since we are kind of training in a continual learning way

    def image_read(self):
        "randomly read one image from the image directory and downsample it to 224/grid_size and return the pixel array of the image"
        if self.mode == "train":
            ctg_files = os.listdir(self.img_dir)
            # check if they are all directories
            ctg_files = [f for f in ctg_files if os.path.isdir(os.path.join(self.img_dir, f))]
            random_ctg_file = os.path.join(self.img_dir, random.sample(ctg_files, 1)[0], "images")
        
        else:
            random_ctg_file = os.path.join(self.img_dir, "images")

        img_files = os.listdir(random_ctg_file)
        random_img = os.path.join(random_ctg_file, random.sample(img_files, 1)[0])
        # print("path to the sampled image file:", random_img)

        
        image=Image.open(random_img)
        test_image = np.array(image)
        while len(test_image.shape)==2: 
            random_img = os.path.join(random_ctg_file, random.sample(img_files, 1)[0])
            image=Image.open(random_img)
            test_image = np.array(image)
        
        image = preprocess(image)
                
        # Resize the image using bilinear interpolation
        downsampled_image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)
        
        downsampled_image = np.array(downsampled_image)
        # print("shape of the downsampeld image:", downsampled_image.shape)
        
        # plt.figure()
        # plt.imshow(torch.tensor(downsampled_image).permute(1,2,0))
        # plt.savefig("/home/xuan/projects/def-bashivan/xuan/AC2023/figures/downsampled_image.png")

        # plt.figure()
        # plt.imshow(torch.tensor(image).permute(1,2,0))
        # plt.savefig("/home/xuan/projects/def-bashivan/xuan/AC2023/figures/original_image.png")
        # print(downsampled_image.shape)
        return downsampled_image
    
    def rendering(self, image, spatial_loc):
        "put the stimuli to the given spatial location and return the rendered the pixel value of the frame"
        template = np.zeros((3, 224,224))
        # print("check point 1")
        template[:, self.target_size[0]*spatial_loc[0]:self.target_size[0]*(spatial_loc[0]+1),self.target_size[1]*spatial_loc[1]:self.target_size[1]*(spatial_loc[1]+1)] = image
        # print("check point 2")
        return template

    def sample_trial(self):
        actions = np.zeros((self.seq_len, self.seq_len + self.delay_period + 1)) # if no action, report 0; otherwise, report the spatial locations
        frames = []
        for i in range(self.seq_len):
            space_x = random.randint(0,self.grid_size-1)
            space_y = random.randint(0,self.grid_size-1)
            image = self.image_read()
            frame = self.rendering(image, (space_x, space_y))
            
            frames.append(frame)
            actions[i, -1] = space_x * self.grid_size  + space_y + 1

        if self.delay_period != 0: # todo: if random delay period
            for i in range(self.delay_period + 1 ): # add empty response frame as well
                # add empty frames todo: do  we need fixations for delay frames?
                frame = np.zeros((3, 224,224))
                frames.append(frame)

        
        frames = np.stack(frames)
        

        return frames, actions


    def __getitem__(self, idx,):
        frames, actions = self.sample_trial() 
        if self.order_type == 1:
            return frames, actions
        elif self.order_type == 0: # random order
            all_actions = np.zeros((self.n_permutations, self.seq_len, self.seq_len + self.delay_period + 1))
            response_actions = actions[:, -1]
            permuted_arrays = list(permutations(response_actions))

            # Convert the permutations to NumPy arrays if needed
            permuted_arrays = [np.array(perm) for perm in permuted_arrays]

            # Print the list of all permutations
            for i, perm in enumerate(permuted_arrays):
                all_actions[i, :, -1] = perm
            
            return frames, all_actions

             
# sanity check     
# task = WM_task(image_path=args.root,seq_len = 2, delay_period = 1, grid_size = 2, mode = "train", order_type = 0)

# for i in range(len(task)):
#     print(i)
#     frames, actions = task[i]
# for i, frame in enumerate(frames):
#     plt.figure()
#     plt.imshow(torch.tensor(frame).permute(1,2,0))
#     plt.savefig("/home/xuan/projects/def-bashivan/xuan/AC2023/figures/frame_%d.png" % i)
# # print(actions[:,-1]) 
# for action in actions:
#     print(action[:, -1])



