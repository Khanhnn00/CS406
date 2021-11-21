import os
import os.path as osp
import glob
import imageio
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((256, 256)),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

class ImageDataset(data.Dataset):
    def __init__(self, images, labels=None, transforms=transform):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data