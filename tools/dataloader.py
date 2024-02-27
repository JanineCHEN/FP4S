import os
import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

class SribbleObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, mask_root, gray_root, edge_root,
                 trainsize, ifNorm):
        self.trainsize = trainsize
        self.ifNorm = ifNorm
        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root) if f.endswith('.npz')
        ]
        self.masks = [
            mask_root + f for f in os.listdir(mask_root) if f.endswith('.png')
        ]
        self.grays = [
            gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')
        ]
        self.edges = [
            edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.edges = sorted(self.edges)
        self.size = len(self.images)
        if self.ifNorm:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.Normalize(
                    [0.9488, 0.9492, 0.9470],
                    [0.1669, 0.1663, 0.1698]),  # value calculated based on all FP images in the dataset
            ])
            print("Images are normalized!")
        if not self.ifNorm:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.trainsize, self.trainsize)),
            ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.trainsize, self.trainsize),
                interpolation=transforms.InterpolationMode.NEAREST),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.trainsize, self.trainsize),
                interpolation=transforms.InterpolationMode.NEAREST),
        ])
        self.gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.trainsize, self.trainsize)),
        ])
        self.edge_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.trainsize, self.trainsize)),
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.npload(self.gts[index])
        gt = gt * 255
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])
        edge = self.binary_loader(self.edges[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        mask = self.mask_transform(mask)
        gray = self.gray_transform(gray)
        edge = self.edge_transform(edge)
        return image, gt, mask, gray, 1 - edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def npload(self, path):
        a = np.load(path)
        return a['a']

    def __len__(self):
        return self.size

def get_loader(image_root,
               gt_root,
               mask_root,
               gray_root,
               edge_root,
               batchsize,
               trainsize,
               
               ifNorm=True,
               shuffle=True,
               num_workers=1,
               pin_memory=True):
    
    dataset = SribbleObjDataset(image_root,
                            gt_root,
                            mask_root,
                            gray_root,
                            edge_root,
                            trainsize,
                            ifNorm=ifNorm)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader


class UnsupervisedDataset(data.Dataset):
    def __init__(self, image_root, trainsize, ifNorm, image_list_path=None):
        self.trainsize=trainsize
        self.ifNorm=ifNorm
        if image_list_path==None:
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        if not image_list_path==None:
            with open(image_list_path, 'r') as file:
                image_list = file.readlines()
            image_list = [line.strip() for line in image_list if line.strip()]
            self.images = [image_root + f for f in image_list]
            print(f"{image_list_path} is loaded!")
        self.images = sorted(self.images)
        self.size = len(self.images)
        if self.ifNorm: 
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.Normalize([0.9488, 0.9492, 0.9470], [0.1669, 0.1663, 0.1698]), 
                ])
            print("Images are normalized!")
        if not self.ifNorm:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.trainsize, self.trainsize)),
                ])

    def __getitem__(self, index):
        path = self.images[index]
        name = path.split("/")[-1].split(".")[0]
        image = self.rgb_loader(path)
        image = self.img_transform(image) 
        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size

def get_unsupervised_loader(image_root, batchsize, trainsize, image_list_path=None, ifNorm = True, shuffle=False, num_workers=1, pin_memory=True):
    dataset = UnsupervisedDataset(image_root, trainsize, ifNorm = ifNorm, image_list_path=image_list_path)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader


class ValDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, ifNorm):
        self.trainsize=trainsize
        self.ifNorm=ifNorm
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.npz')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        if self.ifNorm: 
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.Normalize([0.9488, 0.9492, 0.9470], [0.1669, 0.1663, 0.1698]), 
                ])
            print("Images are normalized!")
        if not self.ifNorm:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.trainsize, self.trainsize)),
                ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            ])

    def __getitem__(self, index):
        path = self.images[index]
        name = path.split("/")[-1].split(".")[0]
        image = self.rgb_loader(path)
        gt = self.npload(self.gts[index])
        gt = gt*255
        image = self.img_transform(image) 
        gt = self.gt_transform(gt) 
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def npload(self, path):
        a = np.load(path)
        return a['a']

    def __len__(self):
        return self.size

def get_val_loader(image_root, gt_root, batchsize, trainsize, ifNorm = True, shuffle=False, num_workers=1, pin_memory=True):

    dataset = ValDataset(image_root, gt_root, trainsize, ifNorm = ifNorm)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader
