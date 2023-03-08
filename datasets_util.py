import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

class CustomImageDataset(Dataset):
    """ Dataset wrapper class for the GTSRB dataset
        - preprocess each image
        - output data in format Faster R-CNN expects:
          - the image data as a tensor of shape C * H * W
          - the ground-truth bounding box of shape 1 * 4
          - the class label of shape 1 * 1
    """
    def __init__(self, num_classes, data_csv, imgs_path="./gtsrb"):
        self.width = 50
        self.height = 50
        # resize and normalise the image
        self.transform = transforms.Compose([
            transforms.Resize((self.width, self.height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])
        self.imgs_path = imgs_path
        self.num_classes = num_classes
        self.data_csv = data_csv
      
    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
      data = self.data_csv.iloc[idx]
      # resize the bounding box
      width = data['Width']
      height = data['Height']
      scale_x = self.width / width
      scale_y = self.height / height
      ul_x = data['Roi.X1'] * scale_x
      ul_y = data['Roi.Y1'] * scale_y
      lr_x = data['Roi.X2'] * scale_x
      lr_y = data['Roi.Y2'] * scale_y
      # prepare data in correct format
      i = data['ClassId']
      path = os.path.join(self.imgs_path, data['Path'])
      im = Image.open(path)
      im = self.transform(im)
      targets = {"boxes": torch.tensor([[ul_x, ul_y, lr_x, lr_y]], dtype=torch.float), "labels": torch.tensor([i], dtype=torch.int64)}
      return im, targets

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
    
    def setup(self, stage=None):
      pass

    def collate_fn(self, data):
        num_entry = len(data)
        input = []
        target = []
        for idx in range(num_entry):
            row = data[idx]
            input.append(row[0])
            target.append(row[1])
        return input, target

    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
      return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
      return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn
        )
    

def get_datasets():
    path = os.path.expanduser('~') + "/fasterrcnn/gtsrb/"
    meta_csv = pd.read_csv(path + "Meta.csv")
    train_csv = pd.read_csv(path + "Train.csv")
    test_csv = pd.read_csv(path + "Test.csv")
    num_classes = len(np.unique(meta_csv['ClassId']))
    return meta_csv, train_csv, test_csv, num_classes

def get_dataloaders(num_classes, train_csv, test_csv):
    path = os.path.expanduser('~') + "/fasterrcnn/gtsrb"
    dataset = CustomImageDataset(num_classes, train_csv, imgs_path=path)
    test_data = CustomImageDataset(num_classes, test_csv)
    train_sample_num = int(len(dataset) * 0.8)
    train_data, val_data = random_split(dataset, [train_sample_num,len(dataset) - train_sample_num], generator=torch.Generator().manual_seed(42))
    datamodule = CustomDataModule(train_data, val_data, test_data, 32)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    return train_dataloader, val_dataloader, test_dataloader