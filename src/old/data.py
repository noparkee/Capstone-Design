from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

import pandas as pd
import numpy as np
import os
from PIL import Image


class ourDataset(Dataset):
    def __init__(self):
        
        description = pd.read_pickle('../data/description.pkl')

        self.path = description['path']
        self.name = description['name']
        self.label = description['label_num']

        self.train_flag = False
        self.transform_train, self.transform_eval = get_transforms()


    def __len__(self):

        return len(self.label)


    def __getitem__(self, index):
        image = Image.open(self.path[index]).convert('RGB')
        image = self.transform_train(image) if self.train_flag else self.transform_eval(image)

        name = self.name[index]
        label = self.label[index]

        return image, name, label


def collate_fn(batch):
    images, names, labels = zip(*batch)

    #images = torch.stack(images, 0)
    labels = torch.stack(images, 0)

    return images, names, labels       # 이름이 필요하면 그 때 사용하자
    #return images, labels


def get_data_iterators():
    BATCH_SIZE = 32
    NUM_WORKERS = 1
    DROP_LAST = True
    SHUFFLE = False

    TOTAL = 2245
    NUM_TRAIN = int(TOTAL * 0.7)
    NUM_TEST = TOTAL - NUM_TRAIN

    dataset = ourDataset()
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [NUM_TRAIN, NUM_TEST])

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, drop_last=DROP_LAST, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, drop_last=DROP_LAST, collate_fn=collate_fn)

    return train_loader, test_loader


def get_transforms():
    """ get transforms for CUB datasets """
    resize, cropsize = 512, 448

    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_eval


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch