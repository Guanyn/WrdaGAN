# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com


from src.utils import *

import os

import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset


def prepare_datasets(configs, transform=None):
    imsize = configs['imsize']
    if transform is not None:
        image_transform = transform
    elif configs['config_name'].find('CelebA') != -1:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    # train dataset
    train_dataset = TextImgDataset(split='train', transform=image_transform, configs=configs)
    val_dataset = TextImgDataset(split='val', transform=image_transform, configs=configs)
    return train_dataset, val_dataset


class TextImgDataset(Dataset):
    def __init__(self, split='train', transform=None, configs=None, tokenizer=None, norm=None):
        if split == 'test':
            split = 'val'
        coco = False
        if configs['dataset_name'] == 'coco':
            split += '2014'
            coco = True
        self.transform = transform
        self.split = split
        self.configs = configs
        self.tokenizer = tokenizer
        self.embeddings_num = configs['text']['captions_per_image']
        self.data_dir = configs['data_dir']
        if norm:
            self.norm = norm
        else:
            self.norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        self.bbox = None
        if configs['dataset_name'] == 'birds':
            self.bbox = self.load_bbox()
        self.data_dir = self.configs['data_dir']

        self.image_dir = os.path.join(self.data_dir, 'images', self.split)

        self.captions = {}
        caption_path = os.path.join(self.data_dir, 'text')
        for root, dirs, files in os.walk(caption_path):
            for file in files:
                if file.endswith(".txt"):
                    captions = []
                    with open(os.path.join(root, file), 'r', encoding='utf8') as f:
                        for line in f.readlines():
                            captions.append(line.strip())
                    if not coco:
                        self.captions[os.path.basename(root) + '/' + file.replace('.txt', '')] = captions
                    else:
                        self.captions[file.replace('.txt', '')] = captions

        self.images = {}
        self.path = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    if not coco:
                        self.images[os.path.basename(root) + '/' + file.replace('.jpg', '').replace('.png', '')] = os.path.join(root, file)
                        self.path.append(os.path.basename(root) + '/' + file.replace('.jpg', '').replace('.png', ''))
                    else:
                        self.images[file.replace('.jpg', '').replace('.png', '')] = os.path.join(root, file)
                        self.path.append(file.replace('.jpg', '').replace('.png', ''))
                    # if len(self.path) >= 1000:
                    #     break

    def load_bbox(self):
        bbox_path = os.path.join(self.data_dir, "CUB_200_2011/bounding_boxes.txt")

        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)

        img_file = os.path.join(self.data_dir, 'CUB_200_2011/images.txt')

        df_img = pd.read_csv(img_file, delim_whitespace=True, header=None)

        img_names = df_img[1].tolist()

        bbox_dict = {img_file: [] for img_file in img_names}

        numImgs = len(img_names)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = img_names[i][:-4]
            bbox_dict[key] = bbox
        #
        return bbox_dict

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        name = self.path[item]

        img_path = self.images[name]

        caption = self.captions[name]

        if self.bbox is not None:
            bbox = self.bbox[name]
        else:
            bbox = None

        sent_ix = random.randint(0, self.embeddings_num)

        img = get_imgs(img_path, bbox, self.transform, normalize=self.norm)

        return img, caption[sent_ix], name

    def get_name(self, name):
        img_path = self.images[name]
        caption = self.captions[name]

        if self.bbox is not None:
            bbox = self.bbox[name]
        else:
            bbox = None

        sent_ix = random.randint(0, self.embeddings_num)

        img = get_imgs(img_path, bbox, self.transform, normalize=self.norm)

        return img, caption[sent_ix], name, bbox

