from os import path as osp

import pandas as pd
import torch
import torchvision.transforms as tf
from PIL import Image

from pyiqa.data.transforms import PairedToTensor, transform_mapping
from pyiqa.utils.registry import DATASET_REGISTRY

from .base_iqa_dataset import BaseIQADataset


@DATASET_REGISTRY.register()
class LIVEChallengeDataset(BaseIQADataset):
    """LIVE Challenge Dataset for no-reference image quality assessment.
    
    CSV format: File, MOS, Scene
    """

    def init_path_mos(self, opt):
        """Initialize paths and MOS labels from CSV file.
        
        CSV format: File, MOS, Scene
        """
        self.meta_info = pd.read_csv(opt['meta_info_file'])
        
        dataroot = opt['dataroot']

        self.paths_mos = []
        for _, row in self.meta_info.iterrows():
            img_path = osp.join(dataroot, row['File'])
            mos_label = float(row['MOS'])
            self.paths_mos.append([img_path, mos_label])

    def get_transforms(self, opt):
        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        self.img_range = opt.get('img_range', 1.0)
        transform_list += [
            PairedToTensor(),
        ]
        self.trans = tf.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}

    def __len__(self):
        return len(self.paths_mos)


@DATASET_REGISTRY.register()
class SPAQDataset(BaseIQADataset):
    """SPAQ Dataset for no-reference image quality assessment.
    
    CSV format: name, mos
    """

    def init_path_mos(self, opt):
        """Initialize paths and MOS labels from CSV file.
        
        CSV format: name, mos
        """
        self.meta_info = pd.read_csv(opt['meta_info_file'])
        
        dataroot = opt['dataroot']

        self.paths_mos = []
        for _, row in self.meta_info.iterrows():
            img_path = osp.join(dataroot, row['name'])
            mos_label = float(row['mos'])
            self.paths_mos.append([img_path, mos_label])

    def get_transforms(self, opt):
        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        self.img_range = opt.get('img_range', 1.0)
        transform_list += [
            PairedToTensor(),
        ]
        self.trans = tf.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}

    def __len__(self):
        return len(self.paths_mos)


@DATASET_REGISTRY.register()
class PCLVQADataset(BaseIQADataset):
    """PCL-VQA Dataset for no-reference image quality assessment.
    
    CSV format: File, MOS, Scene
    """

    def init_path_mos(self, opt):
        """Initialize paths and MOS labels from CSV file.
        
        CSV format: File, MOS, Scene
        """
        self.meta_info = pd.read_csv(opt['meta_info_file'])
        
        dataroot = opt['dataroot']

        self.paths_mos = []
        for _, row in self.meta_info.iterrows():
            img_path = osp.join(dataroot, row['File'])
            mos_label = float(row['MOS'])
            self.paths_mos.append([img_path, mos_label])

    def get_transforms(self, opt):
        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        self.img_range = opt.get('img_range', 1.0)
        transform_list += [
            PairedToTensor(),
        ]
        self.trans = tf.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}

    def __len__(self):
        return len(self.paths_mos)
