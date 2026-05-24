from os import path as osp

import decord
import pandas as pd
import torch
import torchvision.transforms as tf
from PIL import Image

from pyiqa.data.transforms import PairedToTensor, transform_mapping
from pyiqa.utils.registry import DATASET_REGISTRY

from .base_iqa_dataset import BaseIQADataset


@DATASET_REGISTRY.register()
class T2VQADataset(BaseIQADataset):
    def init_path_mos(self, opt):
        # Try to read as CSV first (pyiqa prepared format with header)
        # Fall back to pipe-delimited format (original T2VQA format)
        try:
            # Check if file has header by reading first line
            with open(opt['meta_info_file'], 'r') as f:
                first_line = f.readline().strip()
            
            # Check if first line looks like a header or data
            if 'video_path' in first_line or first_line.startswith('video_path'):
                # Has header, use pandas to read
                self.meta_info = pd.read_csv(opt['meta_info_file'])
            elif '|' in first_line:
                # Original pipe-delimited format without header
                self.meta_info = pd.read_csv(opt['meta_info_file'], sep='|', header=None,
                                             names=['video_path', 'text', 'mos'])
            else:
                # Comma-delimited without header
                self.meta_info = pd.read_csv(opt['meta_info_file'], header=None,
                                             names=['video_path', 'text', 'mos'])
        except:
            # Default to CSV format
            self.meta_info = pd.read_csv(opt['meta_info_file'])

        dataroot = opt['dataroot']
        video_folder = opt.get('video_folder')
        # Handle video_folder: None, empty string, or '~' means videos are directly in dataroot
        if video_folder and video_folder != '~':
            video_base_path = osp.join(dataroot, video_folder)
        else:
            video_base_path = dataroot

        self.paths_mos = []
        for _, row in self.meta_info.iterrows():
            # Support both 'video_path' and 'video_name' column names
            video_name = row.get('video_path', row.get('video_name', ''))
            video_path = osp.join(video_base_path, str(video_name))
            self.paths_mos.append({
                'video_path': video_path,
                'text': str(row.get('text', '')),
                'mos': float(row.get('mos', row.get('score', 0)))
            })

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

    def get_split(self, opt):
        all_split_lists = [x for x in self.meta_info.columns.tolist() if 'split' in x]

        split_index = opt.get('split_index', None)

        if split_index is not None and len(all_split_lists) > 0:
            if isinstance(split_index, str):
                split_name = split_index
            elif isinstance(split_index, int):
                split_ratio = opt.get('split_ratio', '802')
                split_name = f'ratio{split_ratio}_seed123_split_{split_index:02d}'

            if split_name in all_split_lists:
                split_paths_mos = []
                for i in range(len(self.paths_mos)):
                    if self.meta_info[split_name][i] == self.phase:
                        split_paths_mos.append(self.paths_mos[i])
                self.paths_mos = split_paths_mos
                self.logger.info(f'Using split: {split_name}, phase: {self.phase}, samples: {len(self.paths_mos)}')
            else:
                self.logger.info(f'Split {split_name} not found in {all_split_lists}, using all data.')
        elif split_index is None:
            self.logger.info(f'No split_index specified, using all data for phase: {self.phase}')

    def mos_normalize(self, opt):
        mos_range = opt.get('mos_range', None)
        mos_lower_better = opt.get('lower_better', None)
        mos_normalize = opt.get('mos_normalize', False)

        if mos_normalize:
            assert mos_range is not None and mos_lower_better is not None, (
                'mos_range and mos_lower_better should be provided when mos_normalize is True'
            )

            def normalize(mos_label):
                mos_label = (mos_label - mos_range[0]) / (mos_range[1] - mos_range[0])
                if mos_lower_better:
                    mos_label = 1 - mos_label
                return mos_label

            for item in self.paths_mos:
                item['mos'] = normalize(item['mos'])
            self.logger.info(
                f'mos_label is normalized from {mos_range}, lower_better[{mos_lower_better}] to [0, 1], lower_better[False(higher better)].'
            )

    def __getitem__(self, index):
        item = self.paths_mos[index]
        video_path = item['video_path']
        text_description = item['text']
        mos_label = float(item['mos'])

        num_frames = self.opt.get('num_frames', 8)

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
        frames = vr.get_batch(frame_indices).asnumpy()

        frame_pils = [Image.fromarray(frame).convert('RGB') for frame in frames]

        transformed_frames = self.trans(frame_pils)
        for i in range(len(transformed_frames)):
            transformed_frames[i] = transformed_frames[i] * self.img_range
        video_tensor = torch.stack(transformed_frames, dim=0)
        mos_label_tensor = torch.Tensor([mos_label])

        return {
            'video': video_tensor,
            'mos_label': mos_label_tensor,
            'text': text_description,
            'video_path': video_path,
        }

    def __len__(self):
        return len(self.paths_mos)
