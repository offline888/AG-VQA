import os
from os import path as osp

import pandas as pd
import torch
import decord
from PIL import Image
import torchvision.transforms as tf

from pyiqa.utils.registry import DATASET_REGISTRY
from pyiqa.data.transforms import transform_mapping, PairedToTensor
from .base_iqa_dataset import BaseIQADataset


@DATASET_REGISTRY.register()
class AIGVQAMOSDataset(BaseIQADataset):
    """AIGV MOS Dataset for AI-Generated Video Quality Assessment.

    Dataset format: Excel file with video paths and MOS scores.
    Videos are organized by groups (G1, G2, G3) and model names.

    Each video has 4 MOS scores from different annotators.
    The final score can be computed as mean or use individual scores.

    Args:
        opt (dict): Config with the following keys:
            dataroot (str): Root directory of the dataset
            meta_info_file (str): Path to the MOS xlsx file
            video_folder (str): Folder containing video groups (default: 'MOS_Videos')
            num_frames (int): Number of frames to sample from each video
            mos_range (list): Range of MOS values [min, max]
            lower_better (bool): Whether lower MOS is better
            score_mode (str): 'mean' or 'all' - use mean or all individual scores
    """

    def init_path_mos(self, opt):
        meta_file = opt['meta_info_file']
        
        # Determine file type and read accordingly
        if meta_file.endswith('.xlsx') or meta_file.endswith('.xls'):
            # Original Excel format
            self.meta_info = pd.read_excel(meta_file, header=None)
            has_header = False
        else:
            # CSV format (our prepared format with header)
            try:
                self.meta_info = pd.read_csv(meta_file)
                has_header = True
            except:
                # Try without header
                self.meta_info = pd.read_csv(meta_file, header=None)
                has_header = False

        dataroot = opt['dataroot']
        video_folder = opt.get('video_folder', 'MOS_Videos')
        # If video_folder is None or empty, use dataroot directly
        if video_folder:
            video_base_path = osp.join(dataroot, video_folder)
        else:
            video_base_path = dataroot
        self.score_mode = opt.get('score_mode', 'mean')

        self.paths_mos = []
        
        if has_header:
            # CSV format with header
            for _, row in self.meta_info.iterrows():
                video_path_rel = str(row['video_path'])
                video_path = osp.join(video_base_path, video_path_rel)
                
                # Support different column names for scores
                if 'mos' in row and 'score' not in row:
                    mos_mean = float(row['mos'])
                    mos_scores = [mos_mean, mos_mean, mos_mean, mos_mean]
                elif all(c in row for c in ['score1', 'score2', 'score3', 'score4']):
                    mos_scores = [float(row['score1']), float(row['score2']), 
                                  float(row['score3']), float(row['score4'])]
                    mos_mean = sum(mos_scores) / len(mos_scores)
                else:
                    mos_mean = float(row.get('mos', row.get('score', 0)))
                    mos_scores = [mos_mean, mos_mean, mos_mean, mos_mean]

                self.paths_mos.append({
                    'video_path': video_path,
                    'mos_scores': mos_scores,
                    'mos_mean': mos_mean
                })
        else:
            # Original Excel format (no header)
            for row in self.meta_info.values:
                video_path_rel = row[0]
                video_path = osp.join(video_base_path, video_path_rel)
                
                mos_scores = [float(row[i]) for i in range(1, 5)]
                
                self.paths_mos.append({
                    'video_path': video_path,
                    'mos_scores': mos_scores,
                    'mos_mean': sum(mos_scores) / len(mos_scores)
                })

    def get_split(self, opt):
        """AIGVQAMOSDataset uses all data (no train/val/test split column).
        
        Override parent method to skip split filtering.
        """
        pass  # Use all data from xlsx

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

    def mos_normalize(self, opt):
        mos_range = opt.get('mos_range', None)
        mos_lower_better = opt.get('lower_better', None)
        mos_normalize = opt.get('mos_normalize', False)
        target_range = opt.get('normalize_target_range', [0, 1])

        if mos_normalize:
            assert mos_range is not None and mos_lower_better is not None, (
                'mos_range and lower_better should be provided when mos_normalize is True'
            )

            def normalize(mos_label):
                # First normalize to [0, 1]
                mos_label = (mos_label - mos_range[0]) / (mos_range[1] - mos_range[0])
                if mos_lower_better:
                    mos_label = 1 - mos_label
                # Then scale to target range
                mos_label = mos_label * (target_range[1] - target_range[0]) + target_range[0]
                return mos_label

            for item in self.paths_mos:
                if self.score_mode == 'mean':
                    item['mos_mean'] = normalize(item['mos_mean'])
                else:
                    item['mos_scores'] = [normalize(s) for s in item['mos_scores']]

            self.logger.info(
                f'mos_label is normalized from {mos_range} to {target_range}, lower_better[{mos_lower_better}].'
            )

    def __getitem__(self, index):
        item = self.paths_mos[index]
        video_path = item['video_path']
        num_frames = self.opt.get('num_frames', 8)

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
        frames = vr.get_batch(frame_indices).asnumpy()

        frame_pils = [Image.fromarray(frame).convert('RGB') for frame in frames]

        transformed_frames = []
        for frame_pil in frame_pils:
            frame_tensor = self.trans(frame_pil) * self.img_range
            transformed_frames.append(frame_tensor)

        video_tensor = torch.stack(transformed_frames, dim=0)

        result = {
            'video': video_tensor,
            'video_path': video_path,
        }

        if self.score_mode == 'mean':
            result['mos_label'] = torch.Tensor([item['mos_mean']])
        elif self.score_mode == 'all':
            result['mos_label'] = torch.Tensor(item['mos_scores'])
        else:
            result['mos_label'] = torch.Tensor([item['mos_mean']])

        return result

    def __len__(self):
        return len(self.paths_mos)
