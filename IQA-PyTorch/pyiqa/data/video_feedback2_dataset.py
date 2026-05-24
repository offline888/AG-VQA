import json
import re
from os import path as osp

import pandas as pd
import torch
import decord
from PIL import Image
import torchvision.transforms as tf

from pyiqa.utils.registry import DATASET_REGISTRY
from pyiqa.utils import get_root_logger
from pyiqa.data.transforms import transform_mapping, PairedToTensor
from .base_iqa_dataset import BaseIQADataset


@DATASET_REGISTRY.register()
class VideoFeedback2Dataset(BaseIQADataset):
    """VideoFeedback2 Dataset for Multi-dimensional Video Quality Assessment.

    Supports two formats:
    1. JSONL format with conversation data containing quality scores
    2. CSV format with pre-extracted scores (our prepared format)

    Each sample has three quality dimensions (1-5 scale):
        - Visual Quality: Video's visual/optical properties
        - Text-to-Video Alignment: Alignment between prompt and video content
        - Physical Consistency: Physical/common-sense consistency

    Args:
        opt (dict): Config with the following keys:
            dataroot (str): Root directory of the dataset
            meta_info_file (str): Path to the JSONL or CSV file with video info
            num_frames (int): Number of frames to sample from each video
            mos_range (list): Range of MOS values [min, max]
            lower_better (bool): Whether lower MOS is better
            dimension (str): Quality dimension to use ('vq', 't2v', 'pc', or 'all')
    """

    def __init__(self, opt):
        self.opt = opt
        self.logger = get_root_logger()

        if opt.get('override_phase', None) is None:
            self.phase = opt.get('phase', 'train')
        else:
            self.phase = opt['override_phase']

        assert self.phase in ['train', 'val', 'test'], (
            f'phase should be in [train, val, test], got {self.phase}'
        )

        self.init_path_mos(opt)
        self.mos_normalize(opt)
        self.get_split(opt)
        self.get_transforms(opt)

    def init_path_mos(self, opt):
        """Load data from JSONL or CSV file and extract quality scores."""
        dataroot = opt['dataroot']
        meta_file = opt['meta_info_file']
        video_folder = opt.get('video_folder', 'videos')

        # Handle video_folder: None, empty string, or '~' means videos are directly in dataroot
        if video_folder and video_folder != '~':
            video_base_path = osp.join(dataroot, video_folder)
        else:
            video_base_path = dataroot

        self.dimension = opt.get('dimension', 'vq')

        self.paths_mos = []

        # Determine file format based on extension
        if meta_file.endswith('.jsonl'):
            self._load_from_jsonl(meta_file, video_base_path)
        else:
            # Assume CSV format (our prepared format)
            self._load_from_csv(meta_file, video_base_path)

    def _load_from_jsonl(self, meta_file, video_base_path):
        """Load data from JSONL file and extract quality scores."""
        # Store empty DataFrame for consistency with CSV mode
        self.meta_info = pd.DataFrame()
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                videos = data.get('videos', [])
                if not videos:
                    continue

                video_name = videos[0]
                video_path = osp.join(video_base_path, video_name)

                prompt_text = ''
                conversations = data.get('conversations', [])
                for conv in conversations:
                    if conv.get('from') == 'human':
                        prompt_text = conv.get('value', '')
                        break

                scores = self._extract_scores_from_conversation(conversations)

                if scores:
                    self.paths_mos.append({
                        'video_path': video_path,
                        'text': prompt_text,
                        'vq_score': scores['vq'],
                        't2v_score': scores['t2v'],
                        'pc_score': scores['pc'],
                        'raw_data': data
                    })

    def _load_from_csv(self, meta_file, video_base_path):
        """Load data from CSV file with pre-extracted quality scores."""
        self.meta_info = pd.read_csv(meta_file)

        # Find column names for quality scores
        vq_col = 'vq_score' if 'vq_score' in self.meta_info.columns else 'visual_score'
        t2v_col = 't2v_score' if 't2v_score' in self.meta_info.columns else 't2v_score'
        pc_col = 'pc_score' if 'pc_score' in self.meta_info.columns else 'phy_score'
        text_col = 'text' if 'text' in self.meta_info.columns else 'prompt'

        for _, row in self.meta_info.iterrows():
            video_rel_path = str(row['video_path'])
            video_path = osp.join(video_base_path, video_rel_path)

            self.paths_mos.append({
                'video_path': video_path,
                'text': str(row.get(text_col, '')),
                'vq_score': float(row.get(vq_col, 0)),
                't2v_score': float(row.get(t2v_col, 0)),
                'pc_score': float(row.get(pc_col, 0)),
            })

    def _extract_scores_from_conversation(self, conversations):
        """Extract quality scores from gpt response in conversations."""
        scores = {'vq': None, 't2v': None, 'pc': None}

        score_pattern = r'\((\d+)\)\s*(?:visual quality|text-to-video alignment|physical/common-sense consistency)[^:]*:\s*(\d+)'

        for conv in conversations:
            if conv.get('from') == 'gpt':
                gpt_response = conv.get('value', '')
                matches = re.findall(score_pattern, gpt_response.lower())

                for match in matches:
                    dim_idx, score = match
                    score = float(score)

                    if 'visual quality' in gpt_response[max(0, gpt_response.lower().find(match[0])-100):gpt_response.lower().find(match[0])].lower():
                        scores['vq'] = score
                    elif 'text-to-video alignment' in gpt_response[max(0, gpt_response.lower().find(match[0])-100):gpt_response.lower().find(match[0])].lower():
                        scores['t2v'] = score
                    elif 'physical/common-sense consistency' in gpt_response[max(0, gpt_response.lower().find(match[0])-100):gpt_response.lower().find(match[0])].lower():
                        scores['pc'] = score

                break

        if all(v is not None for v in scores.values()):
            return scores
        return None

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
        """Normalize MOS scores to [0, 1] range."""
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
                if self.dimension == 'vq' or self.dimension == 'all':
                    item['vq_score'] = normalize(item['vq_score'])
                if self.dimension == 't2v' or self.dimension == 'all':
                    item['t2v_score'] = normalize(item['t2v_score'])
                if self.dimension == 'pc' or self.dimension == 'all':
                    item['pc_score'] = normalize(item['pc_score'])

            self.logger.info(
                f'mos_label is normalized from {mos_range}, lower_better[{mos_lower_better}] to [0, 1], higher better.'
            )

    def get_split(self, opt):
        """Read train/val/test splits from CSV columns or split file."""
        # Check for split file first (pickle format)
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            import pickle
            with open(split_file_path, 'rb') as f:
                split_dict = pickle.load(f)
                split_index = opt.get('split_index', 1)
                splits = split_dict[split_index][self.phase]
            self.paths_mos = [self.paths_mos[i] for i in splits]
            return

        # Check for split column in meta_info (our CSV format)
        # Skip if meta_info is empty (e.g., JSONL mode without split columns)
        if self.meta_info.empty:
            return
            
        all_split_lists = [x for x in self.meta_info.columns.tolist() if 'split' in x]

        split_index = opt.get('split_index', None)

        if split_index is not None and len(all_split_lists) > 0:
            if isinstance(split_index, str):
                split_name = split_index
            elif isinstance(split_index, int):
                split_ratio = opt.get('split_ratio', '801')
                split_name = f'ratio{split_ratio}_seed123_split_{split_index:02d}'

            # Also check without padding for split_00
            if split_name not in all_split_lists:
                split_name = f'ratio{split_ratio}_seed123_split_00'

            if split_name in all_split_lists:
                split_paths_mos = []
                for i in range(len(self.paths_mos)):
                    if self.meta_info[split_name][i] == self.phase:
                        split_paths_mos.append(self.paths_mos[i])
                self.paths_mos = split_paths_mos
                self.logger.info(f'Using split: {split_name}, phase: {self.phase}, samples: {len(self.paths_mos)}')

    def __getitem__(self, index):
        item = self.paths_mos[index]
        video_path = item['video_path']
        text_description = item['text']

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
            'text': text_description,
            'video_path': video_path,
        }

        if self.dimension == 'vq':
            result['mos_label'] = torch.Tensor([item['vq_score']])
        elif self.dimension == 't2v':
            result['mos_label'] = torch.Tensor([item['t2v_score']])
        elif self.dimension == 'pc':
            result['mos_label'] = torch.Tensor([item['pc_score']])
        elif self.dimension == 'all':
            result['mos_label'] = torch.Tensor([item['vq_score'], item['t2v_score'], item['pc_score']])

        return result

    def __len__(self):
        return len(self.paths_mos)
