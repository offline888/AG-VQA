#!/usr/bin/env python3
"""
Prepare datasets for pyiqa framework integration.

This script converts dataset metadata into pyiqa-compatible formats:
1. T2VQA: CSV with video_path|text|mos (with optional split columns)
2. AIGVQA: Excel-style CSV with video_path|score columns
3. VideoFeedback2: JSONL with conversation data and quality scores
"""

import json
import os
import pandas as pd
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_t2vqa(output_dir: str, split_ratio: str = '802', seed: int = 123):
    """Prepare T2VQA dataset for pyiqa.
    
    Converts info.txt to CSV format with train/val/test split columns.
    
    Args:
        output_dir: Directory to save the prepared meta_info CSV
        split_ratio: Split ratio string like '802' (80% train, 0% val, 20% test)
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    info_file = '/home/dzc/yuanhao/data/T2VQA-DB/info.txt'
    output_file = os.path.join(output_dir, 'meta_info_T2VQADataset.csv')
    
    print("Preparing T2VQA dataset...")
    
    # Read original info.txt
    data = []
    with open(info_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                video_name = parts[0].strip()
                text = parts[1].strip()
                mos = float(parts[2].strip())
                data.append({
                    'video_path': video_name,
                    'text': text,
                    'mos': mos
                })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples from T2VQA")
    
    # Generate train/val/test splits
    random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    # Parse split ratio
    if len(split_ratio) == 3:
        train_ratio = int(split_ratio[0]) / 10
        val_ratio = int(split_ratio[1]) / 10
        test_ratio = int(split_ratio[2]) / 10
    else:
        raise ValueError(f"Invalid split_ratio: {split_ratio}")
    
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    
    # Assign splits
    split_col = f'ratio{split_ratio}_seed{seed}_split_00'
    df[split_col] = 'train'
    df.loc[indices[n_train:n_train + n_val], split_col] = 'val'
    df.loc[indices[n_train + n_val:], split_col] = 'test'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved T2VQA meta_info to {output_file}")
    print(f"  Train: {sum(df[split_col] == 'train')}")
    print(f"  Val: {sum(df[split_col] == 'val')}")
    print(f"  Test: {sum(df[split_col] == 'test')}")
    
    return output_file


def prepare_aigvqa(output_dir: str, score_mode: str = 'mean', split_ratio: str = '801', seed: int = 123):
    """Prepare AIGVQA MOS dataset for pyiqa.
    
    Converts mos.xlsx to CSV format with video_path and mos columns.
    Supports both 'mean' (average of 4 annotators) and 'all' (all 4 scores).
    
    Args:
        output_dir: Directory to save the prepared meta_info CSV
        score_mode: 'mean' for average MOS, 'all' for individual scores
        split_ratio: Split ratio string
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    xlsx_file = '/home/dzc/yuanhao/data/AIGV-DB/MOS/mos.xlsx'
    output_file = os.path.join(output_dir, 'meta_info_AIGVQAMOSDataset.csv')
    
    print("Preparing AIGVQA MOS dataset...")
    
    # Read original xlsx
    df = pd.read_excel(xlsx_file, header=None)
    
    # The xlsx has format: [video_path, score1, score2, score3, score4]
    df.columns = ['video_path', 'score1', 'score2', 'score3', 'score4']
    
    # Calculate mean score
    df['mos'] = df[['score1', 'score2', 'score3', 'score4']].mean(axis=1)
    
    print(f"Loaded {len(df)} samples from AIGVQA MOS")
    
    # Generate train/val/test splits
    random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    # Parse split ratio
    if len(split_ratio) == 3:
        train_ratio = int(split_ratio[0]) / 10
        val_ratio = int(split_ratio[1]) / 10
        test_ratio = int(split_ratio[2]) / 10
    else:
        raise ValueError(f"Invalid split_ratio: {split_ratio}")
    
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    
    # Assign splits
    split_col = f'ratio{split_ratio}_seed{seed}_split_00'
    df[split_col] = 'train'
    df.loc[indices[n_train:n_train + n_val], split_col] = 'val'
    df.loc[indices[n_train + n_val:], split_col] = 'test'
    
    # Keep all relevant columns
    df = df[['video_path', 'score1', 'score2', 'score3', 'score4', 'mos', split_col]]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved AIGVQA MOS meta_info to {output_file}")
    print(f"  Train: {sum(df[split_col] == 'train')}")
    print(f"  Val: {sum(df[split_col] == 'val')}")
    print(f"  Test: {sum(df[split_col] == 'test')}")
    
    # Also save individual scores for 'all' mode
    if score_mode == 'all':
        all_scores_file = os.path.join(output_dir, 'meta_info_AIGVQAMOSDataset_all.csv')
        df_all = df.copy()
        # Rename to match expected format
        df_all.columns = ['video_path', 'mos1', 'mos2', 'mos3', 'mos4', 'mos_mean', split_col]
        df_all.to_csv(all_scores_file, index=False)
        print(f"Saved AIGVQA all-scores meta_info to {all_scores_file}")
    
    return output_file


def prepare_video_feedback2(output_dir: str, split_ratio: str = '801', seed: int = 123):
    """Prepare VideoFeedback2 dataset for pyiqa.
    
    Converts JSON files to CSV format with video_path, text, and quality scores.
    
    Args:
        output_dir: Directory to save the prepared meta_info CSV
        split_ratio: Split ratio string
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = '/home/dzc/yuanhao/data/VideoFeedback2/data_27k_test (VideoScoreBench-v2).json'
    output_file = os.path.join(output_dir, 'meta_info_VideoFeedback2Dataset.csv')
    output_jsonl = os.path.join(output_dir, 'meta_info_VideoFeedback2Dataset.jsonl')
    
    print("Preparing VideoFeedback2 dataset...")
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"Loaded {len(json_data)} samples from VideoFeedback2")
    
    # Convert to DataFrame format
    data = []
    base_dir = '/home/dzc/yuanhao/data/VideoFeedback2'
    
    for item in json_data:
        video_name = item.get('video_name', '')
        
        # Find the actual video folder by checking if the file exists
        # Different suffix patterns are spread across different folders
        folder = None
        for i in range(1, 7):
            test_folder = f'videos_27k_{i}'
            test_path = os.path.join(base_dir, test_folder, f'{video_name}.mp4')
            if os.path.exists(test_path):
                folder = test_folder
                break
        
        # If video not found locally, use folder from URL pattern (for downloaded videos)
        if folder is None:
            video_url = item.get('video_url', '')
            # Extract folder from URL if available
            # Format: .../ModelName/video_name.mp4
            url_parts = video_url.split('/')
            if len(url_parts) >= 2:
                model_name = url_parts[-2] if url_parts[-1].endswith('.mp4') else url_parts[-1]
                # Map common model names to folders
                model_to_folder = {
                    'lavie_base': 'videos_27k_1',
                    'minimax_video': 'videos_27k_2', 
                    'cogvideo': 'videos_27k_3',
                    'model': 'videos_27k_4',
                    'zeroscope': 'videos_27k_5',
                }
                folder = model_to_folder.get(model_name, 'videos_27k_1')
            else:
                folder = 'videos_27k_1'  # Default
        
        video_rel_path = f"{folder}/{video_name}.mp4"
        
        data.append({
            'video_path': video_rel_path,
            'text': item.get('prompt', ''),
            'vq_score': item.get('visual_score', 0),
            't2v_score': item.get('t2v_score', 0),
            'pc_score': item.get('phy_score', 0),
        })
    
    df = pd.DataFrame(data)
    
    # Generate train/val/test splits
    random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    # Parse split ratio
    if len(split_ratio) == 3:
        train_ratio = int(split_ratio[0]) / 10
        val_ratio = int(split_ratio[1]) / 10
        test_ratio = int(split_ratio[2]) / 10
    else:
        raise ValueError(f"Invalid split_ratio: {split_ratio}")
    
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    
    # Assign splits
    split_col = f'ratio{split_ratio}_seed{seed}_split_00'
    df[split_col] = 'train'
    df.loc[indices[n_train:n_train + n_val], split_col] = 'val'
    df.loc[indices[n_train + n_val:], split_col] = 'test'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved VideoFeedback2 meta_info to {output_file}")
    print(f"  Train: {sum(df[split_col] == 'train')}")
    print(f"  Val: {sum(df[split_col] == 'val')}")
    print(f"  Test: {sum(df[split_col] == 'test')}")
    
    # Also save as JSONL for compatibility with VideoFeedback2Dataset
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in json_data:
            # Create conversation-style format
            video_name = item.get('video_name', '')
            
            # Determine video folder
            if video_name.endswith('_h'):
                folder = 'videos_27k_1'
            elif video_name.endswith('_p'):
                folder = 'videos_27k_2'
            else:
                folder = 'videos_27k_1'
            
            entry = {
                'videos': [f"{folder}/{video_name}.mp4"],
                'conversations': [
                    {'from': 'human', 'value': item.get('prompt', '')},
                    {'from': 'gpt', 'value': f'(1) visual quality: {item.get("visual_score", 0)}\n(2) text-to-video alignment: {item.get("t2v_score", 0)}\n(3) physical/common-sense consistency: {item.get("phy_score", 0)}'}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved VideoFeedback2 JSONL to {output_jsonl}")
    
    return output_file, output_jsonl


def create_dataset_configs(output_dir: str, meta_info_dir: str):
    """Create pyiqa dataset configuration YAML files.
    
    Args:
        output_dir: Directory to save YAML configs
        meta_info_dir: Directory containing meta_info files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # T2VQA config
    t2vqa_config = """# T2VQA Dataset Configuration for pyiqa
dataset_name: T2VQA
type: T2VQADataset

# Data paths
dataroot: /home/dzc/yuanhao/data/T2VQA-DB
meta_info_file: {meta_info_dir}/meta_info_T2VQADataset.csv
video_folder: ~  # Videos are directly in dataroot

# Video processing
num_frames: 8

# MOS settings (raw score range: 0-100)
mos_range: [0, 100]
lower_better: false
mos_normalize: false

# Split settings
split_ratio: "801"
split_index: 0

# Augmentation for training
augment:
  resize:
    size: [224, 224]
  hflip: ~

# Image range
img_range: 1.0
"""
    
    # AIGVQA MOS config
    aigvqa_config = """# AIGVQA MOS Dataset Configuration for pyiqa
dataset_name: AIGVQA_MOS
type: AIGVQAMOSDataset

# Data paths
dataroot: /home/dzc/yuanhao/data/AIGV-DB/MOS
meta_info_file: {meta_info_dir}/meta_info_AIGVQAMOSDataset.csv
video_folder: MOS_Videos

# Video processing
num_frames: 8

# MOS settings (raw score range: 0-100)
mos_range: [0, 100]
lower_better: false
mos_normalize: true
normalize_target_range: [1, 5]  # Normalize to 1-5 scale

# Score mode: 'mean' or 'all'
score_mode: mean

# Split settings
split_ratio: "801"
split_index: 0

# Augmentation for training
augment:
  resize:
    size: [224, 224]
  hflip: ~

# Image range
img_range: 1.0
"""
    
    # VideoFeedback2 config (using CSV)
    vfq_csv_config = """# VideoFeedback2 Dataset Configuration for pyiqa (CSV format)
dataset_name: VideoFeedback2
type: VideoFeedback2Dataset

# Data paths
dataroot: /home/dzc/yuanhao/data/VideoFeedback2
meta_info_file: {meta_info_dir}/meta_info_VideoFeedback2Dataset.csv
video_folder: ~

# Video processing
num_frames: 8

# Quality dimension: 'vq', 't2v', 'pc', or 'all'
dimension: vq

# MOS settings (raw score range: 1-5)
mos_range: [1, 5]
lower_better: false
mos_normalize: false

# Split settings
split_ratio: "801"
split_index: 0

# Augmentation for training
augment:
  resize:
    size: [224, 224]
  hflip: ~

# Image range
img_range: 1.0
"""
    
    # VideoFeedback2 config (using JSONL)
    vfq_jsonl_config = """# VideoFeedback2 Dataset Configuration for pyiqa (JSONL format)
dataset_name: VideoFeedback2_JSONL
type: VideoFeedback2Dataset

# Data paths
dataroot: /home/dzc/yuanhao/data/VideoFeedback2
meta_info_file: {meta_info_dir}/meta_info_VideoFeedback2Dataset.jsonl
video_folder: ~

# Video processing
num_frames: 8

# Quality dimension: 'vq', 't2v', 'pc', or 'all'
dimension: vq

# MOS settings (raw score range: 1-5)
mos_range: [1, 5]
lower_better: false
mos_normalize: false

# Split settings (JSONL uses all data, no split)
# To use splits with JSONL, provide split_file separately

# Augmentation for training
augment:
  resize:
    size: [224, 224]
  hflip: ~

# Image range
img_range: 1.0
"""
    
    # Write configs
    with open(os.path.join(output_dir, 'dataset_t2vqa.yml'), 'w') as f:
        f.write(t2vqa_config.format(meta_info_dir=meta_info_dir))
    
    with open(os.path.join(output_dir, 'dataset_aigvqa_mos.yml'), 'w') as f:
        f.write(aigvqa_config.format(meta_info_dir=meta_info_dir))
    
    with open(os.path.join(output_dir, 'dataset_videofeedback2_csv.yml'), 'w') as f:
        f.write(vfq_csv_config.format(meta_info_dir=meta_info_dir))
    
    with open(os.path.join(output_dir, 'dataset_videofeedback2_jsonl.yml'), 'w') as f:
        f.write(vfq_jsonl_config.format(meta_info_dir=meta_info_dir))
    
    print(f"Created dataset configuration files in {output_dir}")


def main():
    """Main function to prepare all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare datasets for pyiqa')
    parser.add_argument('--output_dir', type=str, default='./prepared_datasets',
                        help='Output directory for prepared datasets')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for train/val/test splits')
    parser.add_argument('--split_ratio', type=str, default='801',
                        help='Split ratio (e.g., "801" for 80% train, 0% val, 20% test)')
    parser.add_argument('--skip_t2vqa', action='store_true',
                        help='Skip T2VQA dataset')
    parser.add_argument('--skip_aigvqa', action='store_true',
                        help='Skip AIGVQA dataset')
    parser.add_argument('--skip_videofeedback2', action='store_true',
                        help='Skip VideoFeedback2 dataset')
    parser.add_argument('--configs_only', action='store_true',
                        help='Only create config files (assume meta_info already exists)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    meta_info_dir = os.path.join(output_dir, 'meta_info')
    
    os.makedirs(meta_info_dir, exist_ok=True)
    
    if not args.configs_only:
        # Prepare T2VQA
        if not args.skip_t2vqa:
            prepare_t2vqa(meta_info_dir, split_ratio=args.split_ratio, seed=args.seed)
        
        # Prepare AIGVQA
        if not args.skip_aigvqa:
            prepare_aigvqa(meta_info_dir, score_mode='mean', 
                          split_ratio=args.split_ratio, seed=args.seed)
        
        # Prepare VideoFeedback2
        if not args.skip_videofeedback2:
            prepare_video_feedback2(meta_info_dir, 
                                   split_ratio=args.split_ratio, seed=args.seed)
    
    # Create dataset config YAMLs
    configs_dir = os.path.join(output_dir, 'configs')
    create_dataset_configs(configs_dir, meta_info_dir)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"Meta info files: {meta_info_dir}")
    print(f"Config files: {configs_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
