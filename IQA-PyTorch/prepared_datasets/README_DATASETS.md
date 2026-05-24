# Dataset Preparation Guide for PyIQA

This guide explains how to prepare and use the three video quality assessment datasets with the PyIQA framework.

## Datasets Overview

| Dataset | Type | Samples | Quality Dimensions |
|---------|------|---------|-------------------|
| T2VQA | Text-to-Video Quality Assessment | 10,000 | Overall MOS (0-100) |
| AIGVQA | AI-Generated Video Quality | 2,808 | Overall MOS (0-100) |
| VideoFeedback2 | Multi-dimensional VQA | 500 | VQ, T2V, PC (1-5 each) |

## Directory Structure

```
prepared_datasets/
├── meta_info/
│   ├── meta_info_T2VQADataset.csv        # T2VQA metadata
│   ├── meta_info_AIGVQAMOSDataset.csv    # AIGVQA metadata
│   ├── meta_info_VideoFeedback2Dataset.csv # VideoFeedback2 metadata (CSV)
│   └── meta_info_VideoFeedback2Dataset.jsonl # VideoFeedback2 metadata (JSONL)
├── configs/
│   ├── dataset_t2vqa.yml                 # T2VQA config
│   ├── dataset_aigvqa_mos.yml            # AIGVQA config
│   ├── dataset_videofeedback2_csv.yml    # VideoFeedback2 CSV config
│   └── dataset_videofeedback2_jsonl.yml  # VideoFeedback2 JSONL config
└── README_DATASETS.md                    # This file
```

## Quick Start

### 1. Test Dataset Compatibility

```bash
cd /home/dzc/yuanhao/IQA-PyTorch
python test_dataset_compatibility.py
```

Expected output:
```
T2VQADataset: PASSED
AIGVQAMOSDataset: PASSED
VideoFeedback2Dataset: PASSED
```

### 2. Use in Training Configuration

Example YAML configuration for training:

```yaml
# In your train.yaml
datasets:
  train:
    name: T2VQA
    type: T2VQADataset
    dataroot: /home/dzc/yuanhao/data/T2VQA-DB
    meta_info_file: ./prepared_datasets/meta_info/meta_info_T2VQADataset.csv
    video_folder: ~  # Videos are directly in dataroot

    num_frames: 8
    phase: train
    split_ratio: "801"
    split_index: 0

    mos_range: [0, 100]
    lower_better: false
    mos_normalize: false

    augment:
      resize:
        size: [224, 224]
      hflip: ~

    img_range: 1.0
    batch_size_per_gpu: 16
    num_worker_per_gpu: 8

  val:
    name: T2VQA_val
    type: T2VQADataset
    dataroot: /home/dzc/yuanhao/data/T2VQA-DB
    meta_info_file: ./prepared_datasets/meta_info/meta_info_T2VQADataset.csv
    video_folder: ~
    num_frames: 8
    phase: val
    split_ratio: "801"
    split_index: 0
    mos_range: [0, 100]
    lower_better: false
    mos_normalize: false
    img_range: 1.0
    batch_size_per_gpu: 16
    num_worker_per_gpu: 8
```

### 3. Programmatic Usage

```python
from pyiqa.data import build_dataset

# T2VQA
t2vqa_opt = {
    'name': 'T2VQA',
    'type': 'T2VQADataset',
    'dataroot': '/home/dzc/yuanhao/data/T2VQA-DB',
    'meta_info_file': './prepared_datasets/meta_info/meta_info_T2VQADataset.csv',
    'num_frames': 8,
    'phase': 'train',
    'split_ratio': '801',
    'split_index': 0,
    'mos_range': [0, 100],
    'lower_better': False,
    'mos_normalize': False,
    'augment': {'resize': {'size': [224, 224]}},
    'img_range': 1.0,
}
dataset = build_dataset(t2vqa_opt)
sample = dataset[0]
print(sample.keys())  # dict_keys(['video', 'mos_label', 'text', 'video_path'])

# AIGVQA
aigvqa_opt = {
    'name': 'AIGVQA',
    'type': 'AIGVQAMOSDataset',
    'dataroot': '/home/dzc/yuanhao/data/AIGV-DB/MOS',
    'meta_info_file': './prepared_datasets/meta_info/meta_info_AIGVQAMOSDataset.csv',
    'video_folder': 'MOS_Videos',
    'num_frames': 8,
    'phase': 'train',
    'mos_range': [0, 100],
    'lower_better': False,
    'mos_normalize': True,
    'normalize_target_range': [1, 5],
    'score_mode': 'mean',
    'augment': {'resize': {'size': [224, 224]}},
    'img_range': 1.0,
}
dataset = build_dataset(aigvqa_opt)

# VideoFeedback2
vfq_opt = {
    'name': 'VideoFeedback2',
    'type': 'VideoFeedback2Dataset',
    'dataroot': '/home/dzc/yuanhao/data/VideoFeedback2',
    'meta_info_file': './prepared_datasets/meta_info/meta_info_VideoFeedback2Dataset.csv',
    'video_folder': None,  # Paths in CSV include subfolder
    'num_frames': 8,
    'dimension': 'vq',  # or 't2v', 'pc', 'all'
    'phase': 'train',
    'mos_range': [1, 5],
    'lower_better': False,
    'mos_normalize': False,
    'augment': {'resize': {'size': [224, 224]}},
    'img_range': 1.0,
}
dataset = build_dataset(vfq_opt)
```

## Dataset Details

### T2VQA (Text-to-Video Quality Assessment)

- **Source**: `/home/dzc/yuanhao/data/T2VQA-DB/info.txt`
- **Format**: CSV with columns `video_path|text|mos|split`
- **Video Location**: Videos are directly in dataroot (`/home/dzc/yuanhao/data/T2VQA-DB/*.mp4`)
- **MOS Range**: 0-100 (higher is better)
- **Default Split**: 80% train, 0% val, 20% test
- **Sample Output**:
  ```python
  {
      'video': torch.Tensor([8, 3, 224, 224]),  # 8 frames
      'mos_label': torch.Tensor([35.5]),          # Normalized MOS
      'text': "A dog running in the park",
      'video_path': "/path/to/video.mp4"
  }
  ```

### AIGVQA MOS (AI-Generated Video Quality)

- **Source**: `/home/dzc/yuanhao/data/AIGV-DB/MOS/mos.xlsx`
- **Format**: CSV with columns `video_path|score1|score2|score3|score4|mos|split`
- **Video Location**: `MOS_Videos/G{1,2,3}/{model_name}/*.mp4`
- **MOS Range**: 0-100 (4 annotators), normalized to 1-5 for training
- **Default Split**: 80% train, 0% val, 20% test
- **Sample Output**:
  ```python
  {
      'video': torch.Tensor([8, 3, 224, 224]),
      'video_path': "G1/floor33/100000.mp4",
      'mos_label': torch.Tensor([3.2]),  # Normalized to [1, 5]
  }
  ```

### VideoFeedback2

- **Source**: `/home/dzc/yuanhao/data/VideoFeedback2/data_27k_test (VideoScoreBench-v2).json`
- **Format**: CSV with columns `video_path|text|vq_score|t2v_score|pc_score|split`
- **Video Location**: `videos_27k_{1-6}/*.mp4`
- **Quality Dimensions**:
  - `vq`: Visual Quality (1-5)
  - `t2v`: Text-to-Video Alignment (1-5)
  - `pc`: Physical Consistency (1-5)
- **Default Split**: 80% train, 0% val, 20% test
- **Sample Output**:
  ```python
  {
      'video': torch.Tensor([8, 3, 224, 224]),
      'text': "A person dancing",
      'video_path': "videos_27k_1/001033_p.mp4",
      'mos_label': torch.Tensor([3.0]),  # VQ score (or dimension specified)
  }
  ```

## Re-generating Splits

To generate new random splits with different seeds:

```bash
python prepare_datasets.py --output_dir ./prepared_datasets --seed 456 --split_ratio "703"
```

## Notes

1. **VideoFeedback2 Test Set**: The test set videos may require downloading from URLs provided in the original JSON. Currently, the test configuration uses the `data_27k_test (VideoScoreBench-v2).json` which contains only metadata without actual video files.

2. **MOS Normalization**: By default, `mos_normalize: false`. Set to `true` with appropriate `mos_range` and `normalize_target_range` to normalize scores to a different scale.

3. **Multi-dimensional Training**: For VideoFeedback2, use `dimension: all` to train on all three quality dimensions simultaneously. The model should handle multi-target output.

4. **Frame Sampling**: All datasets use uniform frame sampling (`total_frames / num_frames`) by default. This can be modified in the dataset class for random sampling.
