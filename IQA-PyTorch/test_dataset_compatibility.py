#!/usr/bin/env python3
"""
Test script to verify pyiqa dataset compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyiqa.data import build_dataset
import torch

def test_t2vqa():
    """Test T2VQADataset."""
    print("\n" + "="*60)
    print("Testing T2VQADataset...")
    print("="*60)
    
    dataset_opt = {
        'name': 'T2VQA_test',
        'type': 'T2VQADataset',
        'dataroot': '/home/dzc/yuanhao/data/T2VQA-DB',
        'meta_info_file': './prepared_datasets/meta_info/meta_info_T2VQADataset.csv',
        'num_frames': 8,
        'mos_range': [0, 100],
        'lower_better': False,
        'mos_normalize': False,
        'split_ratio': '801',
        'split_index': 0,
        'phase': 'train',
        'augment': {
            'resize': {'size': [224, 224]},
        },
        'img_range': 1.0,
    }
    
    try:
        dataset = build_dataset(dataset_opt)
        print(f"Dataset created successfully! Length: {len(dataset)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Video shape: {sample['video'].shape}")
        print(f"MOS label: {sample['mos_label'].item():.4f}")
        print(f"Text (first 50 chars): {sample['text'][:50]}...")
        print("\nT2VQADataset: PASSED")
        return True
    except Exception as e:
        print(f"T2VQADataset: FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aigvqa():
    """Test AIGVQAMOSDataset."""
    print("\n" + "="*60)
    print("Testing AIGVQAMOSDataset...")
    print("="*60)
    
    dataset_opt = {
        'name': 'AIGVQA_test',
        'type': 'AIGVQAMOSDataset',
        'dataroot': '/home/dzc/yuanhao/data/AIGV-DB/MOS',
        'meta_info_file': './prepared_datasets/meta_info/meta_info_AIGVQAMOSDataset.csv',
        'video_folder': 'MOS_Videos',
        'num_frames': 8,
        'mos_range': [0, 100],
        'lower_better': False,
        'mos_normalize': True,
        'normalize_target_range': [1, 5],
        'score_mode': 'mean',
        'split_ratio': '801',
        'split_index': 0,
        'phase': 'train',
        'augment': {
            'resize': {'size': [224, 224]},
        },
        'img_range': 1.0,
    }
    
    try:
        dataset = build_dataset(dataset_opt)
        print(f"Dataset created successfully! Length: {len(dataset)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Video shape: {sample['video'].shape}")
        print(f"MOS label: {sample['mos_label'].item():.4f}")
        print("\nAIGVQAMOSDataset: PASSED")
        return True
    except Exception as e:
        print(f"AIGVQAMOSDataset: FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_videofeedback2():
    """Test VideoFeedback2Dataset."""
    print("\n" + "="*60)
    print("Testing VideoFeedback2Dataset...")
    print("="*60)
    
    # Note: video_folder should be empty/None to use paths from CSV directly
    dataset_opt = {
        'name': 'VideoFeedback2_test',
        'type': 'VideoFeedback2Dataset',
        'dataroot': '/home/dzc/yuanhao/data/VideoFeedback2',
        'meta_info_file': './prepared_datasets/meta_info/meta_info_VideoFeedback2Dataset.csv',
        'video_folder': None,  # Video paths in CSV already include subfolder
        'num_frames': 8,
        'dimension': 'vq',
        'mos_range': [1, 5],
        'lower_better': False,
        'mos_normalize': False,
        'split_ratio': '801',
        'split_index': 0,
        'phase': 'train',
        'augment': {
            'resize': {'size': [224, 224]},
        },
        'img_range': 1.0,
    }
    
    try:
        dataset = build_dataset(dataset_opt)
        print(f"Dataset created successfully! Length: {len(dataset)}")
        
        # Check first sample data (don't load video since it may not exist locally)
        item = dataset.paths_mos[0]
        print(f"Sample keys: {item.keys()}")
        print(f"Video path: {item['video_path']}")
        print(f"VQ score: {item['vq_score']}")
        print(f"Text (first 50 chars): {item['text'][:50]}...")
        
        # Check if video exists (optional)
        import os
        if os.path.exists(item['video_path']):
            print("Video file exists locally")
            # Try loading a sample
            sample = dataset[0]
            print(f"Video shape: {sample['video'].shape}")
            print(f"MOS label: {sample['mos_label'].item():.4f}")
        else:
            print("Note: Video file not found locally (may need download)")
        
        print("\nVideoFeedback2Dataset: PASSED (dataset build)")
        return True
    except Exception as e:
        print(f"VideoFeedback2Dataset: FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("PyIQA Dataset Compatibility Test")
    print("="*60)
    
    results = []
    
    results.append(('T2VQADataset', test_t2vqa()))
    results.append(('AIGVQAMOSDataset', test_aigvqa()))
    results.append(('VideoFeedback2Dataset', test_videofeedback2()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
