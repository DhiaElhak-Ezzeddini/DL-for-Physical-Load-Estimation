"""
Video Dataset for VideoMAE Training
====================================
Optimized dataset classes compatible with Hugging Face VideoMAE.

Author: VLoad Project
Date: 2026
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

try:
    from transformers import VideoMAEImageProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from decord import VideoReader, cpu # type: ignore
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


# =============================================================================
# Video Loading Utilities
# =============================================================================

def get_video_info(video_path: str) -> int:
    """Get total frame count from video."""
    if DECORD_AVAILABLE:
        return len(VideoReader(video_path, ctx=cpu(0)))
    elif CV2_AVAILABLE:
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    raise ImportError("Either decord or cv2 is required")


def load_frames(video_path: str, indices: List[int], size: int = 224) -> np.ndarray:
    """Load specific frames from video and resize."""
    if DECORD_AVAILABLE:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(indices).asnumpy()
        if frames.shape[1:3] != (size, size) and CV2_AVAILABLE:
            frames = np.stack([cv2.resize(f, (size, size)) for f in frames])
        return frames
    
    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size))
            else:
                frame = frames[-1].copy() if frames else np.zeros((size, size, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()
        return np.stack(frames)
    
    raise ImportError("Either decord or cv2 is required")


def compute_clip_indices(
    total_frames: int,
    num_frames: int,
    clip_idx: int,
    num_clips: int,
    frame_sample_rate: int = 4,
    use_middle_60: bool = False,
) -> List[int]:
    """Compute frame indices for a specific clip."""
    if use_middle_60:
        # Use middle 60% of video
        start_frame = int(total_frames * 0.2)
        end_frame = int(total_frames * 0.8)
        if end_frame - start_frame < num_frames:
            start_frame, end_frame = 0, total_frames
    else:
        start_frame, end_frame = 0, total_frames
    
    middle_length = end_frame - start_frame
    clip_length = num_frames * frame_sample_rate
    
    if middle_length <= clip_length:
        return list(np.linspace(start_frame, end_frame - 1, num_frames, dtype=int))
    
    usable_length = middle_length - clip_length
    if num_clips == 1:
        clip_start = start_frame + usable_length // 2
    else:
        segment_size = usable_length / (num_clips - 1)
        clip_start = int(start_frame + clip_idx * segment_size)
    
    clip_start = min(clip_start, start_frame + usable_length)
    indices = [clip_start + i * frame_sample_rate for i in range(num_frames)]
    return [min(idx, end_frame - 1) for idx in indices]


# =============================================================================
# Base Video Dataset
# =============================================================================

class BaseVideoDataset(Dataset):
    """Base class for video datasets with common functionality."""
    
    def __init__(
        self,
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        image_size: int = 224,
        num_clips: int = 8,
        augment: bool = True,
        processor: Optional["VideoMAEImageProcessor"] = None,
    ):
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.image_size = image_size
        self.num_clips = num_clips
        self.augment = augment
        
        if processor is None and HF_AVAILABLE:
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        else:
            self.processor = processor
    
    def _load_clip(self, video_path: str, clip_idx: int, use_middle_60: bool = False) -> np.ndarray:
        """Load a specific clip from video."""
        total_frames = get_video_info(video_path)
        indices = compute_clip_indices(
            total_frames, self.num_frames, clip_idx, self.num_clips,
            self.frame_sample_rate, use_middle_60
        )
        return load_frames(video_path, indices, self.image_size)
    
    def _process_video(self, video: np.ndarray) -> torch.Tensor:
        """Process video array to tensor."""
        if self.processor is not None:
            inputs = self.processor([video[i] for i in range(len(video))], return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)
        
        video = video.astype(np.float32) / 255.0
        video = (video - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(video).permute(3, 0, 1, 2).float()
    
    def _apply_augmentation(self, video: np.ndarray, flip_only: bool = False) -> np.ndarray:
        """Apply data augmentation."""
        if random.random() > 0.5:
            video = video[:, :, ::-1, :].copy()
        if not flip_only and random.random() > 0.5:
            video = video[::-1].copy()
        return video


# =============================================================================
# Pretraining Dataset
# =============================================================================

class VideoMAEPretrainingDataset(BaseVideoDataset):
    """Dataset for VideoMAE self-supervised pretraining."""
    
    def __init__(self, video_paths: List[str], **kwargs):
        super().__init__(**kwargs)
        self.video_paths = video_paths
        self.total_samples = len(video_paths) * self.num_clips
        print(f"Pretraining dataset: {len(video_paths)} videos x {self.num_clips} clips = {self.total_samples} samples")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, clip_idx = idx // self.num_clips, idx % self.num_clips
        video_path = self.video_paths[video_idx]
        
        try:
            random.seed(hash(video_path) + clip_idx)
            video = self._load_clip(video_path, clip_idx)
            random.seed()
            
            if self.augment:
                video = self._apply_augmentation(video)
            
            return {"pixel_values": self._process_video(video)}
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return {"pixel_values": torch.zeros(3, self.num_frames, self.image_size, self.image_size)}


# =============================================================================
# Classification Dataset
# =============================================================================

class VideoMAEClassificationDataset(BaseVideoDataset):
    """Dataset for VideoMAE video classification with operator-based split."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        val_operators: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        sampling_strategy: str = "multi_clip",
        **kwargs,
    ):
        # Adjust num_clips based on strategy and split
        if sampling_strategy == "uniform":
            kwargs["num_clips"] = 1
        elif split != "train":
            kwargs["num_clips"] = 1
        
        super().__init__(augment=kwargs.pop("augment", True) and split == "train", **kwargs)
        
        self.root = Path(root)
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.val_operators = val_operators or ["ope7", "ope8"]
        
        self.class_names, self.samples = self._discover_samples(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.total_samples = len(self.samples) * self.num_clips
        
        operators = {op for _, _, op in self.samples}
        print(f"Classification ({split}): {len(self.samples)} videos x {self.num_clips} clips = {self.total_samples} samples")
        print(f"  Classes: {self.class_names}, Operators: {sorted(operators)}, Strategy: {sampling_strategy}")
    
    def _discover_samples(self, class_names: Optional[List[str]]) -> Tuple[List[str], List[Tuple[str, int, str]]]:
        """Discover video samples split by operator."""
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        
        if class_names is None:
            class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        
        samples = []
        for class_name in class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            
            class_idx = class_names.index(class_name)
            for video_path in class_dir.rglob("*"):
                if video_path.suffix.lower() not in extensions:
                    continue
                
                operator = next((p for p in video_path.parts if p.startswith("ope") and len(p) <= 5), "unknown")
                is_val = operator in self.val_operators
                
                if (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    samples.append((str(video_path), class_idx, operator))
        
        return class_names, samples
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, clip_idx = idx // self.num_clips, idx % self.num_clips
        video_path, label, _ = self.samples[video_idx]
        
        try:
            video = self._load_clip(video_path, clip_idx, use_middle_60=True)
            
            if self.augment:
                video = self._apply_augmentation(video, flip_only=True)
            
            return {
                "pixel_values": self._process_video(video),
                "labels": torch.tensor(label, dtype=torch.long),
            }
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return {
                "pixel_values": torch.zeros(3, self.num_frames, self.image_size, self.image_size),
                "labels": torch.tensor(label, dtype=torch.long),
            }


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_pretraining_dataloader(
    video_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 16,
    image_size: int = 224,
    num_clips: int = 8,
    rank: int = 0,
    world_size: int = 1,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for pretraining (supports distributed)."""
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_paths = [str(p) for p in Path(video_dir).rglob("*") if p.suffix.lower() in extensions]
    
    if rank == 0:
        print(f"Found {len(video_paths)} videos for pretraining")
    
    dataset = VideoMAEPretrainingDataset(
        video_paths=video_paths,
        num_frames=num_frames,
        image_size=image_size,
        num_clips=num_clips,
        **kwargs,
    )
    
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True) if world_size > 1 else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_classification_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 16,
    image_size: int = 224,
    val_operators: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    num_clips: int = 8,
    sampling_strategy: str = "multi_clip",
    rank: int = 0,
    world_size: int = 1,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for classification (supports distributed)."""
    val_operators = val_operators or ["ope7", "ope8"]
    
    if rank == 0:
        print(f"Operator-based split: val_operators={val_operators}, Strategy: {sampling_strategy}")
    
    common = dict(num_frames=num_frames, image_size=image_size, val_operators=val_operators,
                  class_names=class_names, sampling_strategy=sampling_strategy, **kwargs)
    
    train_ds = VideoMAEClassificationDataset(data_dir, split="train", num_clips=num_clips, augment=True, **common)
    val_ds = VideoMAEClassificationDataset(data_dir, split="val", num_clips=1, augment=False, **common)
    
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


# Aliases for backward compatibility
create_pretraining_dataloader_distributed = create_pretraining_dataloader
create_classification_dataloaders_distributed = create_classification_dataloaders
