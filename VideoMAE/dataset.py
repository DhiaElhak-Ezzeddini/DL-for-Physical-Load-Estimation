"""
Video Dataset for VideoMAE Training
====================================
Dataset classes compatible with Hugging Face VideoMAE.

Author: VLoad Project
Date: 2026
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    from transformers import VideoMAEImageProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available")

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

def load_video_cv2(
    video_path: str,
    num_frames: int = 16,
    frame_sample_rate: int = 4,
    resize: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load video using OpenCV.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        frame_sample_rate: Sample every N frames
        resize: Output size (H, W)
        
    Returns:
        Video array (T, H, W, C) in RGB format, values [0, 255]
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for video loading")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices
    if total_frames <= num_frames * frame_sample_rate:
        # Video too short, sample uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Sample with frame_sample_rate
        start = random.randint(0, total_frames - num_frames * frame_sample_rate)
        indices = [start + i * frame_sample_rate for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            # Repeat last valid frame
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
        else:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, (resize[1], resize[0]))
        
        frames.append(frame)
    
    cap.release()
    
    # Stack frames: (T, H, W, C)
    video = np.stack(frames, axis=0)
    
    return video


def load_video_decord(
    video_path: str,
    num_frames: int = 16,
    frame_sample_rate: int = 4,
    resize: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load video using Decord (faster than OpenCV).
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        frame_sample_rate: Sample every N frames
        resize: Output size (H, W)
        
    Returns:
        Video array (T, H, W, C) in RGB format
    """
    if not DECORD_AVAILABLE:
        raise ImportError("Decord is required for video loading")
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Calculate frame indices
    if total_frames <= num_frames * frame_sample_rate:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        start = random.randint(0, total_frames - num_frames * frame_sample_rate)
        indices = [start + i * frame_sample_rate for i in range(num_frames)]
    
    # Get frames
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    
    # Resize if needed
    if frames.shape[1:3] != resize:
        if CV2_AVAILABLE:
            resized = []
            for frame in frames:
                resized.append(cv2.resize(frame, (resize[1], resize[0])))
            frames = np.stack(resized, axis=0)
    
    return frames


def load_video(
    video_path: str,
    num_frames: int = 16,
    frame_sample_rate: int = 4,
    resize: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load video using best available backend.
    
    Returns:
        Video array (T, H, W, C)
    """
    if DECORD_AVAILABLE:
        return load_video_decord(video_path, num_frames, frame_sample_rate, resize)
    elif CV2_AVAILABLE:
        return load_video_cv2(video_path, num_frames, frame_sample_rate, resize)
    else:
        raise ImportError("Either decord or cv2 is required for video loading")


# =============================================================================
# Multi-Clip Video Loading
# =============================================================================

def load_video_multi_clip_cv2(
    video_path: str,
    num_frames: int = 16,
    frame_sample_rate: int = 4,
    num_clips: int = 4,
    resize: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """
    Load multiple clips from a video using OpenCV.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames per clip
        frame_sample_rate: Sample every N frames
        num_clips: Number of clips to extract
        resize: Output size (H, W)
        
    Returns:
        List of video arrays, each (T, H, W, C) in RGB format
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for video loading")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_length = num_frames * frame_sample_rate
    
    # Calculate start positions for each clip
    if total_frames <= clip_length:
        # Video too short, use same frames for all clips
        start_positions = [0] * num_clips
    else:
        # Divide video into num_clips segments and sample from each
        segment_length = (total_frames - clip_length) // num_clips
        start_positions = []
        for i in range(num_clips):
            segment_start = i * segment_length
            segment_end = segment_start + segment_length
            # Random start within segment
            start = random.randint(segment_start, max(segment_start, min(segment_end, total_frames - clip_length)))
            start_positions.append(start)
    
    clips = []
    for start in start_positions:
        # Calculate frame indices for this clip
        if total_frames <= clip_length:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = [start + i * frame_sample_rate for i in range(num_frames)]
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (resize[1], resize[0]))
            
            frames.append(frame)
        
        clips.append(np.stack(frames, axis=0))
    
    cap.release()
    return clips


def load_video_multi_clip_decord(
    video_path: str,
    num_frames: int = 16,
    frame_sample_rate: int = 4,
    num_clips: int = 4,
    resize: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """
    Load multiple clips from a video using Decord.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames per clip
        frame_sample_rate: Sample every N frames
        num_clips: Number of clips to extract
        resize: Output size (H, W)
        
    Returns:
        List of video arrays, each (T, H, W, C) in RGB format
    """
    if not DECORD_AVAILABLE:
        raise ImportError("Decord is required for video loading")
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    clip_length = num_frames * frame_sample_rate
    
    # Calculate start positions for each clip
    if total_frames <= clip_length:
        start_positions = [0] * num_clips
    else:
        segment_length = (total_frames - clip_length) // num_clips
        start_positions = []
        for i in range(num_clips):
            segment_start = i * segment_length
            segment_end = segment_start + segment_length
            start = random.randint(segment_start, max(segment_start, min(segment_end, total_frames - clip_length)))
            start_positions.append(start)
    
    clips = []
    for start in start_positions:
        if total_frames <= clip_length:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = [start + i * frame_sample_rate for i in range(num_frames)]
        
        frames = vr.get_batch(indices).asnumpy()
        
        # Resize if needed
        if frames.shape[1:3] != resize:
            if CV2_AVAILABLE:
                resized = []
                for frame in frames:
                    resized.append(cv2.resize(frame, (resize[1], resize[0])))
                frames = np.stack(resized, axis=0)
        
        clips.append(frames)
    
    return clips


def load_video_multi_clip(
    video_path: str,
    num_frames: int = 16,
    frame_sample_rate: int = 4,
    num_clips: int = 4,
    resize: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """
    Load multiple clips from a video using best available backend.
    
    Returns:
        List of video arrays, each (T, H, W, C)
    """
    if DECORD_AVAILABLE:
        return load_video_multi_clip_decord(video_path, num_frames, frame_sample_rate, num_clips, resize)
    elif CV2_AVAILABLE:
        return load_video_multi_clip_cv2(video_path, num_frames, frame_sample_rate, num_clips, resize)
    else:
        raise ImportError("Either decord or cv2 is required for video loading")


# =============================================================================
# Video Dataset for Pretraining (Multi-Clip)
# =============================================================================

class VideoMAEPretrainingDataset(Dataset):
    """
    Dataset for VideoMAE self-supervised pretraining.
    
    Loads videos and prepares them for masked autoencoding.
    Supports multi-clip sampling to get more training samples per video.
    """
    
    def __init__(
        self,
        video_paths: List[str],
        processor: Optional["VideoMAEImageProcessor"] = None,
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        image_size: int = 224,
        num_clips: int = 4,
        augment: bool = True,
    ):
        """
        Initialize pretraining dataset.
        
        Args:
            video_paths: List of paths to video files
            processor: HuggingFace VideoMAE processor
            num_frames: Number of frames to sample per clip
            frame_sample_rate: Sample every N frames
            image_size: Output image size
            num_clips: Number of clips to extract per video
            augment: Apply data augmentation
        """
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.image_size = image_size
        self.num_clips = num_clips
        self.augment = augment
        
        if processor is None and HF_AVAILABLE:
            self.processor = VideoMAEImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base"
            )
        else:
            self.processor = processor
        
        # Total samples = videos * clips_per_video
        self.total_samples = len(video_paths) * num_clips
        print(f"Pretraining dataset: {len(video_paths)} videos x {num_clips} clips = {self.total_samples} samples")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map idx to video and clip index
        video_idx = idx // self.num_clips
        clip_idx = idx % self.num_clips
        video_path = self.video_paths[video_idx]
        
        try:
            # Load all clips from this video (cached approach would be better for performance)
            # For now, load single clip by adjusting the start position
            video = self._load_single_clip(video_path, clip_idx)
            
            # Apply augmentation
            if self.augment:
                video = self._augment(video)
            
            # Process with HF processor
            if self.processor is not None:
                frames = [video[i] for i in range(video.shape[0])]
                inputs = self.processor(frames, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)
            else:
                video = video.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                video = (video - mean) / std
                pixel_values = torch.from_numpy(video).permute(3, 0, 1, 2).float()
            
            return {"pixel_values": pixel_values}
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return {"pixel_values": torch.zeros(3, self.num_frames, self.image_size, self.image_size)}
    
    def _load_single_clip(self, video_path: str, clip_idx: int) -> np.ndarray:
        """Load a specific clip from a video."""
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
        elif CV2_AVAILABLE:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            raise ImportError("Either decord or cv2 is required")
        
        clip_length = self.num_frames * self.frame_sample_rate
        
        # Calculate start position for this specific clip
        if total_frames <= clip_length:
            start = 0
        else:
            # Divide video into num_clips segments
            segment_length = (total_frames - clip_length) // self.num_clips
            segment_start = clip_idx * segment_length
            segment_end = segment_start + segment_length
            # Use deterministic offset within segment based on clip_idx for consistency
            # But add some randomness for training
            random.seed(hash(video_path) + clip_idx)  # Reproducible per video+clip
            start = random.randint(segment_start, max(segment_start, min(segment_end, total_frames - clip_length)))
            random.seed()  # Reset seed
        
        # Calculate frame indices
        if total_frames <= clip_length:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = [start + i * self.frame_sample_rate for i in range(self.num_frames)]
        
        # Load frames
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = vr.get_batch(indices).asnumpy()
            if frames.shape[1:3] != (self.image_size, self.image_size):
                if CV2_AVAILABLE:
                    resized = [cv2.resize(f, (self.image_size, self.image_size)) for f in frames]
                    frames = np.stack(resized, axis=0)
        else:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                else:
                    frame = frames[-1].copy() if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                frames.append(frame)
            cap.release()
            frames = np.stack(frames, axis=0)
        
        return frames
    
    def _augment(self, video: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            video = video[:, :, ::-1, :].copy()
        
        # Random temporal reverse
        if random.random() > 0.5:
            video = video[::-1].copy()
        
        return video


# =============================================================================
# Video Dataset for Classification (Multi-Clip)
# =============================================================================

class VideoMAEClassificationDataset(Dataset):
    """
    Dataset for VideoMAE video classification.
    
    Supports multi-clip sampling for training data augmentation.
    Uses OPERATOR-BASED split to avoid data leakage.
    
    Expects folder structure:
        root/
            class1/
                ope1/
                    carry/
                        video1.mp4
                    walk/
                        video1.mp4
                ope2/
                    ...
            class2/
                ope1/
                    ...
    """
    
    def __init__(
        self,
        root: str,
        processor: Optional["VideoMAEImageProcessor"] = None,
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        image_size: int = 224,
        split: str = "train",
        val_operators: Optional[List[str]] = None,
        augment: bool = True,
        class_names: Optional[List[str]] = None,
        num_clips: int = 4,
        sampling_strategy: str = "multi_clip",
    ):
        """
        Initialize classification dataset with operator-based split.
        
        Args:
            root: Root directory of dataset
            processor: HuggingFace VideoMAE processor
            num_frames: Number of frames to sample per clip
            frame_sample_rate: Sample every N frames (only used for multi_clip strategy)
            image_size: Output image size
            split: "train" or "val"
            val_operators: List of operator names for validation (e.g., ["ope7", "ope8"])
                          If None, defaults to ["ope7", "ope8"]
            augment: Apply augmentation (only for train)
            class_names: Optional list of class names
            num_clips: Number of clips to extract per video (only for multi_clip strategy)
            sampling_strategy: Sampling strategy for frames:
                - "multi_clip": 4 equally-spaced clips from middle 60% of video
                - "uniform": 16 uniformly sampled frames from middle 60% (1 clip)
        """
        self.root = Path(root)
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == "train")
        self.sampling_strategy = sampling_strategy
        
        # For uniform strategy, always 1 clip; for multi_clip, use num_clips for train
        if sampling_strategy == "uniform":
            self.num_clips = 1
        else:
            self.num_clips = num_clips if split == "train" else 1
        
        # Default validation operators
        if val_operators is None:
            val_operators = ["ope7", "ope8"]
        self.val_operators = val_operators
        
        if processor is None and HF_AVAILABLE:
            self.processor = VideoMAEImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base"
            )
        else:
            self.processor = processor
        
        # Discover classes and videos with operator info
        self.class_names, self.samples = self._discover_samples_by_operator(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Total samples = videos * clips_per_video
        self.total_samples = len(self.samples) * self.num_clips
        
        # Count operators
        operators_in_split = set()
        for video_path, _, operator in self.samples:
            operators_in_split.add(operator)
        
        print(f"Classification dataset ({split}): {len(self.samples)} videos x {self.num_clips} clips = {self.total_samples} samples")
        print(f"  Classes: {self.class_names}")
        print(f"  Operators: {sorted(operators_in_split)}")
        print(f"  Sampling strategy: {self.sampling_strategy}")
    
    def _discover_samples_by_operator(
        self,
        class_names: Optional[List[str]] = None
    ) -> Tuple[List[str], List[Tuple[str, int, str]]]:
        """
        Discover video samples and split by operator.
        
        Returns:
            class_names: List of class names
            samples: List of (video_path, class_idx, operator_name)
        """
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        
        if class_names is None:
            class_names = sorted([
                d.name for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])
        
        samples = []
        for class_name in class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            
            class_idx = class_names.index(class_name)
            
            # Find all videos and extract operator from path
            for video_path in class_dir.rglob("*"):
                if video_path.suffix.lower() not in video_extensions:
                    continue
                
                # Extract operator from path (e.g., .../empty/ope1/carry/video.mp4)
                operator = self._extract_operator(video_path)
                
                # Filter based on split
                is_val_operator = operator in self.val_operators
                
                if self.split == "val" and is_val_operator:
                    samples.append((str(video_path), class_idx, operator))
                elif self.split == "train" and not is_val_operator:
                    samples.append((str(video_path), class_idx, operator))
        
        return class_names, samples
    
    def _extract_operator(self, video_path: Path) -> str:
        """
        Extract operator name from video path.
        
        Expected path: .../class/opeX/action/video.mp4
        Returns: "opeX" or "unknown"
        """
        parts = video_path.parts
        for part in parts:
            if part.startswith("ope") and len(part) <= 5:  # ope1, ope2, ..., ope8
                return part
        return "unknown"
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx = idx // self.num_clips
        clip_idx = idx % self.num_clips
        video_path, label, _ = self.samples[video_idx]
        
        try:
            video = self._load_single_clip(video_path, clip_idx)
            
            if self.augment:
                video = self._augment(video)
            
            if self.processor is not None:
                frames = [video[i] for i in range(video.shape[0])]
                inputs = self.processor(frames, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)
            else:
                video = video.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                video = (video - mean) / std
                pixel_values = torch.from_numpy(video).permute(3, 0, 1, 2).float()
            
            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return {
                "pixel_values": torch.zeros(3, self.num_frames, self.image_size, self.image_size),
                "labels": torch.tensor(label, dtype=torch.long),
            }
    
    def _load_single_clip(self, video_path: str, clip_idx: int) -> np.ndarray:
        """
        Load a specific clip from a video using the configured sampling strategy.
        
        Strategies:
        - "multi_clip": Extract 4 equally-spaced clips from middle 60% of video
        - "uniform": Sample 16 frames uniformly from middle 60% of video (1 clip only)
        """
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
        elif CV2_AVAILABLE:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            raise ImportError("Either decord or cv2 is required")
        
        # Calculate middle 60% range (skip first/last 20%)
        skip_ratio = 0.20
        start_frame = int(total_frames * skip_ratio)
        end_frame = int(total_frames * (1 - skip_ratio))
        middle_length = end_frame - start_frame
        
        # Ensure we have enough frames
        if middle_length < self.num_frames:
            # Fall back to full video if middle portion is too short
            start_frame = 0
            end_frame = total_frames
            middle_length = total_frames
        
        if self.sampling_strategy == "uniform":
            # UNIFORM STRATEGY: Sample 16 frames uniformly from middle 60%
            indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        
        else:  # "multi_clip" strategy
            # MULTI_CLIP STRATEGY: 4 equally-spaced clips from middle 60%
            clip_length = self.num_frames * self.frame_sample_rate
            
            if middle_length <= clip_length:
                # Not enough frames for proper clips, sample uniformly
                indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
            else:
                # Divide middle portion into num_clips equal segments
                # Each clip starts at a different position to cover the movement
                usable_length = middle_length - clip_length
                
                if self.num_clips == 1:
                    # Single clip: center of middle portion
                    clip_start = start_frame + usable_length // 2
                else:
                    # Multiple clips: equally distributed start positions
                    segment_size = usable_length / (self.num_clips - 1) if self.num_clips > 1 else 0
                    clip_start = int(start_frame + clip_idx * segment_size)
                
                # Ensure clip_start is valid
                clip_start = min(clip_start, start_frame + usable_length)
                
                # Sample frames with frame_sample_rate spacing
                indices = [clip_start + i * self.frame_sample_rate for i in range(self.num_frames)]
                indices = [min(idx, end_frame - 1) for idx in indices]  # Clamp to valid range
        
        # Load frames
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = vr.get_batch(indices).asnumpy()
            if frames.shape[1:3] != (self.image_size, self.image_size):
                if CV2_AVAILABLE:
                    resized = [cv2.resize(f, (self.image_size, self.image_size)) for f in frames]
                    frames = np.stack(resized, axis=0)
        else:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                else:
                    frame = frames[-1].copy() if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                frames.append(frame)
            cap.release()
            frames = np.stack(frames, axis=0)
        
        return frames
    
    def _augment(self, video: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            video = video[:, :, ::-1, :].copy()
        
        return video


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_pretraining_dataloader(
    video_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 16,
    image_size: int = 224,
    num_clips: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for pretraining.
    
    Args:
        video_dir: Directory containing video files
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_frames: Number of frames per video
        image_size: Image size
        num_clips: Number of clips to extract per video (default: 4)
        
    Returns:
        DataLoader
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_dir = Path(video_dir)
    
    video_paths = [
        str(p) for p in video_dir.rglob("*")
        if p.suffix.lower() in video_extensions
    ]
    
    print(f"Found {len(video_paths)} videos for pretraining")
    
    dataset = VideoMAEPretrainingDataset(
        video_paths=video_paths,
        num_frames=num_frames,
        image_size=image_size,
        num_clips=num_clips,
        **kwargs,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
    num_clips: int = 4,
    sampling_strategy: str = "multi_clip",
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for classification.
    
    Uses OPERATOR-BASED split to prevent data leakage.
    
    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_frames: Number of frames per video
        image_size: Image size
        val_operators: Operators to use for validation (default: ["ope7", "ope8"])
        class_names: Optional list of class names
        num_clips: Number of clips to extract per video (default: 4, only for multi_clip)
        sampling_strategy: Sampling strategy for frames:
            - "multi_clip": 4 equally-spaced clips from middle 60% of video
            - "uniform": 16 uniformly sampled frames from middle 60% (1 clip)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if val_operators is None:
        val_operators = ["ope7", "ope8"]
    
    print(f"Operator-based split: val_operators={val_operators}")
    print(f"Sampling strategy: {sampling_strategy}")
    
    train_dataset = VideoMAEClassificationDataset(
        root=data_dir,
        num_frames=num_frames,
        image_size=image_size,
        split="train",
        val_operators=val_operators,
        augment=True,
        class_names=class_names,
        num_clips=num_clips,
        sampling_strategy=sampling_strategy,
        **kwargs,
    )
    
    val_dataset = VideoMAEClassificationDataset(
        root=data_dir,
        num_frames=num_frames,
        image_size=image_size,
        split="val",
        val_operators=val_operators,
        augment=False,
        class_names=class_names,
        num_clips=1,  # Validation always uses single clip
        sampling_strategy=sampling_strategy,
        **kwargs,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
