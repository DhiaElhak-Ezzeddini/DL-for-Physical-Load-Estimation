"""
Dataset Augmentation Pipeline for VLoad Weight Estimation
==========================================================

This script performs the following operations:
1. Splits each video from the original dataset into N clips (configurable, default 4)
2. Resizes all frames to target size (224x224)
3. Applies horizontal flip augmentation to each clip

Original: 72 walk videos (3 classes × 8 operators × 1 action × 3 angles)
After split (4 clips): 288 videos (× 4)
After flip: 576 videos (× 2)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for dataset augmentation pipeline."""
    
    # Input/Output paths
    input_dir: str = "dataset_weight_estimation/dataset_weight_estimation"
    output_dir: str = "Augmented_Dataset"
    
    # Video processing
    num_splits: int = 4  # Number of clips to split each video into
    apply_horizontal_flip: bool = True
    
    # Resize options
    target_size: Tuple[int, int] = (224, 224)  # (height, width)
    interpolation: str = "bilinear"  # 'bilinear', 'nearest', 'area', 'cubic'
    
    # Padding options
    padding_mode: str = "loop"  # 'loop', 'reflect', 'last_frame', 'black'
    target_fps: Optional[int] = None  # None = keep original FPS
    
    # Output format
    output_format: str = "mp4"
    codec: str = "mp4v"  # 'mp4v', 'XVID', 'avc1'
    
    # Processing
    num_workers: int = 4
    
    # Classes and structure
    classes: List[str] = field(default_factory=lambda: ["empty", "light", "heavy"])
    operators: List[str] = field(default_factory=lambda: [f"ope{i}" for i in range(1, 9)])
    actions: List[str] = field(default_factory=lambda: ["carry", "walk"])
    angles: List[str] = field(default_factory=lambda: ["angle1", "angle2", "angle3"])


class VideoProcessor:
    """Handles individual video processing operations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def read_video(self, video_path: str) -> Tuple[np.ndarray, dict]:
        """
        Read video file and return frames with metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames array [N, H, W, C], metadata dict)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "source_path": video_path
        }
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")
        
        return np.array(frames), metadata
    
    def write_video(self, frames: np.ndarray, output_path: str, fps: int) -> bool:
        """
        Write frames to video file.
        
        Args:
            frames: Array of frames [N, H, W, C]
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            Success status
        """
        if len(frames) == 0:
            logger.error(f"No frames to write for {output_path}")
            return False
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Cannot create video writer for {output_path}")
            return False
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    
    def split_video(self, frames: np.ndarray) -> List[np.ndarray]:
        """
        Split video frames into N equal clips.
        
        Args:
            frames: Array of frames [N, H, W, C]
            
        Returns:
            List of frame arrays, one for each clip
        """
        num_splits = self.config.num_splits
        total_frames = len(frames)
        
        # Calculate frames per clip
        frames_per_clip = total_frames // num_splits
        
        clips = []
        for i in range(num_splits):
            start_idx = i * frames_per_clip
            # Last clip gets any remaining frames
            if i == num_splits - 1:
                end_idx = total_frames
            else:
                end_idx = start_idx + frames_per_clip
            
            clip = frames[start_idx:end_idx]
            if len(clip) > 0:  # Ensure non-empty clip
                clips.append(clip)
        
        return clips
    
    def resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Resize all frames to target size.
        
        Args:
            frames: Array of frames [N, H, W, C]
            
        Returns:
            Resized frames
        """
        target_h, target_w = self.config.target_size
        
        # Map interpolation string to OpenCV constant
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }
        interp = interp_map.get(self.config.interpolation, cv2.INTER_LINEAR)
        
        resized = np.array([
            cv2.resize(frame, (target_w, target_h), interpolation=interp)
            for frame in frames
        ])
        
        return resized
    
    def horizontal_flip(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply horizontal flip to all frames.
        
        Args:
            frames: Array of frames [N, H, W, C]
            
        Returns:
            Flipped frames
        """
        return np.array([cv2.flip(frame, 1) for frame in frames])
    
    def pad_video(self, frames: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad video to target length using specified padding mode.
        
        Args:
            frames: Array of frames [N, H, W, C]
            target_length: Target number of frames
            
        Returns:
            Padded frames
        """
        current_length = len(frames)
        
        if current_length >= target_length:
            return frames[:target_length]
        
        padding_needed = target_length - current_length
        
        if self.config.padding_mode == "loop":
            # Loop the video until target length
            repeats = (target_length // current_length) + 1
            padded = np.tile(frames, (repeats, 1, 1, 1))[:target_length]
            
        elif self.config.padding_mode == "reflect":
            # Reflect/mirror the video
            padded_frames = list(frames)
            while len(padded_frames) < target_length:
                # Alternate between forward and reverse
                if len(padded_frames) // current_length % 2 == 1:
                    padded_frames.extend(frames[::-1])
                else:
                    padded_frames.extend(frames)
            padded = np.array(padded_frames[:target_length])
            
        elif self.config.padding_mode == "last_frame":
            # Repeat the last frame
            last_frame = frames[-1:] 
            padding = np.repeat(last_frame, padding_needed, axis=0)
            padded = np.concatenate([frames, padding], axis=0)
            
        elif self.config.padding_mode == "black":
            # Pad with black frames
            h, w, c = frames[0].shape
            black_frames = np.zeros((padding_needed, h, w, c), dtype=frames.dtype)
            padded = np.concatenate([frames, black_frames], axis=0)
            
        else:
            raise ValueError(f"Unknown padding mode: {self.config.padding_mode}")
        
        return padded


class DatasetAugmentor:
    """Main class for dataset augmentation pipeline."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.processor = VideoProcessor(config)
        self.video_info: List[Dict] = []
        self.frame_counts: List[int] = []
        
    def discover_videos(self) -> List[Dict]:
        """
        Discover all videos in the input directory.
        
        Returns:
            List of video information dictionaries
        """
        videos = []
        input_path = Path(self.config.input_dir)
        
        for class_name in self.config.classes:
            for operator in self.config.operators:
                for action in self.config.actions:
                    for angle in self.config.angles:
                        video_name = f"{angle}.mp4"
                        video_path = input_path / class_name / operator / action / video_name
                        
                        if video_path.exists():
                            videos.append({
                                "path": str(video_path),
                                "class": class_name,
                                "operator": operator,
                                "action": action,
                                "angle": angle
                            })
                        else:
                            logger.warning(f"Video not found: {video_path}")
        
        logger.info(f"Discovered {len(videos)} videos")
        return videos
    
    def analyze_videos(self, videos: List[Dict]) -> Dict:
        """
        Analyze all videos to determine frame statistics.
        
        Args:
            videos: List of video info dictionaries
            
        Returns:
            Statistics dictionary
        """
        logger.info("Analyzing video frame counts...")
        
        frame_counts = []
        
        for video_info in tqdm(videos, desc="Analyzing videos"):
            cap = cv2.VideoCapture(video_info["path"])
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_counts.append(frame_count)
                video_info["frame_count"] = frame_count
                cap.release()
            else:
                logger.warning(f"Cannot open: {video_info['path']}")
        
        self.frame_counts = frame_counts
        
        # After splitting, each clip will have approximately 1/num_splits of the frames
        num_splits = self.config.num_splits
        clip_frame_counts = [fc // num_splits for fc in frame_counts]
        
        stats = {
            "total_videos": len(videos),
            "num_splits": num_splits,
            "original_frame_stats": {
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
                "mean": np.mean(frame_counts) if frame_counts else 0,
                "std": np.std(frame_counts) if frame_counts else 0
            },
            "after_split_stats": {
                "min": min(clip_frame_counts) if clip_frame_counts else 0,
                "max": max(clip_frame_counts) if clip_frame_counts else 0,
                "mean": np.mean(clip_frame_counts) if clip_frame_counts else 0
            },
            "target_frames": max(clip_frame_counts) if clip_frame_counts else 0
        }
        
        logger.info(f"Original frame counts: min={stats['original_frame_stats']['min']}, "
                   f"max={stats['original_frame_stats']['max']}, "
                   f"mean={stats['original_frame_stats']['mean']:.1f}")
        logger.info(f"After {num_splits}-way split: min={stats['after_split_stats']['min']}, "
                   f"max={stats['after_split_stats']['max']}")
        logger.info(f"Target frame count for padding: {stats['target_frames']}")
        
        return stats
    
    def process_single_video(
        self, 
        video_info: Dict
    ) -> List[Dict]:
        """
        Process a single video: split, resize, and augment.
        
        Args:
            video_info: Video information dictionary
            
        Returns:
            List of output video information
        """
        output_videos = []
        
        try:
            # Read video
            frames, metadata = self.processor.read_video(video_info["path"])
            fps = metadata["fps"]
            
            # Resize frames to target size (224x224)
            frames = self.processor.resize_frames(frames)
            
            # Split video into N clips
            clips = self.processor.split_video(frames)
            
            # Create output paths
            output_base = Path(self.config.output_dir)
            class_dir = output_base / video_info["class"] / video_info["operator"] / video_info["action"]
            
            # Process each clip
            for clip_idx, clip_frames in enumerate(clips, 1):
                clip_name = f"part{clip_idx}"
                
                # Save original (no padding)
                output_name = f"{video_info['angle']}_{clip_name}.{self.config.output_format}"
                output_path = class_dir / output_name
                
                if self.processor.write_video(clip_frames, str(output_path), fps):
                    output_videos.append({
                        "path": str(output_path),
                        "class": video_info["class"],
                        "operator": video_info["operator"],
                        "action": video_info["action"],
                        "angle": video_info["angle"],
                        "part": clip_name,
                        "augmentation": "original",
                        "frame_count": len(clip_frames)
                    })
                
                # Apply horizontal flip if enabled
                if self.config.apply_horizontal_flip:
                    flipped_frames = self.processor.horizontal_flip(clip_frames)
                    
                    output_name_flip = f"{video_info['angle']}_{clip_name}_hflip.{self.config.output_format}"
                    output_path_flip = class_dir / output_name_flip
                    
                    if self.processor.write_video(flipped_frames, str(output_path_flip), fps):
                        output_videos.append({
                            "path": str(output_path_flip),
                            "class": video_info["class"],
                            "operator": video_info["operator"],
                            "action": video_info["action"],
                            "angle": video_info["angle"],
                            "part": clip_name,
                            "augmentation": "horizontal_flip",
                            "frame_count": len(flipped_frames)
                        })
            
        except Exception as e:
            logger.error(f"Error processing {video_info['path']}: {str(e)}")
        
        return output_videos
    
    def run(self) -> Dict:
        """
        Execute the full augmentation pipeline.
        
        Returns:
            Summary statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting Dataset Augmentation Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Discover videos
        videos = self.discover_videos()
        
        if len(videos) == 0:
            logger.error("No videos found in input directory!")
            return {"status": "error", "message": "No videos found"}
        
        # Step 2: Analyze videos
        stats = self.analyze_videos(videos)
        
        # Step 3: Create output directory
        output_path = Path(self.config.output_dir)
        if output_path.exists():
            logger.warning(f"Output directory exists. Removing: {output_path}")
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 4: Process all videos (no padding applied)
        logger.info("Processing videos...")
        all_output_videos = []
        
        for video_info in tqdm(videos, desc="Augmenting videos"):
            output_videos = self.process_single_video(video_info)
            all_output_videos.extend(output_videos)
        
        # Step 5: Generate summary
        summary = self._generate_summary(videos, all_output_videos, stats)
        
        # Save metadata
        metadata_path = output_path / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "config": {
                    "input_dir": self.config.input_dir,
                    "output_dir": self.config.output_dir,
                    "num_splits": self.config.num_splits,
                    "horizontal_flip": self.config.apply_horizontal_flip,
                    "target_size": list(self.config.target_size),
                    "interpolation": self.config.interpolation
                },
                "statistics": stats,
                "summary": summary,
                "videos": all_output_videos
            }, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info("=" * 60)
        logger.info("Augmentation Pipeline Complete!")
        logger.info("=" * 60)
        
        return summary
    
    def _generate_summary(
        self, 
        input_videos: List[Dict], 
        output_videos: List[Dict],
        stats: Dict
    ) -> Dict:
        """Generate summary statistics."""
        
        # Count by class
        class_counts = {}
        for video in output_videos:
            cls = video["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Collect frame counts from output videos
        frame_counts = [v["frame_count"] for v in output_videos]
        
        summary = {
            "input_videos": len(input_videos),
            "output_videos": len(output_videos),
            "multiplication_factor": len(output_videos) / len(input_videos) if input_videos else 0,
            "num_splits": self.config.num_splits,
            "target_size": list(self.config.target_size),
            "videos_per_class": class_counts,
            "frame_count_stats": {
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
                "mean": np.mean(frame_counts) if frame_counts else 0
            },
            "augmentations_applied": [f"split_into_{self.config.num_splits}_clips", "resize"]
        }
        
        if self.config.apply_horizontal_flip:
            summary["augmentations_applied"].append("horizontal_flip")
        
        logger.info(f"Input videos: {summary['input_videos']}")
        logger.info(f"Output videos: {summary['output_videos']}")
        logger.info(f"Multiplication factor: {summary['multiplication_factor']}x")
        logger.info(f"Videos per class: {class_counts}")
        logger.info(f"Frame size: {self.config.target_size[0]}x{self.config.target_size[1]}")
        logger.info(f"Frame counts - min: {summary['frame_count_stats']['min']}, max: {summary['frame_count_stats']['max']}, mean: {summary['frame_count_stats']['mean']:.1f}")
        
        return summary


def main():
    """Main entry point for the augmentation pipeline."""
    
    # Configuration
    config = AugmentationConfig(
        input_dir="dataset_weight_estimation/dataset_weight_estimation",
        output_dir="Augmented_Dataset",
        num_splits=4,                      # Split each video into 4 clips
        apply_horizontal_flip=True,
        target_size=(224, 224),           # Resize to 224x224
        interpolation="bilinear",          # Interpolation method
        padding_mode="loop",               # Options: 'loop', 'reflect', 'last_frame', 'black'
        output_format="mp4",
        codec="mp4v"
    )
    
    # Run pipeline
    augmentor = DatasetAugmentor(config)
    summary = augmentor.run()
    
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"✓ Original videos: {summary['input_videos']}")
    print(f"✓ Augmented videos: {summary['output_videos']}")
    print(f"✓ Data multiplication: {summary['multiplication_factor']}x")
    print(f"✓ Frame size: {summary['target_size'][0]}x{summary['target_size'][1]}")
    print(f"✓ Frame counts - min: {summary['frame_count_stats']['min']}, max: {summary['frame_count_stats']['max']}")
    print(f"✓ Augmentations: {', '.join(summary['augmentations_applied'])}")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    main()
