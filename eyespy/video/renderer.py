import cv2
import numpy as np
import asyncio
import logging
import time
import os
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import uuid
from ..models import Keypoint
from ..config import settings
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor, gather_with_concurrency
from ..pose.validation import PoseValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RenderingConfig:
    """Configuration for video rendering"""
    # General settings
    output_dir: Path = Path(settings.TEMP_DIR) / "rendered"
    output_quality: int = 95  # JPEG quality for frames
    
    # Drawing settings
    skeleton_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    skeleton_thickness: int = 3
    joint_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    joint_radius: int = 4
    
    # Text settings
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    font_color: Tuple[int, int, int] = (255, 255, 255)  # White
    font_thickness: int = 1
    
    # Angle measurement settings
    angle_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    show_angles: bool = True
    
    # Performance settings
    batch_size: int = 30
    max_workers: int = 4

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class VideoRenderer:
    """Renders processed video with skeleton overlay and analytics"""
    
    def __init__(self, config: Optional[RenderingConfig] = None):
        """Initialize the video renderer with optional custom config"""
        self.config = config or RenderingConfig()
        self.validator = PoseValidator()
        
        # Define skeleton connections (pairs of keypoints to draw lines between)
        self.skeleton_connections = [
            # Torso
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),
            
            # Arms
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
            
            # Legs
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE"),
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ]
        
        # Define angles to measure and display
        self.angle_definitions = {
            "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
            "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
            "left_hip": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
            "right_hip": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
            "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        }
        
        # Angle assessment criteria (for analytics)
        self.angle_assessments = {
            "elbow": {
                "normal_range": (10, 160),
                "messages": {
                    "low": "Elbow extended too much",
                    "high": "Elbow bent too much",
                    "normal": "Good elbow angle"
                }
            },
            "knee": {
                "normal_range": (10, 170),
                "messages": {
                    "low": "Knee extended too much",
                    "high": "Knee bent too much",
                    "normal": "Good knee angle"
                }
            },
            "hip": {
                "normal_range": (80, 160),
                "messages": {
                    "low": "Forward hip angle too small",
                    "high": "Forward hip angle too large",
                    "normal": "Good hip alignment"
                }
            },
            "shoulder": {
                "normal_range": (50, 170),
                "messages": {
                    "low": "Shoulder rotated too much",
                    "high": "Shoulder extended too much",
                    "normal": "Good shoulder alignment"
                }
            }
        }
    
    async def render_video(
        self,
        frames: List[np.ndarray],
        keypoints_per_frame: List[List[Keypoint]],
        metadata: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Render a video with skeleton overlay and analytics
        
        Args:
            frames: List of video frames
            keypoints_per_frame: List of keypoints for each frame
            metadata: Video metadata
            output_filename: Optional filename for the output video
            
        Returns:
            Path to the rendered video file
        """
        start_time = time.time()
        logger.info(f"Starting video rendering for {len(frames)} frames")
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"rendered_{uuid.uuid4()}.mp4"
        
        output_path = self.config.output_dir / output_filename
        
        # Extract video properties
        if frames and len(frames) > 0:
            height, width = frames[0].shape[:2]
        else:
            raise ValueError("No frames to render")
        
        # Get the original FPS from metadata
        fps = metadata.get("fps", 30)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        try:
            # Process frames in batches to optimize performance
            total_frames = len(frames)
            batch_size = self.config.batch_size
            
            for i in range(0, total_frames, batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_keypoints = keypoints_per_frame[i:i + batch_size]
                
                # Render batch in parallel
                rendered_frames = await self._render_batch(
                    batch_frames,
                    batch_keypoints
                )
                
                # Write frames to video
                for frame in rendered_frames:
                    writer.write(frame)
                
                logger.debug(f"Rendered batch {i // batch_size + 1}/{(total_frames + batch_size - 1) // batch_size}")
                
                # Free memory
                del batch_frames
                del batch_keypoints
                del rendered_frames
                
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)
            
            logger.info(f"Video rendering completed in {time.time() - start_time:.2f} seconds")
            return str(output_path)
            
        finally:
            writer.release()
    
    async def _render_batch(
        self,
        frames: List[np.ndarray],
        keypoints_per_frame: List[List[Keypoint]]
    ) -> List[np.ndarray]:
        """
        Render a batch of frames in parallel
        
        Args:
            frames: List of frames to render
            keypoints_per_frame: List of keypoints for each frame
            
        Returns:
            List of rendered frames
        """
        # Use gather with concurrency to limit parallel processing
        render_tasks = []
        
        for frame, keypoints in zip(frames, keypoints_per_frame):
            task = self._render_frame(frame.copy(), keypoints)
            render_tasks.append(task)
        
        return await gather_with_concurrency(
            self.config.max_workers,
            *render_tasks
        )
    
    async def _render_frame(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """
        Render a single frame with skeleton overlay and analytics
        
        Args:
            frame: Video frame
            keypoints: List of keypoints for the frame
            
        Returns:
            Rendered frame
        """
        return await run_in_executor(
            self._render_frame_sync,
            frame,
            keypoints
        )
    
    def _render_frame_sync(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """
        Synchronous implementation of frame rendering
        
        Args:
            frame: Video frame
            keypoints: List of keypoints for the frame
            
        Returns:
            Rendered frame
        """
        # Skip if no keypoints
        if not keypoints:
            return frame
        
        # Convert keypoints to dictionary for easier access
        keypoint_dict = {kp.name: kp for kp in keypoints}
        
        # Draw skeleton
        self._draw_skeleton(frame, keypoint_dict)
        
        # Draw joints
        self._draw_joints(frame, keypoint_dict)
        
        # Draw angles
        if self.config.show_angles:
            angle_results = self._draw_angles(frame, keypoint_dict)
            
            # Add analytics based on angles
            self._add_analytics_text(frame, angle_results)
        
        return frame
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoint_dict: Dict[str, Keypoint]
    ) -> None:
        """
        Draw skeleton lines connecting keypoints
        
        Args:
            frame: Video frame
            keypoint_dict: Dictionary of keypoints
        """
        height, width = frame.shape[:2]
        
        for p1_name, p2_name in self.skeleton_connections:
            if p1_name in keypoint_dict and p2_name in keypoint_dict:
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.4 or p2.confidence < 0.4:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Draw line with thickness based on confidence
                confidence = min(p1.confidence, p2.confidence)
                thickness = max(1, int(self.config.skeleton_thickness * confidence))
                
                cv2.line(
                    frame,
                    p1_px,
                    p2_px,
                    self.config.skeleton_color,
                    thickness
                )
    
    def _draw_joints(
        self,
        frame: np.ndarray,
        keypoint_dict: Dict[str, Keypoint]
    ) -> None:
        """
        Draw circles at joint positions
        
        Args:
            frame: Video frame
            keypoint_dict: Dictionary of keypoints
        """
        height, width = frame.shape[:2]
        
        for name, keypoint in keypoint_dict.items():
            # Skip if confidence is too low
            if keypoint.confidence < 0.4:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            x_px = int(keypoint.x * width)
            y_px = int(keypoint.y * height)
            
            # Draw circle with radius based on confidence
            radius = max(1, int(self.config.joint_radius * keypoint.confidence))
            
            cv2.circle(
                frame,
                (x_px, y_px),
                radius,
                self.config.joint_color,
                -1  # Filled circle
            )
    
    def _draw_angles(
        self,
        frame: np.ndarray,
        keypoint_dict: Dict[str, Keypoint]
    ) -> Dict[str, float]:
        """
        Calculate and draw joint angles
        
        Args:
            frame: Video frame
            keypoint_dict: Dictionary of keypoints
            
        Returns:
            Dictionary of angle measurements
        """
        height, width = frame.shape[:2]
        angle_results = {}
        
        for angle_name, (p1_name, p2_name, p3_name) in self.angle_definitions.items():
            if all(p_name in keypoint_dict for p_name in [p1_name, p2_name, p3_name]):
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                p3 = keypoint_dict[p3_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.4 or p2.confidence < 0.4 or p3.confidence < 0.4:
                    continue
                
                # Calculate angle
                angle = self._calculate_angle(p1, p2, p3)
                angle_results[angle_name] = angle
                
                # Convert normalized coordinates to pixel coordinates
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Draw angle text
                cv2.putText(
                    frame,
                    f"{angle:.1f}°",
                    (p2_px[0] + 10, p2_px[1] + 10),
                    self.config.font,
                    self.config.font_scale,
                    self.config.angle_color,
                    self.config.font_thickness
                )
        
        return angle_results
    
    def _calculate_angle(
        self,
        p1: Keypoint,
        p2: Keypoint,
        p3: Keypoint
    ) -> float:
        """
        Calculate the angle between three points
        
        Args:
            p1: First point
            p2: Middle point (apex of the angle)
            p3: Third point
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _add_analytics_text(
        self,
        frame: np.ndarray,
        angle_results: Dict[str, float]
    ) -> None:
        """
        Add analytics text based on angle measurements
        
        Args:
            frame: Video frame
            angle_results: Dictionary of angle measurements
        """
        height, width = frame.shape[:2]
        
        # Create a semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (300, 120),
            (0, 0, 0),
            -1
        )
        
        # Apply overlay with transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add title
        cv2.putText(
            frame,
            "POSE ANALYSIS",
            (10, 20),
            self.config.font,
            self.config.font_scale * 1.5,
            self.config.font_color,
            self.config.font_thickness
        )
        
        # Add analytics for each joint type
        y_offset = 40
        for joint_type in ["elbow", "knee", "hip", "shoulder"]:
            # Check relevant angles
            assessment = "N/A"
            
            # Look for corresponding angles (e.g., left_elbow, right_elbow for "elbow")
            angles = [angle_results.get(f"left_{joint_type}"), angle_results.get(f"right_{joint_type}")]
            angles = [a for a in angles if a is not None]
            
            if angles:
                avg_angle = sum(angles) / len(angles)
                
                # Get assessment criteria
                criteria = self.angle_assessments.get(joint_type, {})
                normal_range = criteria.get("normal_range", (0, 180))
                messages = criteria.get("messages", {})
                
                # Determine assessment
                if avg_angle < normal_range[0]:
                    assessment = messages.get("low", "Below normal range")
                elif avg_angle > normal_range[1]:
                    assessment = messages.get("high", "Above normal range")
                else:
                    assessment = messages.get("normal", "Within normal range")
            
            # Add assessment text
            cv2.putText(
                frame,
                f"{joint_type.title()}: {assessment}",
                (10, y_offset),
                self.config.font,
                self.config.font_scale,
                self.config.font_color,
                self.config.font_thickness
            )
            
            y_offset += 20
    
    async def create_thumbnail(
        self,
        video_path: str,
        keypoints: List[Keypoint]
    ) -> str:
        """
        Create a thumbnail image with skeleton overlay
        
        Args:
            video_path: Path to the video file
            keypoints: List of keypoints for the thumbnail frame
            
        Returns:
            Path to the thumbnail image
        """
        # Generate thumbnail filename
        thumbnail_filename = f"thumbnail_{uuid.uuid4()}.jpg"
        thumbnail_path = self.config.output_dir / thumbnail_filename
        
        # Open video to extract a frame
        cap = cv2.VideoCapture(video_path)
        
        try:
            # Extract middle frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to extract frame for thumbnail")
                return ""
            
            # Render the frame with skeleton
            rendered_frame = await self._render_frame(frame, keypoints)
            
            # Save the thumbnail
            cv2.imwrite(str(thumbnail_path), rendered_frame)
            
            return str(thumbnail_path)
            
        finally:
            cap.release()