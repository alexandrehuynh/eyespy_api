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
import psutil

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
    
    # Visual effects
    draw_motion_trails: bool = False
    draw_pelvis_origin: bool = False
    draw_spine_overlay: bool = False
    
    # Video output settings
    render_mode: str = "standard"  # "standard", "wireframe", "analysis", "xray"
    
    # Optimization settings
    use_parallel_rendering: bool = True
    max_parallel_frames: int = 8  # Maximum number of frames to render in parallel
    use_frame_caching: bool = True  # Cache rendered elements for performance

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-tune rendering workers based on system
        cpu_count = psutil.cpu_count(logical=False) or 4
        self.max_parallel_frames = min(cpu_count * 2, 12)


class VideoRenderer:
    """Enhanced video renderer with additional visualization options from both projects"""
    
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
        
        # Store motion history for trails
        self.motion_trails = {}
        
        # Cache for rendered elements
        self.cached_angles = {}
        self.cached_keypoints = {}
        self.cached_joints = {}
        
        # Set optimal number of parallel renderers
        self.max_render_workers = self.config.max_parallel_frames
        logger.info(f"VideoRenderer initialized with {self.max_render_workers} parallel rendering workers")
        
    async def render_video(
        self,
        frames: List[np.ndarray],
        keypoints_per_frame: List[List[Keypoint]],
        metadata: Dict[str, Any],
        output_filename: Optional[str] = None,
        render_mode: Optional[str] = None
    ) -> str:
        """
        Render a video with skeleton overlay and analytics
        
        Args:
            frames: List of video frames
            keypoints_per_frame: List of keypoints for each frame
            metadata: Video metadata
            output_filename: Optional filename for the output video
            render_mode: Rendering mode to use (overrides config)
            
        Returns:
            Path to the rendered video file
        """
        start_time = time.time()
        logger.info(f"Starting video rendering for {len(frames)} frames with mode: {render_mode or self.config.render_mode}")
        
        # Set render mode (use parameter if provided, otherwise use config)
        actual_render_mode = render_mode or self.config.render_mode
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"rendered_{actual_render_mode}_{uuid.uuid4()}.mp4"
        
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
        
        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter at {output_path}")
            raise ValueError(f"Failed to open VideoWriter at {output_path}")
            
        logger.info(f"Created video writer with fps={fps}, size={width}x{height}")
        
        # Render frames in batches for better memory management
        total_frames = len(frames)
        frames_written = 0
        batch_size = self.config.batch_size
        
        try:
            for i in range(0, total_frames, batch_size):
                batch_start = time.time()
                end_idx = min(i + batch_size, total_frames)
                batch_frames = frames[i:end_idx]
                batch_keypoints = keypoints_per_frame[i:end_idx]
                
                logger.info(f"Rendering batch {i//batch_size + 1}/{(total_frames+batch_size-1)//batch_size}")
                
                # Render batch with the appropriate mode
                if actual_render_mode == "wireframe":
                    rendered_frames = await self._render_wireframe_batch(batch_frames, batch_keypoints)
                elif actual_render_mode == "analysis":
                    rendered_frames = await self._render_analysis_batch(batch_frames, batch_keypoints)
                elif actual_render_mode == "xray":
                    rendered_frames = await self._render_xray_batch(batch_frames, batch_keypoints)
                else:
                    # Default standard rendering
                    rendered_frames = await self._render_batch(batch_frames, batch_keypoints)
                
                # Write rendered frames
                for frame in rendered_frames:
                    await run_in_executor(writer.write, frame)
                    frames_written += 1
                
                batch_time = time.time() - batch_start
                frames_per_second = len(rendered_frames) / batch_time if batch_time > 0 else 0
                logger.info(f"Rendered {len(rendered_frames)} frames in {batch_time:.2f}s ({frames_per_second:.1f} FPS)")
                
                # Free memory explicitly
                del rendered_frames
                del batch_frames
                del batch_keypoints
                
                # Small delay to prevent event loop congestion
                await asyncio.sleep(0.001)
                
            # Release writer
            await run_in_executor(writer.release)
            
            # Log rendering stats
            total_time = time.time() - start_time
            logger.info(f"Video rendering completed in {total_time:.2f}s")
            logger.info(f"Output: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error rendering video: {str(e)}")
            
            # Ensure writer is released
            if writer.isOpened():
                await run_in_executor(writer.release)
                
            raise

    async def _render_batch(
        self,
        batch_frames: List[np.ndarray],
        batch_keypoints: List[List[Keypoint]]
    ) -> List[np.ndarray]:
        """
        Render a batch of frames with their keypoints in parallel
        """
        render_tasks = []
        
        # Create a render task for each frame
        for i, (frame, keypoints) in enumerate(zip(batch_frames, batch_keypoints)):
            task = self._render_frame(frame, keypoints)
            render_tasks.append(task)
            
        # Process frames in parallel with concurrency limit
        if self.config.use_parallel_rendering:
            return await gather_with_concurrency(
                self.max_render_workers,  # Limit concurrent rendering
                *render_tasks
            )
        else:
            # Sequential rendering (fallback)
            rendered_frames = []
            for task in render_tasks:
                rendered_frame = await task
                rendered_frames.append(rendered_frame)
            return rendered_frames

    async def _render_wireframe_batch(
        self,
        frames: List[np.ndarray],
        keypoints_per_frame: List[List[Keypoint]]
    ) -> List[np.ndarray]:
        """
        Render a batch of frames with wireframe visualization in parallel
        """
        render_tasks = []
        
        # Create a render task for each frame
        for i, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
            task = self._render_wireframe_frame(frame, keypoints)
            render_tasks.append(task)
            
        # Process frames in parallel with concurrency limit
        if self.config.use_parallel_rendering:
            return await gather_with_concurrency(
                self.max_render_workers,
                *render_tasks
            )
        else:
            # Sequential rendering (fallback)
            rendered_frames = []
            for task in render_tasks:
                rendered_frame = await task
                rendered_frames.append(rendered_frame)
            return rendered_frames

    async def _render_analysis_batch(
        self,
        frames: List[np.ndarray],
        keypoints_per_frame: List[List[Keypoint]]
    ) -> List[np.ndarray]:
        """
        Render a batch of frames with analysis visualization in parallel
        """
        render_tasks = []
        
        # Create a render task for each frame
        for i, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
            task = self._render_analysis_frame(frame, keypoints)
            render_tasks.append(task)
            
        # Process frames in parallel with concurrency limit
        if self.config.use_parallel_rendering:
            return await gather_with_concurrency(
                self.max_render_workers,
                *render_tasks
            )
        else:
            # Sequential rendering (fallback)
            rendered_frames = []
            for task in render_tasks:
                rendered_frame = await task
                rendered_frames.append(rendered_frame)
            return rendered_frames

    async def _render_xray_batch(
        self,
        frames: List[np.ndarray],
        keypoints_per_frame: List[List[Keypoint]]
    ) -> List[np.ndarray]:
        """
        Render a batch of frames with xray visualization in parallel
        """
        render_tasks = []
        
        # Create a render task for each frame
        for i, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
            task = self._render_xray_frame(frame, keypoints)
            render_tasks.append(task)
            
        # Process frames in parallel with concurrency limit
        if self.config.use_parallel_rendering:
            return await gather_with_concurrency(
                self.max_render_workers,
                *render_tasks
            )
        else:
            # Sequential rendering (fallback)
            rendered_frames = []
            for task in render_tasks:
                rendered_frame = await task
                rendered_frames.append(rendered_frame)
            return rendered_frames
    
    async def _render_frame(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """
        Render a single frame with skeleton overlay
        
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
        
        # Make a copy of the frame to avoid modifying the original
        rendered_frame = frame.copy()
        
        # Convert keypoints to dictionary for easier access
        keypoint_dict = {kp.name: kp for kp in keypoints}
        
        # Draw skeleton
        self._draw_skeleton(rendered_frame, keypoint_dict)
        
        # Draw joints
        self._draw_joints(rendered_frame, keypoint_dict)
        
        # Draw angles
        if self.config.show_angles:
            angle_results = self._draw_angles(rendered_frame, keypoint_dict)
            
            # Add analytics based on angles
            self._add_analytics_text(rendered_frame, angle_results)
        
        # Optional: Draw motion trails
        if self.config.draw_motion_trails:
            rendered_frame = self._draw_motion_trails(rendered_frame, keypoint_dict)
            
        # Optional: Draw pelvis origin
        if self.config.draw_pelvis_origin:
            rendered_frame = self._draw_pelvis_origin(rendered_frame, keypoint_dict)
            
        # Optional: Draw spine overlay
        if self.config.draw_spine_overlay:
            rendered_frame = self._draw_spine_overlay(rendered_frame, keypoint_dict)
        
        return rendered_frame
    
    async def _render_wireframe_frame(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """Render a frame with wireframe visualization"""
        return await run_in_executor(
            self._render_wireframe_frame_sync,
            frame,
            keypoints
        )
    
    def _render_wireframe_frame_sync(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """Wireframe visualization (from backend_fresh)"""
        # Create a black canvas
        height, width = frame.shape[:2]
        wireframe = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Skip if no keypoints
        if not keypoints:
            return wireframe
        
        # Convert keypoints to dictionary
        keypoint_dict = {kp.name: kp for kp in keypoints}
        
        # Draw wireframe skeleton with bright colors
        for p1_name, p2_name in self.skeleton_connections:
            if p1_name in keypoint_dict and p2_name in keypoint_dict:
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.5 or p2.confidence < 0.5:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Draw bright lines
                cv2.line(wireframe, p1_px, p2_px, (0, 255, 255), 3)
        
        # Draw joints - use a list of items() to avoid modifying during iteration
        for name, kp in list(keypoint_dict.items()):
            if kp.confidence < 0.5:
                continue
                
            x_px = int(kp.x * width)
            y_px = int(kp.y * height)
            
            # Draw larger circles for joints
            cv2.circle(wireframe, (x_px, y_px), 6, (255, 255, 0), -1)
            
        # Add angles for key joints
        angle_results = self._calculate_angles(keypoint_dict)
        for joint_name, angle in list(angle_results.items()):
            if "elbow" in joint_name or "knee" in joint_name:
                landmark_name = joint_name.replace("left_", "LEFT_").replace("right_", "RIGHT_").replace("elbow", "ELBOW").replace("knee", "KNEE")
                if landmark_name in keypoint_dict:
                    kp = keypoint_dict[landmark_name]
                    x_px = int(kp.x * width) + 10
                    y_px = int(kp.y * height) + 10
                    cv2.putText(wireframe, f"{angle:.1f}", (x_px, y_px), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return wireframe
    
    async def _render_analysis_frame(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """Render a frame with detailed analysis visualization"""
        return await run_in_executor(
            self._render_analysis_frame_sync,
            frame,
            keypoints
        )
    
    def _render_analysis_frame_sync(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """Analysis visualization (combination of standard and additional metrics)"""
        # First draw standard skeleton - make a copy to avoid modifying original frame
        frame_copy = frame.copy()
        rendered = self._render_frame_sync(frame_copy, keypoints)
        
        # Skip if no keypoints
        if not keypoints:
            return rendered
        
        # Convert keypoints to dictionary
        keypoint_dict = {kp.name: kp for kp in keypoints}
        
        # Add large analytics panel
        height, width = rendered.shape[:2]
        overlay = rendered.copy()
        cv2.rectangle(overlay, (width-350, 0), (width, 400), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, rendered, 1 - alpha, 0, rendered)
        
        # Calculate joint angles
        angles = self._calculate_angles(keypoint_dict)
        
        # Add detailed analytics text
        cv2.putText(rendered, "MOVEMENT ANALYSIS", (width-340, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Joint angles - use a list of items() to avoid modifying during iteration
        y_offset = 70
        for joint, angle in list(angles.items()):
            color = (0, 255, 0) if self._is_angle_in_range(joint, angle) else (0, 0, 255)
            cv2.putText(rendered, f"{joint}: {angle:.1f}", (width-340, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Form assessment
        y_offset += 20
        cv2.putText(rendered, "FORM ASSESSMENT", (width-340, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Add form assessment based on angles
        form_issues = self._assess_form(angles)
        if form_issues:
            for issue in form_issues[:3]:  # Show up to 3 issues
                cv2.putText(rendered, f"- {issue}", (width-340, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 25
        else:
            cv2.putText(rendered, "- Good form", (width-340, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return rendered
    
    async def _render_xray_frame(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """Render a frame with X-ray style visualization"""
        return await run_in_executor(
            self._render_xray_frame_sync,
            frame,
            keypoints
        )
    
    def _render_xray_frame_sync(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint]
    ) -> np.ndarray:
        """X-ray style visualization (dark background with glowing skeleton)"""
        # Create a dark version of the original frame
        height, width = frame.shape[:2]
        xray = cv2.addWeighted(frame, 0.2, np.zeros_like(frame), 0.8, 0)
        
        # Skip if no keypoints
        if not keypoints:
            return xray
        
        # Convert keypoints to dictionary
        keypoint_dict = {kp.name: kp for kp in keypoints}
        
        # Create a black canvas for the skeleton
        skeleton_canvas = np.zeros_like(frame)
        
        # Draw thick bright lines for skeleton
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
                
                # Draw thick bright lines
                cv2.line(skeleton_canvas, p1_px, p2_px, (0, 255, 255), 4)
        
        # Draw joints - use a list of items() to avoid modifying during iteration
        for name, kp in list(keypoint_dict.items()):
            if kp.confidence < 0.4:
                continue
                
            x_px = int(kp.x * width)
            y_px = int(kp.y * height)
            
            # Draw larger circles for joints
            cv2.circle(skeleton_canvas, (x_px, y_px), 8, (0, 255, 255), -1)
        
        # Apply blur for glow effect
        skeleton_glow = cv2.GaussianBlur(skeleton_canvas, (15, 15), 0)
        
        # Overlay the glowing skeleton on the dark frame
        result = cv2.addWeighted(xray, 1.0, skeleton_glow, 1.0, 0)
        
        return result
    
    def _calculate_angles(self, keypoint_dict: Dict[str, Keypoint]) -> Dict[str, float]:
        """Calculate joint angles from keypoints"""
        angles = {}
        
        for angle_name, (p1_name, p2_name, p3_name) in self.angle_definitions.items():
            if all(p_name in keypoint_dict for p_name in [p1_name, p2_name, p3_name]):
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                p3 = keypoint_dict[p3_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.5 or p2.confidence < 0.5 or p3.confidence < 0.5:
                    continue
                
                # Calculate angle
                angle = self._calculate_angle(p1, p2, p3)
                angles[angle_name] = angle
                
        return angles
    
    def _calculate_angle(self, p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
        """Calculate angle between three points"""
        # Convert to numpy arrays
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        
        # Calculate angle
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _is_angle_in_range(self, joint_name: str, angle: float) -> bool:
        """Check if angle is in normal range for the joint"""
        ranges = {
            'left_elbow': (10, 160),
            'right_elbow': (10, 160),
            'left_knee': (10, 170),
            'right_knee': (10, 170),
            'left_hip': (80, 160),
            'right_hip': (80, 160),
            'left_shoulder': (50, 170),
            'right_shoulder': (50, 170)
        }
        
        if joint_name in ranges:
            min_val, max_val = ranges[joint_name]
            return min_val <= angle <= max_val
        
        return True
    
    def _assess_form(self, angles: Dict[str, float]) -> List[str]:
        """Assess form based on joint angles"""
        issues = []
        
        # Check knee angles
        if 'left_knee' in angles and angles['left_knee'] < 10:
            issues.append("Left knee hyperextended")
        if 'right_knee' in angles and angles['right_knee'] < 10:
            issues.append("Right knee hyperextended")
            
        # Check knee alignment
        if 'left_knee' in angles and 'right_knee' in angles:
            if abs(angles['left_knee'] - angles['right_knee']) > 15:
                issues.append("Uneven knee bend")
                
        # Check hip alignment
        if 'left_hip' in angles and 'right_hip' in angles:
            if abs(angles['left_hip'] - angles['right_hip']) > 15:
                issues.append("Uneven hip position")
                
        # Check shoulder alignment
        if 'left_shoulder' in angles and 'right_shoulder' in angles:
            if abs(angles['left_shoulder'] - angles['right_shoulder']) > 15:
                issues.append("Uneven shoulder position")
                
        return issues
    
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
        
        # Create a copy of the skeleton connections list to avoid modification issues
        connections_to_draw = list(self.skeleton_connections)
        
        for p1_name, p2_name in connections_to_draw:
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
        
        # Create a copy of the dictionary items to avoid modification issues
        joints_to_draw = list(keypoint_dict.items())
        
        for name, keypoint in joints_to_draw:
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
        
        # Create a copy of the angle definitions to avoid modification issues
        angle_defs_to_process = list(self.angle_definitions.items())
        
        for angle_name, (p1_name, p2_name, p3_name) in angle_defs_to_process:
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
                    f"{angle:.1f}",
                    (p2_px[0] + 10, p2_px[1] + 10),
                    self.config.font,
                    self.config.font_scale,
                    self.config.angle_color,
                    self.config.font_thickness
                )
        
        return angle_results
    
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
        
        # Create a copy of angle_results to avoid modification during iteration
        angle_results_copy = dict(angle_results)
        
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
        
        # Add analytics for each joint type - use predefined list to avoid modification issues
        joint_types = ["elbow", "knee", "hip", "shoulder"]
        
        y_offset = 40
        for joint_type in joint_types:
            # Check relevant angles
            assessment = "N/A"
            
            # Look for corresponding angles (e.g., left_elbow, right_elbow for "elbow")
            angles = [angle_results_copy.get(f"left_{joint_type}"), angle_results_copy.get(f"right_{joint_type}")]
            angles = [a for a in angles if a is not None]
            
            if angles:
                avg_angle = sum(angles) / len(angles)
                
                # Get assessment message
                if joint_type == "elbow":
                    if avg_angle < 10:
                        assessment = "Arm too straight"
                    elif avg_angle > 160:
                        assessment = "Arm too bent"
                    else:
                        assessment = "Good elbow position"
                elif joint_type == "knee":
                    if avg_angle < 10:
                        assessment = "Knee locked"
                    elif avg_angle > 170:
                        assessment = "Knee too bent"
                    else:
                        assessment = "Good knee position"
                elif joint_type == "hip":
                    if avg_angle < 80:
                        assessment = "Hip too bent"
                    elif avg_angle > 160:
                        assessment = "Hip too straight"
                    else:
                        assessment = "Good hip position"
                elif joint_type == "shoulder":
                    if avg_angle < 50:
                        assessment = "Shoulder too forward"
                    elif avg_angle > 170:
                        assessment = "Shoulder too back"
                    else:
                        assessment = "Good shoulder position"
            
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
    
    def _draw_motion_trails(
        self,
        frame: np.ndarray,
        keypoint_dict: Dict[str, Keypoint]
    ) -> np.ndarray:
        """
        Draw motion trails for tracked keypoints
        
        Args:
            frame: Video frame
            keypoint_dict: Dictionary of keypoints
            
        Returns:
            Frame with motion trails
        """
        height, width = frame.shape[:2]
        
        # Key points to track for trails
        trail_points = [
            "LEFT_WRIST", "RIGHT_WRIST",  # Hands
            "LEFT_ANKLE", "RIGHT_ANKLE",  # Feet
            "NOSE"                         # Head
        ]
        
        # Update motion trails
        for point_name in trail_points:
            if point_name in keypoint_dict and keypoint_dict[point_name].confidence > 0.5:
                kp = keypoint_dict[point_name]
                pos = (int(kp.x * width), int(kp.y * height))
                
                if point_name not in self.motion_trails:
                    self.motion_trails[point_name] = []
                
                self.motion_trails[point_name].append(pos)
                
                # Limit trail length
                if len(self.motion_trails[point_name]) > self.config.batch_size:
                    self.motion_trails[point_name].pop(0)
        
        # Create a copy of the motion trails dictionary before iterating
        trails_to_draw = dict(self.motion_trails)
        
        # Draw trails
        for point_name, trail in trails_to_draw.items():
            if len(trail) >= 2:
                for i in range(1, len(trail)):
                    # Fade color based on position in trail
                    alpha = i / len(trail)
                    
                    if "WRIST" in point_name:
                        color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # Yellow to cyan
                    elif "ANKLE" in point_name:
                        color = (int(255 * (1 - alpha)), int(255 * alpha), 0)  # Green to yellow
                    else:
                        color = (int(255 * alpha), 0, int(255 * (1 - alpha)))  # Purple to blue
                    
                    cv2.line(frame, trail[i-1], trail[i], color, 2)
        
        return frame
    
    def _draw_pelvis_origin(
        self,
        frame: np.ndarray,
        keypoint_dict: Dict[str, Keypoint]
    ) -> np.ndarray:
        """
        Draw pelvis origin and coordinate axes
        
        Args:
            frame: Video frame
            keypoint_dict: Dictionary of keypoints
            
        Returns:
            Frame with pelvis origin visualization
        """
        height, width = frame.shape[:2]
        
        # Check if required keypoints are available
        if not all(k in keypoint_dict for k in ["LEFT_HIP", "RIGHT_HIP"]):
            return frame
            
        left_hip = keypoint_dict["LEFT_HIP"]
        right_hip = keypoint_dict["RIGHT_HIP"]
        
        # Skip if confidence is too low
        if left_hip.confidence < 0.5 or right_hip.confidence < 0.5:
            return frame
            
        # Calculate pelvis origin
        left_hip_px = (int(left_hip.x * width), int(left_hip.y * height))
        right_hip_px = (int(right_hip.x * width), int(right_hip.y * height))
        
        pelvis_origin = (
            (left_hip_px[0] + right_hip_px[0]) // 2,
            (left_hip_px[1] + right_hip_px[1]) // 2
        )
        
        # Create overlay for axes
        overlay = frame.copy()
        
        # Draw 3D axes
        axis_length = int(min(width, height) * 0.2)
        
        # X-axis (red)
        cv2.line(
            overlay,
            pelvis_origin,
            (pelvis_origin[0] + axis_length, pelvis_origin[1]),
            (0, 0, 255),
            2
        )
        cv2.putText(
            overlay,
            "X",
            (pelvis_origin[0] + axis_length + 10, pelvis_origin[1]),
            self.config.font,
            self.config.font_scale,
            (0, 0, 255),
            self.config.font_thickness
        )
        
        # Y-axis (green)
        cv2.line(
            overlay,
            pelvis_origin,
            (pelvis_origin[0], pelvis_origin[1] - axis_length),
            (0, 255, 0),
            2
        )
        cv2.putText(
            overlay,
            "Y",
            (pelvis_origin[0], pelvis_origin[1] - axis_length - 10),
            self.config.font,
            self.config.font_scale,
            (0, 255, 0),
            self.config.font_thickness
        )
        
        # Z-axis (blue)
        cv2.line(
            overlay,
            pelvis_origin,
            (pelvis_origin[0], pelvis_origin[1] + axis_length),
            (255, 0, 0),
            2
        )
        cv2.putText(
            overlay,
            "Z",
            (pelvis_origin[0], pelvis_origin[1] + axis_length + 20),
            self.config.font,
            self.config.font_scale,
            (255, 0, 0),
            self.config.font_thickness
        )
        
        # Apply overlay with transparency
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def _draw_spine_overlay(
        self,
        frame: np.ndarray,
        keypoint_dict: Dict[str, Keypoint]
    ) -> np.ndarray:
        """
        Draw spine overlay connecting shoulders to hips
        
        Args:
            frame: Video frame
            keypoint_dict: Dictionary of keypoints
            
        Returns:
            Frame with spine overlay
        """
        height, width = frame.shape[:2]
        
        # Check if required keypoints are available
        if not all(k in keypoint_dict for k in [
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"
        ]):
            return frame
            
        left_shoulder = keypoint_dict["LEFT_SHOULDER"]
        right_shoulder = keypoint_dict["RIGHT_SHOULDER"]
        left_hip = keypoint_dict["LEFT_HIP"]
        right_hip = keypoint_dict["RIGHT_HIP"]
        
        # Skip if confidence is too low
        if (left_shoulder.confidence < 0.5 or right_shoulder.confidence < 0.5 or
            left_hip.confidence < 0.5 or right_hip.confidence < 0.5):
            return frame
            
        # Calculate midpoints
        left_shoulder_px = (int(left_shoulder.x * width), int(left_shoulder.y * height))
        right_shoulder_px = (int(right_shoulder.x * width), int(right_shoulder.y * height))
        left_hip_px = (int(left_hip.x * width), int(left_hip.y * height))
        right_hip_px = (int(right_hip.x * width), int(right_hip.y * height))
        
        shoulder_midpoint = (
            (left_shoulder_px[0] + right_shoulder_px[0]) // 2,
            (left_shoulder_px[1] + right_shoulder_px[1]) // 2
        )
        
        hip_midpoint = (
            (left_hip_px[0] + right_hip_px[0]) // 2,
            (left_hip_px[1] + right_hip_px[1]) // 2
        )
        
        # Generate spine points
        spine_points = []
        num_points = 5
        
        for i in range(num_points + 1):
            t = i / num_points
            x = int((1 - t) * shoulder_midpoint[0] + t * hip_midpoint[0])
            y = int((1 - t) * shoulder_midpoint[1] + t * hip_midpoint[1])
            spine_points.append((x, y))
        
        # Draw spine segments
        for i in range(len(spine_points) - 1):
            cv2.line(
                frame,
                spine_points[i],
                spine_points[i + 1],
                (0, 255, 0),
                2
            )
            cv2.circle(
                frame,
                spine_points[i],
                3,
                (255, 255, 255),
                -1
            )
            
        cv2.circle(
            frame,
            spine_points[-1],
            3,
            (255, 255, 255),
            -1
        )
        
        return frame