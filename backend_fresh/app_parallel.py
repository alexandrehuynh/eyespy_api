"""
Enhanced video processing application with improved architecture and performance.
"""
import asyncio
import cv2
import logging
import mediapipe as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed as futures_completed
from dataclasses import dataclass
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from queue import Queue
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration parameters for video processing."""
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    smoothing_window: int = 5
    motion_trail_length: int = 10
    axis_scale: float = 0.2
    axis_opacity: float = 0.5
    skeleton_save_interval: int = 30
    buffer_size: int = 30
    velocity_update_interval: int = 10
    max_workers_io: int = 4
    max_workers_cpu: int = 6
    queue_size: int = 100

class QueueManager:
    """Manages frame processing queues and buffers."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.input_queue = asyncio.Queue(maxsize=config.queue_size)
        self.processing_queue = asyncio.Queue(maxsize=config.queue_size)
        self.output_queue = asyncio.Queue(maxsize=config.queue_size)
        self.frame_buffer = []
        self.is_processing = True

    async def add_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """Add a frame to the input queue."""
        await self.input_queue.put((frame, frame_number))
        
        # Maintain frame buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.config.buffer_size:
            self.frame_buffer.pop(0)

    async def get_frame_for_processing(self) -> Tuple[np.ndarray, int]:
        """Get a frame from the input queue for processing."""
        return await self.input_queue.get()

    async def put_processed_frame(self, processed_frame: dict) -> None:
        """Add a processed frame to the output queue."""
        await self.output_queue.put(processed_frame)

    async def get_frame_for_rendering(self) -> dict:
        """Get a processed frame for rendering."""
        return await self.output_queue.get()

    def stop(self) -> None:
        """Stop queue processing."""
        self.is_processing = False

class FrameProcessor:
    """Handles raw frame processing using MediaPipe."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        self.executor = ProcessPoolExecutor(max_workers=config.max_workers_cpu)

    async def process_frame(self, frame: np.ndarray, frame_number: int) -> dict:
        """Process a single frame using MediaPipe pose detection."""
        try:
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame in separate process
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_frame_cpu,
                frame_rgb,
                frame_number
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {str(e)}")
            return None

    def _process_frame_cpu(self, frame_rgb: np.ndarray, frame_number: int) -> dict:
        """CPU-intensive frame processing (runs in separate process)."""
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Extract and process landmarks
        landmarks = self._extract_landmarks(results.pose_landmarks)
        
        # Calculate additional features
        angles = self._calculate_angles(landmarks)
        velocities = self._calculate_velocities(landmarks, frame_number)
        
        return {
            'frame_number': frame_number,
            'landmarks': landmarks,
            'angles': angles,
            'velocities': velocities,
            'timestamp': datetime.now().timestamp()
        }

    def _extract_landmarks(self, pose_landmarks) -> List[List[float]]:
        """Extract landmark coordinates from MediaPipe results."""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        return landmarks

    def _calculate_angles(self, landmarks: List[List[float]]) -> Dict[str, Tuple[int, float]]:
        """
        Calculate joint angles from landmarks.
        Returns a dictionary mapping joint names to tuples of (landmark_index, angle).
        Thread-safe implementation that processes angles in parallel.
        """
        angles = {}
        
        def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
            """Calculate angle between three points using vectorization."""
            if not all(p is not None for p in [p1, p2, p3]):
                return None
                
            vector1 = p1 - p2
            vector2 = p3 - p2
            
            # Vectorized operations for better performance
            cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return angle

        # Convert landmarks to numpy array for vectorized operations
        landmark_array = np.array([lm[:3] for lm in landmarks])
        
        # Dictionary defining joint angle calculations
        joint_configs = {
            'left_elbow': (11, 13, 15),    # Left shoulder, elbow, wrist
            'right_elbow': (12, 14, 16),   # Right shoulder, elbow, wrist
            'left_shoulder': (13, 11, 23),  # Left elbow, shoulder, hip
            'right_shoulder': (14, 12, 24), # Right elbow, shoulder, hip
            'left_hip': (11, 23, 25),      # Left shoulder, hip, knee
            'right_hip': (12, 24, 26),     # Right shoulder, hip, knee
            'left_knee': (23, 25, 27),     # Left hip, knee, ankle
            'right_knee': (24, 26, 28),    # Right hip, knee, ankle
        }

        # Process angles in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as executor:
            # Create tasks for each joint angle calculation
            future_to_joint = {
                executor.submit(
                    calculate_angle,
                    landmark_array[p1],
                    landmark_array[p2],
                    landmark_array[p3]
                ): (joint_name, p2)  # p2 is the vertex point
                for joint_name, (p1, p2, p3) in joint_configs.items()
            }

            # Collect results as they complete
            for future in futures_completed(future_to_joint):
                joint_name, landmark_idx = future_to_joint[future]
                try:
                    angle = future.result()
                    if angle is not None:
                        angles[joint_name] = (landmark_idx, angle)
                except Exception as e:
                    logger.error(f"Error calculating angle for {joint_name}: {str(e)}")

        return angles

    def _calculate_velocities(self, landmarks: List[List[float]], frame_number: int) -> Dict[int, float]:
        """
        Calculate landmark velocities (only updated every N frames).
        Uses numpy for vectorized calculations and parallel processing for efficiency.
        """
        # Only calculate velocities periodically to improve performance
        if frame_number % self.config.velocity_update_interval != 0:
            return {}

        velocities = {}
        
        try:
            # Convert current landmarks to numpy array
            current_landmarks = np.array([lm[:3] for lm in landmarks])
            
            if not hasattr(self, '_previous_landmarks'):
                self._previous_landmarks = current_landmarks
                return {}

            # Calculate velocities using vectorized operations
            with ProcessPoolExecutor(max_workers=4) as executor:
                # Split landmarks into chunks for parallel processing
                chunk_size = len(current_landmarks) // 4
                chunks = [
                    (
                        current_landmarks[i:i + chunk_size],
                        self._previous_landmarks[i:i + chunk_size],
                        range(i, i + chunk_size)
                    )
                    for i in range(0, len(current_landmarks), chunk_size)
                ]

                # Process chunks in parallel
                future_to_chunk = {
                    executor.submit(
                        self._calculate_velocity_chunk,
                        current_chunk,
                        prev_chunk,
                        indices
                    ): indices
                    for current_chunk, prev_chunk, indices in chunks
                }

                # Collect results
                for future in futures_completed(future_to_chunk):
                    try:
                        chunk_velocities = future.result()
                        velocities.update(chunk_velocities)
                    except Exception as e:
                        logger.error(f"Error calculating velocities for chunk: {str(e)}")

            # Update previous landmarks
            self._previous_landmarks = current_landmarks
            
            return velocities

        except Exception as e:
            logger.error(f"Error calculating velocities: {str(e)}")
            return {}

    def _calculate_velocity_chunk(
        self,
        current_chunk: np.ndarray,
        prev_chunk: np.ndarray,
        indices: range
    ) -> Dict[int, float]:
        """
        Calculate velocities for a chunk of landmarks.
        Helper method for parallel velocity calculations.
        """
        chunk_velocities = {}
        
        # Vectorized velocity calculation
        velocities = np.linalg.norm(current_chunk - prev_chunk, axis=1)
        
        # Store results
        for idx, velocity in zip(indices, velocities):
            if velocity > 0.001:  # Filter out very small movements
                chunk_velocities[idx] = velocity
                
        return chunk_velocities

    async def cleanup(self):
        """Clean up resources."""
        self.pose.close()
        self.executor.shutdown()

class VisualEffectsRenderer:
    """Handles rendering of visual effects and overlays."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.previous_frame = None

    async def render_frame(self, frame: np.ndarray, processing_result: dict) -> np.ndarray:
        """Render visual effects on processed frame."""
        if processing_result is None:
            return frame

        rendered_frame = frame.copy()
        
        # Apply visual effects
        rendered_frame = await self._draw_skeleton(rendered_frame, processing_result)
        rendered_frame = await self._draw_motion_trails(rendered_frame, processing_result)
        rendered_frame = await self._draw_velocity_indicators(rendered_frame, processing_result)
        rendered_frame = await self._draw_joint_angles(rendered_frame, processing_result)
        
        self.previous_frame = rendered_frame
        return rendered_frame

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.motion_trails = defaultdict(lambda: deque(maxlen=config.motion_trail_length))
        self.previous_landmarks = None
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Define custom pose connections (excluding face landmarks)
        self.custom_pose_connections = frozenset([
            (start, end) for start, end in mp.solutions.pose.POSE_CONNECTIONS
            if 11 <= start < 33 and 11 <= end < 33  # Only include body landmarks (11-32)
        ])
        
        # Color configurations
        self.colors = {
            'skeleton': (255, 255, 255),    # White
            'joint': (128, 128, 128),       # Gray
            'trail': (0, 255, 255),         # Cyan
            'velocity': (0, 255, 0),        # Green
            'angle_text': (0, 255, 0),      # Green
            'angle_warning': (0, 0, 255)    # Red
        }

    async def _draw_skeleton(self, frame: np.ndarray, processing_result: dict) -> np.ndarray:
        """
        Draw skeleton overlay on frame with custom styling.
        
        Args:
            frame: Input frame
            processing_result: Dict containing landmarks and other processing data
            
        Returns:
            Frame with skeleton overlay
        """
        if not processing_result or 'landmarks' not in processing_result:
            return frame
            
        landmarks = processing_result['landmarks']
        
        try:
            # Create a copy of the frame for drawing
            overlay = frame.copy()
            
            # Draw connections using custom pose connections
            for start_idx, end_idx in self.custom_pose_connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    landmarks[start_idx] is not None and landmarks[end_idx] is not None):
                    
                    start_point = tuple(map(int, landmarks[start_idx][:2]))
                    end_point = tuple(map(int, landmarks[end_idx][:2]))
                    
                    # Draw connection line
                    cv2.line(overlay, start_point, end_point, self.colors['skeleton'], 2, cv2.LINE_AA)
            
            # Draw landmarks
            for i, landmark in enumerate(landmarks):
                if 11 <= i < 33 and landmark is not None:  # Only body landmarks
                    point = tuple(map(int, landmark[:2]))
                    
                    # Draw outer circle
                    cv2.circle(overlay, point, 4, self.colors['skeleton'], -1, cv2.LINE_AA)
                    # Draw inner circle
                    cv2.circle(overlay, point, 2, self.colors['joint'], -1, cv2.LINE_AA)
            
            # Blend overlay with original frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing skeleton: {str(e)}")
            return frame

    async def _draw_motion_trails(self, frame: np.ndarray, processing_result: dict) -> np.ndarray:
        """
        Draw motion trails for tracked landmarks with fade effect.
        
        Args:
            frame: Input frame
            processing_result: Dict containing landmarks and other processing data
            
        Returns:
            Frame with motion trails
        """
        if not processing_result or 'landmarks' not in processing_result:
            return frame
            
        landmarks = processing_result['landmarks']
        
        try:
            # Update motion trails
            for start_idx, end_idx in self.custom_pose_connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    landmarks[start_idx] is not None and landmarks[end_idx] is not None):
                    
                    start_point = tuple(map(int, landmarks[start_idx][:2]))
                    end_point = tuple(map(int, landmarks[end_idx][:2]))
                    
                    # Store points for trail
                    self.motion_trails[start_idx].append(start_point)
                    self.motion_trails[end_idx].append(end_point)
                    
                    # Draw trails with fade effect
                    for trail_id in [start_idx, end_idx]:
                        points = list(self.motion_trails[trail_id])
                        if len(points) > 1:
                            for i in range(1, len(points)):
                                # Calculate alpha based on position in trail
                                alpha = i / len(points)
                                color = tuple(int(c * alpha) for c in self.colors['trail'])
                                
                                cv2.line(frame,
                                        points[i-1],
                                        points[i],
                                        color,
                                        max(1, int(2 * alpha)),  # Line thickness fades too
                                        cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing motion trails: {str(e)}")
            return frame

    async def _draw_velocity_indicators(self, frame: np.ndarray, processing_result: dict) -> np.ndarray:
        """
        Draw velocity indicators for landmarks (updated periodically).
        
        Args:
            frame: Input frame
            processing_result: Dict containing landmarks and velocities
            
        Returns:
            Frame with velocity indicators
        """
        if (not processing_result or
            'landmarks' not in processing_result or
            'velocities' not in processing_result or
            not processing_result['velocities']):
            return frame
            
        try:
            landmarks = processing_result['landmarks']
            velocities = processing_result['velocities']
            
            # Draw velocity indicators for landmarks in custom pose connections
            for start_idx, end_idx in self.custom_pose_connections:
                for idx in [start_idx, end_idx]:
                    if idx in velocities and landmarks[idx] is not None:
                        velocity = velocities[idx]
                        
                        # Only draw for significant movement
                        if velocity > 5:
                            position = tuple(map(int, landmarks[idx][:2]))
                            
                            # Scale circle size with velocity
                            radius = int(min(velocity * 2, 30))  # Cap maximum size
                            
                            # Draw velocity indicator
                            cv2.circle(frame,
                                     position,
                                     radius,
                                     self.colors['velocity'],
                                     max(1, int(velocity / 10)),  # Line thickness scales with velocity
                                     cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing velocity indicators: {str(e)}")
            return frame

    async def _draw_joint_angles(self, frame: np.ndarray, processing_result: dict) -> np.ndarray:
        """
        Draw joint angle measurements with color coding.
        
        Args:
            frame: Input frame
            processing_result: Dict containing landmarks and angles
            
        Returns:
            Frame with joint angle measurements
        """
        if not processing_result or 'angles' not in processing_result:
            return frame
            
        try:
            landmarks = processing_result['landmarks']
            angles = processing_result['angles']
            
            # Draw angle measurements for each joint
            for joint_name, (landmark_idx, angle) in angles.items():
                if angle is not None and landmarks[landmark_idx] is not None:
                    position = landmarks[landmark_idx]
                    text_pos = (
                        int(position[0] + 20),
                        int(position[1] - 20)
                    )
                    
                    # Determine text color based on angle
                    color = (self.colors['angle_text'] if angle < 90 
                            else self.colors['angle_warning'])
                    
                    # Format angle text
                    text = f"{joint_name}: {int(angle)}°"
                    
                    # Draw text with background for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(
                        text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        2
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        frame,
                        (text_pos[0] - 2, text_pos[1] - text_height - 2),
                        (text_pos[0] + text_width + 2, text_pos[1] + 2),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame,
                        text,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA
                    )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing joint angles: {str(e)}")
            return frame

class VideoWriter:
    """Handles video writing operations."""
    
    def __init__(self, output_path: Path, frame_shape: Tuple[int, int], fps: int):
        self.output_path = output_path
        self.frame_shape = frame_shape
        self.fps = fps
        self.writer = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self) -> None:
        """Initialize video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.frame_shape[1], self.frame_shape[0])
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to initialize video writer for {self.output_path}")

    async def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to video file asynchronously."""
        if self.writer is None:
            await self.initialize()
            
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.writer.write,
            frame
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.writer is not None:
            self.writer.release()
        self.executor.shutdown()

class VideoProcessor:
    """Main class coordinating the video processing pipeline."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.queue_manager = QueueManager(config)
        self.frame_processor = FrameProcessor(config)
        self.visual_effects = VisualEffectsRenderer(config)
        self.writers = {}
        self.metadata = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'frames': []
        }

    async def process_video(self, input_path: Path, output_paths: Dict[str, Path]) -> dict:
        """Process video through the pipeline."""
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open input video")

            # Initialize video writers
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_shape = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            )

            for name, path in output_paths.items():
                self.writers[name] = VideoWriter(path, frame_shape, fps)
                await self.writers[name].initialize()

            # Start processing pipeline
            await self._process_frames(cap)

            return self.metadata

        finally:
            await self._cleanup()

    async def _process_frames(self, cap: cv2.VideoCapture) -> None:
        """Process frames through the pipeline."""
        frame_number = 0
        
        # Create tasks for pipeline stages
        tasks = [
            asyncio.create_task(self._frame_reader(cap)),
            asyncio.create_task(self._frame_processor()),
            asyncio.create_task(self._frame_renderer())
        ]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    async def _frame_reader(self, cap: cv2.VideoCapture) -> None:
        """Read frames from video and add to input queue."""
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            await self.queue_manager.add_frame(frame, frame_number)
            frame_number += 1
            
        self.queue_manager.stop()

    async def _frame_processor(self) -> None:
        """Process frames from input queue."""
        while self.queue_manager.is_processing:
            frame, frame_number = await self.queue_manager.get_frame_for_processing()
            result = await self.frame_processor.process_frame(frame, frame_number)
            await self.queue_manager.put_processed_frame({
                'frame': frame,
                'result': result,
                'frame_number': frame_number
            })

    async def _frame_renderer(self) -> None:
        """Render processed frames and write to output."""
        while self.queue_manager.is_processing:
            frame_data = await self.queue_manager.get_frame_for_rendering()
            
            # Render visual effects
            rendered_frame = await self.visual_effects.render_frame(
                frame_data['frame'],
                frame_data['result']
            )
            
            # Write to output videos
            for writer in self.writers.values():
                await writer.write_frame(rendered_frame)
            
            # Update metadata
            self.metadata['frames'].append(frame_data['result'])

    async def _cleanup(self) -> None:
        """Clean up resources."""
        await self.frame_processor.cleanup()
        for writer in self.writers.values():
            await writer.cleanup()

# FastAPI application setup
app = FastAPI()

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    """Process video endpoint."""
    try:
        # Initialize configuration
        config = ProcessingConfig()
        
        # Setup paths
        input_path, output_paths = setup_paths(file)
        
        # Process video
        processor = VideoProcessor(config)
        metadata = await processor.process_video(input_path, output_paths)
        
        return {
            "status": "success",
            "videos": {
                name: FileResponse(path, media_type="video/mp4", filename=path.name)
                for name, path in output_paths.items()
            },
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def setup_paths(file: UploadFile) -> Tuple[Path, Dict[str, Path]]:
    """Setup input and output paths."""
    # Implementation of path setup
    pass