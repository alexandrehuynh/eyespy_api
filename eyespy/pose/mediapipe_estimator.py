import mediapipe as mp
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from ..models import Keypoint
from ..config import settings
import asyncio
import cv2
from .validation import PoseValidator
from .confidence import AdaptiveConfidenceAssessor
import time
import psutil
import logging
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor, gather_with_concurrency
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameContinuityTracker:
    """Helper class to track continuity between frames for better pose estimation"""
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.frame_history = []
        self.keypoint_history = []
        self.confidence_history = []
    
    def add_frame(self, frame: np.ndarray, keypoints: Optional[List[Keypoint]], confidence: float):
        """Add frame and pose data to history"""
        # Add new data
        self.frame_history.append(frame)
        self.keypoint_history.append(keypoints)
        self.confidence_history.append(confidence)
        
        # Maintain max history size
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
            self.keypoint_history.pop(0)
            self.confidence_history.pop(0)
    
    def get_best_reference_frame(self) -> Tuple[Optional[np.ndarray], Optional[List[Keypoint]], float]:
        """Get the best reference frame based on confidence"""
        if not self.confidence_history:
            return None, None, 0.0
        
        # Find index of frame with highest confidence
        best_idx = np.argmax(self.confidence_history)
        
        return (
            self.frame_history[best_idx].copy() if self.frame_history[best_idx] is not None else None,
            self.keypoint_history[best_idx],
            self.confidence_history[best_idx]
        )
    
    def get_avg_confidence(self) -> float:
        """Get average confidence over history"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def has_history(self) -> bool:
        """Check if there's any history"""
        return len(self.frame_history) > 0

class MediaPipeEstimator:
    def __init__(self, enable_frame_continuity: bool = True):
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose

        # For better accuracy, use non-static mode to leverage temporal info
        # But initialize with static mode for the first frame
        self.static_pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD
        )
        
        # For subsequent frames, leverage the tracking features
        self.tracking_pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD * 0.8,  # Lower threshold for tracking
            min_tracking_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD * 0.8
        )
        
        # Set the active pose model
        self.pose = self.static_pose
        
        # Frame continuity system for better tracking
        self.enable_frame_continuity = enable_frame_continuity
        self.continuity_tracker = FrameContinuityTracker()
        self.seen_first_frame = False
        
        # Thresholds and validators
        self.keypoint_thresholds = settings.KEYPOINT_THRESHOLDS
        self.validator = PoseValidator()
        self.confidence_assessor = AdaptiveConfidenceAssessor()
        
        # Performance settings
        self.batch_size = settings.BATCH_SIZE
        self.frame_metadata = {}
        
        # Add performance monitoring
        self.processing_times = []
        
        # Set optimal number of parallel preprocessors
        self.max_preprocessors = min(4, psutil.cpu_count(logical=False) or 2)
        
        # Log MediaPipe version
        logger.info(f"Using MediaPipe version: {mp.__version__}")
        logger.info(f"Using {self.max_preprocessors} preprocessors for frame preparation")
        logger.info(f"Frame continuity tracking: {enable_frame_continuity}")

    async def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[Optional[List[Keypoint]]]:
        """Process frames using a hybrid approach: parallel preprocessing, sequential pose estimation"""
        if batch_size is None:
            batch_size = self.batch_size
            
        total_frames = len(frames)
        all_keypoints = []
        
        logger.info(f"Starting to process {total_frames} frames with hybrid approach")
        
        # Process frames in batches
        for i in range(0, total_frames, batch_size):
            batch = frames[i:i + batch_size]
            batch_size_actual = len(batch)
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_frames+batch_size-1)//batch_size}")
            
            # 1. Pre-process frames in parallel (RGB conversion and resizing if needed)
            # This part can be parallelized since it doesn't depend on previous frames
            preprocessed_frames = await self._preprocess_frames_parallel(batch)
            
            # 2. Process preprocessed frames sequentially with MediaPipe
            # This part must be sequential for MediaPipe to maintain temporal consistency
            pose_results = []
            for j, frame in enumerate(preprocessed_frames):
                start_time = time.time()
                
                # For the first frame or if continuity is disabled, use static mode
                if not self.seen_first_frame or j == 0:
                    self.pose = self.static_pose
                    self.seen_first_frame = True
                else:
                    # Switch to tracking mode for subsequent frames
                    self.pose = self.tracking_pose
                
                # Process the frame with MediaPipe
                result = await run_in_executor(self.pose.process, frame)
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                pose_results.append(result)
                
                # Small delay between frames to prevent event loop congestion
                await asyncio.sleep(0.001)
            
            # 3. Post-process results in parallel (converting to keypoints and filtering)
            # This part can be parallelized since each frame's results are independent
            batch_keypoints = await self._postprocess_results_parallel(pose_results, batch_size_actual)
            
            # 4. Update the continuity tracker for each frame if enabled
            if self.enable_frame_continuity:
                for j, (frame, keypoints) in enumerate(zip(preprocessed_frames, batch_keypoints)):
                    # Calculate average confidence for this frame's keypoints
                    avg_confidence = 0.0
                    if keypoints and len(keypoints) > 0:
                        avg_confidence = np.mean([kp.confidence for kp in keypoints if hasattr(kp, 'confidence')])
                    
                    # Add frame data to continuity tracker
                    self.continuity_tracker.add_frame(frames[i + j], keypoints, avg_confidence)
            
            all_keypoints.extend(batch_keypoints)
            
            # Small delay between batches
            await asyncio.sleep(0.01)
        
        return all_keypoints

    async def _preprocess_frames_parallel(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Pre-process frames in parallel (RGB conversion and any necessary preparation)"""
        preprocess_tasks = []
        
        for frame in frames:
            task = self._preprocess_frame(frame)
            preprocess_tasks.append(task)
        
        # Process frames in parallel with limited concurrency
        return await gather_with_concurrency(
            self.max_preprocessors,  # Limit concurrent preprocessing
            *preprocess_tasks
        )

    async def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pre-process a single frame (convert BGR to RGB)"""
        # Run in executor to avoid blocking the event loop
        return await run_in_executor(cv2.cvtColor, frame, cv2.COLOR_BGR2RGB)

    async def _postprocess_results_parallel(
        self, 
        pose_results: List[Any], 
        batch_size: int
    ) -> List[Optional[List[Keypoint]]]:
        """Post-process MediaPipe results in parallel"""
        postprocess_tasks = []
        
        for i, result in enumerate(pose_results):
            task = self._postprocess_result(result)
            postprocess_tasks.append(task)
        
        # Process results in parallel with limited concurrency
        return await gather_with_concurrency(
            self.max_preprocessors,  # Limit concurrent postprocessing
            *postprocess_tasks
        )

    async def _postprocess_result(self, result: Any) -> Optional[List[Keypoint]]:
        """Post-process a single MediaPipe result"""
        try:
            if not result.pose_landmarks:
                # If frame continuity is enabled and we have history, 
                # we could fill in missing poses from previous frames
                if self.enable_frame_continuity and self.continuity_tracker.has_history():
                    _, best_keypoints, best_confidence = self.continuity_tracker.get_best_reference_frame()
                    if best_confidence > settings.GLOBAL_CONFIDENCE_THRESHOLD * 1.2:
                        # Only use history if the best frame had good confidence
                        logger.debug(f"Using historical keypoints with confidence {best_confidence:.3f}")
                        
                        # Reduce confidence slightly to indicate these are interpolated
                        if best_keypoints:
                            reduced_keypoints = []
                            for kp in best_keypoints:
                                reduced_keypoints.append(
                                    Keypoint(
                                        x=kp.x,
                                        y=kp.y,
                                        confidence=kp.confidence * 0.8,  # Reduce confidence for historical points
                                        name=kp.name
                                    )
                                )
                            return reduced_keypoints
                
                return None
            
            # Convert landmarks to keypoints
            keypoints = [
                Keypoint(
                    x=landmark.x,
                    y=landmark.y,
                    confidence=landmark.visibility,
                    name=self.mp_pose.PoseLandmark(idx).name
                )
                for idx, landmark in enumerate(result.pose_landmarks.landmark)
            ]
            
            # Apply confidence thresholding
            filtered_keypoints = await self._filter_keypoints(keypoints)
            
            # Basic validation
            if not self._validate_pose(filtered_keypoints):
                return None
            
            return filtered_keypoints
            
        except Exception as e:
            logger.error(f"Error post-processing result: {str(e)}")
            return None

    async def _filter_keypoints(self, keypoints: List[Keypoint]) -> List[Keypoint]:
        """Filter keypoints with confidence assessment"""
        if not keypoints:
            return []
        
        positions = {kp.name: (kp.x, kp.y) for kp in keypoints}
        confidences = {kp.name: kp.confidence for kp in keypoints}
        
        adjusted_confidences = await self.confidence_assessor.assess_keypoints(
            positions,
            confidences
        )
        
        return [
            Keypoint(
                x=kp.x,
                y=kp.y,
                confidence=adjusted_confidences.get(kp.name, kp.confidence),
                name=kp.name
            )
            for kp in keypoints
            if adjusted_confidences.get(kp.name, 0) > self.keypoint_thresholds.get(
                kp.name, settings.GLOBAL_CONFIDENCE_THRESHOLD
            )
        ]

    def _validate_pose(self, keypoints: List[Keypoint]) -> bool:
        """Validate if enough critical keypoints are detected"""
        if not keypoints:
            return False
        
        critical_points = {
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_HIP", "RIGHT_HIP"
        }
        
        detected_critical = {
            kp.name for kp in keypoints 
            if kp.name in critical_points
        }
        
        return len(detected_critical) >= 3

    def __del__(self):
        """Cleanup resources with proper error handling"""
        try:
            # Cleanup both pose objects
            if hasattr(self, 'static_pose'):
                try:
                    self.static_pose.close()
                except Exception:
                    pass
                    
            if hasattr(self, 'tracking_pose'):
                try:
                    self.tracking_pose.close()
                except Exception:
                    pass
        except Exception as e:
            # Specifically ignore MediaPipe timestamp errors during cleanup
            error_msg = str(e)
            if "Packet timestamp mismatch" not in error_msg and "CalculatorGraph" not in error_msg:
                logger.error(f"Error during cleanup: {error_msg}")