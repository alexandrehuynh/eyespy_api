import mediapipe as mp
import numpy as np
from typing import List, Optional, Dict, Tuple
from ..models import Keypoint
from ..config import settings
import asyncio
import cv2
from concurrent.futures import ThreadPoolExecutor
from .validation import PoseValidator
from .confidence import AdaptiveConfidenceAssessor
from .tracker import ConfidenceTracker
from .movenet_estimator import MovenetEstimator
from .fusion import PoseFusion
import time
import psutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipeEstimator:
    def __init__(self):
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD
        )
        
        # Thresholds and validators
        self.keypoint_thresholds = settings.KEYPOINT_THRESHOLDS
        self.validator = PoseValidator()
        
        # Initialize components for dual-model system
        self.movenet = MovenetEstimator()
        self.fusion = PoseFusion()
        self.confidence_tracker = ConfidenceTracker()
        
        # Performance optimizations
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
        self.executor = ThreadPoolExecutor(
            max_workers=max(4, cpu_count - 1)  # Leave one core free for other tasks
        )
        self.batch_size = 50  # Increased from default
        self.frame_metadata = {}
        
        # Add performance monitoring
        self.processing_times: List[float] = []

    async def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[Optional[List[Keypoint]]]:
        """Process frames using optimized batching"""
        if batch_size is None:
            batch_size = self.batch_size
            
        all_keypoints = []
        total_frames = len(frames)
        
        # Process in optimized batches
        for i in range(0, total_frames, batch_size):
            batch_start = time.time()
            
            # Get current batch
            batch = frames[i:i + batch_size]
            
            try:
                # Process batch
                batch_keypoints = await self._process_mediapipe_batch(batch)
                all_keypoints.extend(batch_keypoints)
                
                # Record performance
                batch_time = time.time() - batch_start
                self.processing_times.append(batch_time)
                
                # Log performance
                fps = len(batch) / batch_time
                logger.info(
                    f"MediaPipe Batch {i//batch_size}: "
                    f"Processed {len(batch)} frames at {fps:.1f} FPS"
                )
                
            except Exception as e:
                logger.error(f"Error in MediaPipe batch {i//batch_size}: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
            
            # Allow other tasks to run
            await asyncio.sleep(0)
            
        return all_keypoints

    async def _process_mediapipe_batch(
        self,
        batch: List[np.ndarray]
    ) -> List[Optional[List[Keypoint]]]:
        """Process a batch of frames with MediaPipe in parallel"""
        # Process frames in parallel using asyncio.gather
        tasks = [self._process_single_frame(frame) for frame in batch]
        return await asyncio.gather(*tasks)

    async def _process_batch_results(
        self,
        mp_keypoints: List[Optional[List[Keypoint]]],
        mn_keypoints: List[Optional[List[Keypoint]]]
    ) -> List[Optional[List[Keypoint]]]:
        """Process and fuse batch results"""
        processed_keypoints = []
        
        for mp_kp, mn_kp in zip(mp_keypoints, mn_keypoints):
            # Fuse predictions
            fused_keypoints = await self.fusion.fuse_predictions(mp_kp, mn_kp)
            
            if fused_keypoints:
                # Convert to format for tracker
                kp_dict = {
                    kp.name: (kp.x, kp.y, kp.confidence)
                    for kp in fused_keypoints
                }
                
                # Update tracking
                tracked_kp = await self.confidence_tracker.update(kp_dict)
                
                # Convert back to keypoints
                tracked_keypoints = [
                    Keypoint(
                        name=name,
                        x=pos[0],
                        y=pos[1],
                        confidence=pos[2]
                    )
                    for name, pos in tracked_kp.items()
                ]
                
                # Validate tracked pose
                is_valid, validation_metrics = self.validator.validate_pose(tracked_keypoints)
                
                if is_valid:
                    self.frame_metadata.update(validation_metrics)
                    processed_keypoints.append(tracked_keypoints)
                else:
                    processed_keypoints.append(None)
            else:
                processed_keypoints.append(None)
        
        return processed_keypoints

    async def _process_single_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame with MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.pose.process(rgb_frame)
            )
            
            if not results.pose_landmarks:
                return None
            
            # Convert landmarks to keypoints
            keypoints = [
                Keypoint(
                    x=landmark.x,
                    y=landmark.y,
                    confidence=landmark.visibility,
                    name=self.mp_pose.PoseLandmark(idx).name
                )
                for idx, landmark in enumerate(results.pose_landmarks.landmark)
            ]
            
            # Apply confidence thresholding
            filtered_keypoints = await self._filter_keypoints(keypoints)
            
            # Basic validation
            if not self._validate_pose(filtered_keypoints):
                return None
            
            return filtered_keypoints
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    async def _filter_keypoints(self, keypoints: List[Keypoint]) -> List[Keypoint]:
        """Filter keypoints with enhanced confidence assessment"""
        if not keypoints:
            return []
        
        confidence_assessor = AdaptiveConfidenceAssessor()
        positions = {kp.name: (kp.x, kp.y) for kp in keypoints}
        confidences = {kp.name: kp.confidence for kp in keypoints}
        
        adjusted_confidences = await confidence_assessor.assess_keypoints(
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
            if adjusted_confidences.get(kp.name, 0) > 0
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
        """Cleanup resources"""
        try:
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")