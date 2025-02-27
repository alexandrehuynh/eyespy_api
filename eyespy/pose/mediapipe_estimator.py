import mediapipe as mp
import numpy as np
from typing import List, Optional, Dict, Tuple
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
from ..utils.async_utils import run_in_executor

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
        self.confidence_assessor = AdaptiveConfidenceAssessor()
        
        # Performance settings
        self.batch_size = settings.BATCH_SIZE
        self.frame_metadata = {}
        
        # Add performance monitoring
        self.processing_times = []

    async def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[Optional[List[Keypoint]]]:
        """Process frames using optimized batching"""
        if batch_size is None:
            batch_size = self.batch_size
            
        total_frames = len(frames)
        tasks = []
        
        # Create all tasks at once
        for i in range(0, total_frames, batch_size):
            batch = frames[i:i + batch_size]
            tasks.append(self._process_batch(batch))
        
        # Process all batches concurrently
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_keypoints = []
        for batch_keypoints in results:
            all_keypoints.extend(batch_keypoints)
        
        return all_keypoints

    async def _process_batch(
        self,
        batch: List[np.ndarray]
    ) -> List[Optional[List[Keypoint]]]:
        """Process a batch of frames with MediaPipe"""
        # Process frames in parallel using asyncio.gather
        tasks = [self._process_single_frame(frame) for frame in batch]
        return await asyncio.gather(*tasks)

    async def _process_single_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame with MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe using shared executor
            results = await run_in_executor(self.pose.process, rgb_frame)
            
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
            logger.error(f"Error processing frame: {str(e)}")
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
            if hasattr(self, 'pose'):
                # Add a short delay to ensure all processing is complete
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.run_until_complete(asyncio.sleep(0.1))
                except:
                    pass
                    
                # Close pose object
                self.pose.close()
        except Exception as e:
            # Specifically ignore MediaPipe timestamp errors during cleanup
            error_msg = str(e)
            if "Packet timestamp mismatch" not in error_msg and "CalculatorGraph" not in error_msg:
                logger.error(f"Error during cleanup: {error_msg}")