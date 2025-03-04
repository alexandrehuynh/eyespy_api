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
            static_image_mode=True,
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
        
        # Log MediaPipe version
        logger.info(f"Using MediaPipe version: {mp.__version__}")

    async def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[Optional[List[Keypoint]]]:
        """Process frames sequentially to maintain timestamp order"""
        if batch_size is None:
            batch_size = self.batch_size
            
        total_frames = len(frames)
        all_keypoints = []
        
        logger.info(f"Starting to process {total_frames} frames sequentially")
        
        # Process frames in strict sequential order, batch by batch
        for i in range(0, total_frames, batch_size):
            batch = frames[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_frames+batch_size-1)//batch_size}")
            
            # Process one batch at a time
            batch_keypoints = await self._process_batch(batch)
            all_keypoints.extend(batch_keypoints)
            
            # Small delay between batches to prevent event loop congestion
            await asyncio.sleep(0.001)
        
        return all_keypoints

    async def _process_batch(self, batch: List[np.ndarray]) -> List[Optional[List[Keypoint]]]:
        """Process batch frames sequentially to maintain timestamp order"""
        results = []
        
        # Process each frame in the batch sequentially with a small delay
        for i, frame in enumerate(batch):
            logger.debug(f"Processing frame {i} in batch")
            result = await self._process_single_frame(frame)
            results.append(result)
            
            # Increase the delay to give MediaPipe more time between frames
            await asyncio.sleep(0.01)  # Increased from 0.001
        
        return results

    async def _process_single_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame with MediaPipe"""
        try:
            start_time = time.time()
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Sending frame to MediaPipe, size: {frame.shape}")
            # Process the frame with MediaPipe using shared executor
            results = await run_in_executor(self.pose.process, rgb_frame)
            process_time = time.time() - start_time

            if not results.pose_landmarks:
                logger.debug(f"No pose landmarks detected, processing took {process_time:.3f}s")
                return None
            
            logger.debug(f"Got pose landmarks, processing took {process_time:.3f}s")
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
                # Use a small synchronous delay instead of asyncio.sleep
                # This avoids the "coroutine was never awaited" warning
                try:
                    import time
                    time.sleep(0.1)  # Use time.sleep instead of asyncio.sleep
                except:
                    pass
                    
                # Close pose object
                self.pose.close()
        except Exception as e:
            # Specifically ignore MediaPipe timestamp errors during cleanup
            error_msg = str(e)
            if "Packet timestamp mismatch" not in error_msg and "CalculatorGraph" not in error_msg:
                logger.error(f"Error during cleanup: {error_msg}")