import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
import time
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor, gather_with_concurrency

logger = logging.getLogger(__name__)

@dataclass
class QualityThresholds:
    # Brightness thresholds (0-1)
    min_brightness: float = 0.1  # Was higher
    max_brightness: float = 0.9
    optimal_brightness: float = 0.5
    
    # Contrast thresholds (0-1)
    min_contrast: float = 0.1  # Was higher
    optimal_contrast: float = 0.5
    
    # Blur thresholds
    min_blur_score: float = 10.0  # Was much higher, lowering to match actual video quality
    optimal_blur_score: float = 50.0
    
    # Coverage thresholds (0-1)
    min_coverage: float = 0.1
    optimal_coverage: float = 0.8
    
    # Overall quality threshold
    min_overall_score: float = 0.5   # Minimum acceptable overall quality

    def to_dict(self) -> Dict[str, float]:
        return {
            'min_brightness': self.min_brightness,
            'max_brightness': self.max_brightness,
            'optimal_brightness': self.optimal_brightness,
            'min_contrast': self.min_contrast,
            'optimal_contrast': self.optimal_contrast,
            'min_blur_score': self.min_blur_score,
            'optimal_blur_score': self.optimal_blur_score,
            'min_coverage': self.min_coverage,
            'optimal_coverage': self.optimal_coverage,
            'min_overall_score': self.min_overall_score
        }

@dataclass
class QualityMetrics:
    is_valid: bool
    brightness: float
    contrast: float
    blur_score: float
    coverage_score: float
    overall_score: float
    details: Dict[str, float] = field(default_factory=dict)

class AdaptiveFrameQualityAssessor:
    def __init__(self, initial_thresholds: Optional[QualityThresholds] = None):
        self.thresholds = initial_thresholds or QualityThresholds()
        self.calibrated = False
        self.calibration_size = 30
        self.calibration_stats = {
            'brightness': [],
            'contrast': [],
            'blur': [],
            'coverage': []
        }

    async def calibrate_thresholds(self, calibration_frames: List[np.ndarray]) -> bool:
        """Calibrate quality thresholds based on sample frames"""
        try:
            logger.info(f"Starting calibration with {len(calibration_frames)} frames")
            if not calibration_frames:
                return False

            # Process calibration frames in parallel using the shared executor
            tasks = []
            for frame in calibration_frames:
                if frame is None:
                    continue
                tasks.append(self._process_calibration_frame(frame))
            
            results = await asyncio.gather(*tasks)
            
            # Update calibration stats with results
            for brightness, contrast, blur, coverage in results:
                self.calibration_stats['brightness'].append(brightness)
                self.calibration_stats['contrast'].append(contrast)
                self.calibration_stats['blur'].append(blur)
                self.calibration_stats['coverage'].append(coverage)

            # Calculate adaptive thresholds
            if all(len(stats) > 0 for stats in self.calibration_stats.values()):
                # Update brightness thresholds
                brightness_mean = np.mean(self.calibration_stats['brightness'])
                self.thresholds.min_brightness = max(0.1, brightness_mean - 0.2)
                self.thresholds.max_brightness = min(0.9, brightness_mean + 0.2)
                self.thresholds.optimal_brightness = brightness_mean

                # Update contrast thresholds
                contrast_mean = np.mean(self.calibration_stats['contrast'])
                self.thresholds.min_contrast = max(0.1, contrast_mean - 0.15)
                self.thresholds.optimal_contrast = contrast_mean

                # Update blur thresholds
                blur_mean = np.mean(self.calibration_stats['blur'])
                self.thresholds.min_blur_score = max(20, blur_mean * 0.5)
                self.thresholds.optimal_blur_score = blur_mean

                # Update coverage thresholds
                coverage_mean = np.mean(self.calibration_stats['coverage'])
                self.thresholds.min_coverage = max(0.1, coverage_mean - 0.2)
                self.thresholds.optimal_coverage = coverage_mean

                logger.info("Calibration completed successfully")
                logger.info(f"Adjusted thresholds: {self.thresholds.to_dict()}")
                
                self.calibrated = True
                return True
            
            return False

        except Exception as e:
            logger.error(f"Error during calibration: {str(e)}")
            return False

    async def _process_calibration_frame(self, frame: np.ndarray) -> Tuple[float, float, float, float]:
        """Process a single calibration frame"""
        # Convert to grayscale
        gray = await run_in_executor(cv2.cvtColor, frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics in parallel
        brightness_task = run_in_executor(self._calculate_brightness, frame)
        contrast_task = run_in_executor(self._calculate_contrast, frame)
        blur_task = run_in_executor(self._calculate_blur, frame)
        coverage_task = run_in_executor(self._calculate_coverage, frame)
        
        brightness, contrast, blur, coverage = await asyncio.gather(
            brightness_task, contrast_task, blur_task, coverage_task
        )
        
        return brightness, contrast, blur, coverage

    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255.0

    def _calculate_contrast(self, frame: np.ndarray) -> float:
        """Calculate contrast using standard deviation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 255.0

    def _calculate_blur(self, frame: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_coverage(self, frame: np.ndarray) -> float:
        """Calculate frame coverage (non-black areas)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        return np.count_nonzero(thresh) / thresh.size

    def _calculate_overall_score(self, brightness: float, contrast: float, blur: float, coverage: float) -> float:
        """Calculate overall quality score with adjusted weights"""
        # Normalize blur score (higher is better)
        norm_blur = min(1.0, blur / self.thresholds.optimal_blur_score)
        
        # Adjusted weights to be more lenient
        weights = {
            'brightness': 0.2,
            'contrast': 0.2,
            'blur': 0.3,      # Increased weight for blur
            'coverage': 0.3   # Increased weight for coverage
        }
        
        # Calculate individual scores
        brightness_score = 1 - abs(brightness - self.thresholds.optimal_brightness) * 2
        contrast_score = contrast / self.thresholds.optimal_contrast
        blur_score = norm_blur
        coverage_score = coverage / self.thresholds.optimal_coverage
        
        # Calculate weighted average
        score = (
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score +
            weights['blur'] * blur_score +
            weights['coverage'] * coverage_score
        )
        
        return max(0.0, min(1.0, score))

    async def assess_frame(self, frame: np.ndarray) -> QualityMetrics:
        """Assess frame quality with detailed logging"""
        try:
            # Validate frame
            if frame is None:
                logger.error("Received None frame")
                return self._create_invalid_metrics("Frame is None")
            
            if not isinstance(frame, np.ndarray):
                logger.error(f"Invalid frame type: {type(frame)}")
                return self._create_invalid_metrics(f"Invalid frame type: {type(frame)}")
            
            if len(frame.shape) != 3:
                logger.error(f"Invalid frame dimensions: {frame.shape}")
                return self._create_invalid_metrics(f"Invalid frame dimensions: {frame.shape}")
            
            if frame.shape[2] != 3:  # Check if frame is BGR
                logger.error(f"Invalid color channels: {frame.shape[2]}")
                return self._create_invalid_metrics(f"Invalid color channels: {frame.shape[2]}")

            # Ensure frame is in uint8 format
            if frame.dtype != np.uint8:
                logger.warning(f"Converting frame from {frame.dtype} to uint8")
                frame = frame.astype(np.uint8)

            # Log frame properties
            logger.debug(f"Assessing frame: shape={frame.shape}, dtype={frame.dtype}")
            
            # Calculate metrics in parallel using shared executor
            brightness_task = run_in_executor(self._calculate_brightness, frame)
            contrast_task = run_in_executor(self._calculate_contrast, frame)
            blur_task = run_in_executor(self._calculate_blur, frame)
            coverage_task = run_in_executor(self._calculate_coverage, frame)
            
            brightness, contrast, blur, coverage = await asyncio.gather(
                brightness_task, contrast_task, blur_task, coverage_task
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(brightness, contrast, blur, coverage)
            
            # Create detailed metrics dictionary
            details = {
                'raw_brightness': brightness,
                'raw_contrast': contrast,
                'raw_blur': blur,
                'raw_coverage': coverage,
                'brightness_score': brightness,
                'contrast_score': contrast,
                'blur_score': blur,
                'coverage_score': coverage,
                'overall_score': overall_score,
                'threshold_values': self.thresholds.to_dict()
            }
            
            # Create metrics object
            metrics = QualityMetrics(
                is_valid=True,
                brightness=brightness,
                contrast=contrast,
                blur_score=blur,
                coverage_score=coverage,
                overall_score=overall_score,
                details=details
            )
            
            # Validate against thresholds
            if (brightness < self.thresholds.min_brightness or 
                brightness > self.thresholds.max_brightness or
                contrast < self.thresholds.min_contrast or
                blur < self.thresholds.min_blur_score or
                coverage < self.thresholds.min_coverage):
                
                metrics.is_valid = False
                logger.warning(f"""
Frame failed quality thresholds:
- Brightness: {brightness:.3f} {'✓' if self.thresholds.min_brightness <= brightness <= self.thresholds.max_brightness else '✗'}
- Contrast: {contrast:.3f} {'✓' if contrast >= self.thresholds.min_contrast else '✗'}
- Blur: {blur:.3f} {'✓' if blur >= self.thresholds.min_blur_score else '✗'}
- Coverage: {coverage:.3f} {'✓' if coverage >= self.thresholds.min_coverage else '✗'}
                """)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing frame quality: {str(e)}")
            return self._create_invalid_metrics(str(e))

    def _create_invalid_metrics(self, error_message: str) -> QualityMetrics:
        """Create metrics object for invalid frames"""
        return QualityMetrics(
            is_valid=False,
            brightness=0.0,
            contrast=0.0,
            blur_score=0.0,
            coverage_score=0.0,
            overall_score=0.0,
            details={
                'error': error_message,
                'timestamp': time.time()
            }
        )

    def __del__(self):
        """Cleanup executor on deletion"""
        self.executor.shutdown(wait=False)