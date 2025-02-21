import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityThresholds:
    # Brightness thresholds (0-1)
    min_brightness: float = 0.15    # Dark but still visible
    max_brightness: float = 0.85    # Bright but not overexposed
    optimal_brightness: float = 0.5  # Ideal brightness
    
    # Contrast thresholds (0-1)
    min_contrast: float = 0.15      # Minimum acceptable contrast
    optimal_contrast: float = 0.4    # Ideal contrast
    
    # Blur thresholds
    min_blur_score: float = 50      # Minimum Laplacian variance
    optimal_blur_score: float = 150  # Ideal sharpness
    
    # Coverage thresholds (0-1)
    min_coverage: float = 0.15      # Minimum subject coverage
    optimal_coverage: float = 0.4    # Ideal subject coverage
    
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
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.calibrated = False
        self.calibration_size = 30
        self.calibration_stats = {
            'brightness': [],
            'contrast': [],
            'blur': [],
            'coverage': []
        }

    def calibrate_thresholds(self, calibration_frames: List[np.ndarray]) -> bool:
        """Calibrate quality thresholds based on sample frames"""
        try:
            print(f"Starting calibration with {len(calibration_frames)} frames")
            if not calibration_frames:
                return False

            # Process calibration frames
            for frame in calibration_frames:
                if frame is None:
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Collect metrics
                self.calibration_stats['brightness'].append(self._calculate_brightness(frame))
                self.calibration_stats['contrast'].append(self._calculate_contrast(frame))
                self.calibration_stats['blur'].append(self._calculate_blur(frame))
                self.calibration_stats['coverage'].append(self._calculate_coverage(frame))

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

                print("Calibration completed successfully")
                print(f"Adjusted thresholds: {self.thresholds.to_dict()}")
                
                self.calibrated = True
                return True
            
            return False

        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            return False

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
        """Calculate overall quality score"""
        # Normalize blur score (higher is better)
        norm_blur = min(1.0, blur / 1000.0)  # Adjust divisor based on your needs
        
        # Weighted average of all metrics
        weights = {
            'brightness': 0.25,
            'contrast': 0.25,
            'blur': 0.25,
            'coverage': 0.25
        }
        
        score = (
            weights['brightness'] * (1 - abs(brightness - 0.5) * 2) +  # Closer to 0.5 is better
            weights['contrast'] * contrast +
            weights['blur'] * norm_blur +
            weights['coverage'] * coverage
        )
        
        return max(0.0, min(1.0, score))

    async def assess_frame(self, frame: np.ndarray) -> QualityMetrics:
        """Assess frame quality with detailed logging"""
        try:
            # Log frame properties
            logger.debug(f"Assessing frame: shape={frame.shape}, dtype={frame.dtype}")
            
            # Calculate metrics
            brightness = self._calculate_brightness(frame)
            contrast = self._calculate_contrast(frame)
            blur = self._calculate_blur(frame)
            coverage = self._calculate_coverage(frame)
            
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
            return QualityMetrics(
                is_valid=False,
                brightness=0.0,
                contrast=0.0,
                blur_score=0.0,
                coverage_score=0.0,
                overall_score=0.0,
                details={}
            )

    def __del__(self):
        """Cleanup executor on deletion"""
        self.executor.shutdown(wait=False)