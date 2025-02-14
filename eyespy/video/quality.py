import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    brightness: float
    contrast: float
    blur_score: float
    coverage_score: float
    overall_score: float
    is_valid: bool
    details: Dict[str, float]

class AdaptiveFrameQualityAssessor:
    def __init__(self, initial_thresholds: Optional[QualityThresholds] = None):
        self.thresholds = initial_thresholds or QualityThresholds()
        self.calibrated = False
        self.calibration_size = 30
        self.executor = ThreadPoolExecutor(max_workers=4)
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
                self.calibration_stats['brightness'].append(self._measure_brightness(gray))
                self.calibration_stats['contrast'].append(self._measure_contrast(gray))
                self.calibration_stats['blur'].append(cv2.Laplacian(gray, cv2.CV_64F).var())
                self.calibration_stats['coverage'].append(self._measure_coverage(gray))

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

    async def assess_frame(self, frame: np.ndarray) -> QualityMetrics:
        """Assess frame quality with parallel measurements"""
        # Convert to grayscale once for all measurements
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create tasks for parallel execution
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._measure_brightness, gray),
            loop.run_in_executor(self.executor, self._measure_contrast, gray),
            loop.run_in_executor(self.executor, self._measure_blur, gray),
            loop.run_in_executor(self.executor, self._measure_coverage, gray)
        ]
        
        # Wait for all measurements to complete
        brightness, contrast, blur, coverage = await asyncio.gather(*tasks)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            brightness, contrast, blur, coverage
        )
        
        # Determine validity
        is_valid = self._check_validity(
            brightness, contrast, blur, coverage, overall_score
        )
        
        return QualityMetrics(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur,
            coverage_score=coverage,
            overall_score=overall_score,
            is_valid=is_valid,
            details=self._create_details(
                brightness, contrast, blur, coverage, overall_score
            )
        )


    def _measure_brightness(self, gray: np.ndarray) -> float:
        """Measure frame brightness"""
        return np.mean(gray) / 255.0

    def _measure_contrast(self, gray: np.ndarray) -> float:
        """Measure frame contrast"""
        return np.std(gray) / 255.0

    def _measure_blur(self, gray: np.ndarray) -> float:
        """Measure frame blurriness"""
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(1.0, blur_score / self.thresholds.optimal_blur_score)

    def _measure_coverage(self, gray: np.ndarray) -> float:
        """Measure frame coverage"""
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        return np.count_nonzero(thresh) / thresh.size

    def _calculate_overall_score(
        self,
        brightness: float,
        contrast: float,
        blur: float,
        coverage: float
    ) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'brightness': 0.25,
            'contrast': 0.25,
            'blur': 0.3,
            'coverage': 0.2
        }
        
        return (
            brightness * weights['brightness'] +
            contrast * weights['contrast'] +
            blur * weights['blur'] +
            coverage * weights['coverage']
        )

    def _check_validity(
        self,
        brightness: float,
        contrast: float,
        blur: float,
        coverage: float,
        overall_score: float
    ) -> bool:
        """Check if frame meets quality thresholds"""
        return all([
            self.thresholds.min_brightness <= brightness <= self.thresholds.max_brightness,
            contrast >= self.thresholds.min_contrast,
            blur >= self.thresholds.min_blur_score / self.thresholds.optimal_blur_score,
            coverage >= self.thresholds.min_coverage,
            overall_score >= self.thresholds.min_overall_score
        ])

    def _create_details(
        self,
        brightness: float,
        contrast: float,
        blur: float,
        coverage: float,
        overall_score: float
    ) -> Dict[str, float]:
        """Create detailed metrics dictionary"""
        return {
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

    async def process_batch(self, frames: List[np.ndarray]) -> List[QualityMetrics]:
        """Process multiple frames in parallel"""
        return await asyncio.gather(
            *[self.assess_frame(frame) for frame in frames]
        )

    def __del__(self):
        """Cleanup executor on deletion"""
        self.executor.shutdown(wait=False)