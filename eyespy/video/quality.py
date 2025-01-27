from dataclasses import dataclass
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional

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
        self.calibration_size = 30  # Number of frames for calibration

    def calculate_quality_score(self, metric: float, min_val: float, optimal_val: float, max_val: Optional[float] = None) -> float:
        """Calculate normalized quality score"""
        if max_val is not None:
            if metric < min_val or metric > max_val:
                return 0.0
            elif metric < optimal_val:
                return (metric - min_val) / (optimal_val - min_val)
            else:
                return 1.0 - ((metric - optimal_val) / (max_val - optimal_val))
        else:
            if metric < min_val:
                return 0.0
            elif metric < optimal_val:
                return (metric - min_val) / (optimal_val - min_val)
            else:
                return 1.0

    def calibrate_thresholds(self, frames: List[np.ndarray]) -> QualityThresholds:
        """Calibrate thresholds based on sample frames"""
        metrics = []
        for frame in frames[:self.calibration_size]:
            quality = self.assess_frame(frame)
            metrics.append(quality)
        
        # Calculate statistics from sample frames
        brightness_values = [m.details['raw_brightness'] for m in metrics]
        contrast_values = [m.details['raw_contrast'] for m in metrics]
        blur_values = [m.details['raw_blur'] for m in metrics]
        coverage_values = [m.details['raw_coverage'] for m in metrics]
        
        # Adjust thresholds based on observed values
        self.thresholds = QualityThresholds(
            min_brightness=max(0.1, np.percentile(brightness_values, 10)),
            max_brightness=min(0.9, np.percentile(brightness_values, 90)),
            optimal_brightness=np.median(brightness_values),
            
            min_contrast=max(0.1, np.percentile(contrast_values, 10)),
            optimal_contrast=np.median(contrast_values),
            
            min_blur_score=max(30, np.percentile(blur_values, 10)),
            optimal_blur_score=np.median(blur_values),
            
            min_coverage=max(0.1, np.percentile(coverage_values, 10)),
            optimal_coverage=np.median(coverage_values)
        )
        
        self.calibrated = True
        return self.thresholds

    def assess_frame(self, frame: np.ndarray) -> QualityMetrics:
        """Assess frame quality with adaptive thresholds"""
        # Basic measurements
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_blur = min(1.0, blur_score / self.thresholds.optimal_blur_score)
        
        # Calculate coverage
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        coverage = np.count_nonzero(thresh) / thresh.size
        
        # Calculate quality scores
        brightness_score = self.calculate_quality_score(
            brightness,
            self.thresholds.min_brightness,
            self.thresholds.optimal_brightness,
            self.thresholds.max_brightness
        )
        
        contrast_score = self.calculate_quality_score(
            contrast,
            self.thresholds.min_contrast,
            self.thresholds.optimal_contrast
        )
        
        blur_score = self.calculate_quality_score(
            normalized_blur,
            self.thresholds.min_blur_score / self.thresholds.optimal_blur_score,
            1.0
        )
        
        coverage_score = self.calculate_quality_score(
            coverage,
            self.thresholds.min_coverage,
            self.thresholds.optimal_coverage
        )
        
        # Calculate weighted overall score
        weights = {
            'brightness': 0.25,
            'contrast': 0.25,
            'blur': 0.3,
            'coverage': 0.2
        }
        
        overall_score = (
            brightness_score * weights['brightness'] +
            contrast_score * weights['contrast'] +
            blur_score * weights['blur'] +
            coverage_score * weights['coverage']
        )
        
        is_valid = overall_score >= self.thresholds.min_overall_score
        
        return QualityMetrics(
            brightness=brightness_score,
            contrast=contrast_score,
            blur_score=blur_score,
            coverage_score=coverage_score,
            overall_score=overall_score,
            is_valid=is_valid,
            details={
                'raw_brightness': brightness,
                'raw_contrast': contrast,
                'raw_blur': normalized_blur,
                'raw_coverage': coverage,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'blur_score': blur_score,
                'coverage_score': coverage_score,
                'threshold_values': self.thresholds.to_dict()
            }
        )