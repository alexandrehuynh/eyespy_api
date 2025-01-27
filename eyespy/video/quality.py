# Create new file: app/video/quality.py
import cv2
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    brightness: float
    contrast: float
    blur_score: float
    coverage_score: float
    overall_score: float
    is_valid: bool
    details: Dict[str, float]

class FrameQualityAssessor:
    def __init__(
        self,
        min_brightness: float = 0.1,
        max_brightness: float = 0.9,
        min_contrast: float = 0.1,
        min_blur_score: float = 100,
        min_coverage: float = 0.1,
        min_overall_score: float = 0.6
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_blur_score = min_blur_score
        self.min_coverage = min_coverage
        self.min_overall_score = min_overall_score

    def assess_brightness(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Assess frame brightness"""
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        
        # Check if brightness is within acceptable range
        is_valid = self.min_brightness <= mean_brightness <= self.max_brightness
        
        return mean_brightness, is_valid

    def assess_contrast(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Assess frame contrast"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast using standard deviation
        contrast = np.std(gray) / 255.0
        is_valid = contrast >= self.min_contrast
        
        return contrast, is_valid

    def assess_blur(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Assess frame blurriness using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (higher = less blurry)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_valid = blur_score >= self.min_blur_score
        
        # Normalize score for consistency
        normalized_score = min(1.0, blur_score / (self.min_blur_score * 2))
        
        return normalized_score, is_valid

    def assess_coverage(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Assess if the frame has enough non-background content"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate foreground from background
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Calculate coverage ratio
        coverage = np.count_nonzero(thresh) / thresh.size
        is_valid = coverage >= self.min_coverage
        
        return coverage, is_valid

    def assess_frame(self, frame: np.ndarray) -> QualityMetrics:
        """Perform complete frame quality assessment"""
        # Get individual metrics
        brightness, brightness_valid = self.assess_brightness(frame)
        contrast, contrast_valid = self.assess_contrast(frame)
        blur_score, blur_valid = self.assess_blur(frame)
        coverage, coverage_valid = self.assess_coverage(frame)
        
        # Calculate overall score (weighted average)
        weights = {
            'brightness': 0.25,
            'contrast': 0.25,
            'blur': 0.3,
            'coverage': 0.2
        }
        
        overall_score = (
            brightness * weights['brightness'] +
            contrast * weights['contrast'] +
            blur_score * weights['blur'] +
            coverage * weights['coverage']
        )
        
        # Frame is valid if overall score and individual metrics meet minimums
        is_valid = (
            overall_score >= self.min_overall_score and
            brightness_valid and
            contrast_valid and
            blur_valid and
            coverage_valid
        )
        
        details = {
            'brightness_score': brightness,
            'contrast_score': contrast,
            'blur_score': blur_score,
            'coverage_score': coverage,
            'brightness_valid': brightness_valid,
            'contrast_valid': contrast_valid,
            'blur_valid': blur_valid,
            'coverage_valid': coverage_valid
        }
        
        return QualityMetrics(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            coverage_score=coverage,
            overall_score=overall_score,
            is_valid=is_valid,
            details=details
        )
