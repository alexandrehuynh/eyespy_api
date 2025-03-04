from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float
    name: str

class ConfidenceMetrics(BaseModel):
    average_confidence: float
    keypoints: Dict[str, Any]

class PoseEstimationResponse(BaseModel):
    status: ProcessingStatus
    keypoints: Optional[List[Keypoint]] = None
    metadata: Dict[str, Any] = {}  
    confidence_metrics: ConfidenceMetrics = ConfidenceMetrics(
        average_confidence=0.0,
        keypoints={}
    )
    validation_metrics: Dict[str, Any] = {} 
    error: Optional[str] = None

class VideoRenderingResponse(BaseModel):
    status: ProcessingStatus
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class RepetitionData(BaseModel):
    reps: int
    avg_form_score: float
    avg_symmetry_score: float

class MovementQuality(BaseModel):
    score: float
    assessment: str
    recommendations: List[str]

class MovementAnalysisResult(BaseModel):
    repetitions: Dict[str, RepetitionData]
    form_issues: List[str]
    metrics: Dict[str, Any]
    movement_quality: MovementQuality

class MovementAnalysisResponse(BaseModel):
    status: ProcessingStatus
    movement_analysis: Optional[MovementAnalysisResult] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class CompleteProcessingResponse(BaseModel):
    status: ProcessingStatus
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    movement_analysis: Optional[MovementAnalysisResult] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None