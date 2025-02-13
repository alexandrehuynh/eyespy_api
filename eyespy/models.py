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

class PoseEstimationResponse(BaseModel):
    status: ProcessingStatus
    keypoints: Optional[List[Keypoint]] = None
    metadata: Dict[str, Any] = {}  
    confidence_metrics: Dict[str, float] = {}
    validation_metrics: Dict[str, Any] = {} 
    error: Optional[str] = None