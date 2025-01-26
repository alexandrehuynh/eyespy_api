from pydantic import BaseModel
from typing import List, Dict, Optional
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
    metadata: Dict[str, any] = {}
    error: Optional[str] = None