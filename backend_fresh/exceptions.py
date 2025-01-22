"""Custom exceptions for video processing application."""

class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class MediaPipeError(VideoProcessingError):
    """Errors related to MediaPipe processing"""
    pass

class ResourceError(VideoProcessingError):
    """Errors related to resource management"""
    pass

class FrameProcessingError(VideoProcessingError):
    """Errors related to frame processing"""
    pass

class FrameValidationError(VideoProcessingError):
    """Errors related to frame validation"""
    pass