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

class VideoWriterError(VideoProcessingError):
    """Errors related to video writing operations"""
    pass

class IOError(VideoProcessingError):
    """Errors related to Input/Output operations"""
    pass

class ValidationError(VideoProcessingError):
    """Errors related to validation"""
    pass

class RenderingError(VideoProcessingError):
    """Errors related to visual effects rendering"""
    pass

class DrawingError(RenderingError):
    """Errors related to drawing operations"""
    pass

class EffectProcessingError(RenderingError):
    """Errors related to effect processing"""
    pass