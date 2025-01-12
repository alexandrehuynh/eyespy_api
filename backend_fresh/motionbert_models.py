"""
MotionBERT Model Loading and Configuration Module

This module handles the loading and configuration of MotionBERT models for motion analysis.
It provides functionality to load pretrained models while allowing customizable inference
parameters through YAML configuration.

Model Architecture Parameters (Fixed):
    - dim_in (int): Input dimension (3 for x,y,z coordinates)
    - dim_out (int): Output dimension
    - dim_feat (int): Feature dimension
    - dim_rep (int): Representation dimension
    - depth (int): Transformer depth
    - num_heads (int): Number of attention heads
    - mlp_ratio (int): MLP expansion ratio
    - num_joints (int): Number of body joints (17 for standard pose)
    - maxlen (int): Maximum sequence length

Configurable Parameters (via YAML):
    Inference:
        - batch_size (int): Batch size for inference
        - confidence_threshold (float): Minimum confidence for action recognition
        - sequence_length (int): Number of frames to process at once
        - stride (int): Frame stride for sequence processing
    
    Output:
        - return_features (bool): Whether to return intermediate features
        - return_confidence (bool): Whether to return confidence scores
        - return_3d_poses (bool): Whether to return 3D pose estimates
"""

import os
import sys
import torch
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass

# Add MotionBERT/lib to sys.path
script_dir = Path(__file__).resolve().parent
motionbert_path = script_dir / "MotionBERT" / "lib"
sys.path.append(str(motionbert_path))

from model.DSTformer import DSTformer

logger = logging.getLogger(__name__)

@dataclass
class ModelPaths:
    """Stores paths for model files and configs."""
    base_dir: Path
    motionbert: Path
    pose3d: Path
    mesh: Path
    configs_dir: Path

    @classmethod
    def from_base_dir(cls, base_dir: Union[str, Path]) -> 'ModelPaths':
        """Create ModelPaths instance from base directory."""
        base = Path(base_dir)
        return cls(
            base_dir=base,
            motionbert=base / "motionbert_pretrained.bin",
            pose3d=base / "MB_ft_h36m.bin",
            mesh=base / "MB_ft_pw3d.bin",
            configs_dir=base / "configs"
        )

@dataclass
class InferenceConfig:
    """Configurable inference parameters."""
    batch_size: int = 32
    confidence_threshold: float = 0.5
    sequence_length: int = 243
    stride: int = 1
    return_features: bool = False
    return_confidence: bool = True
    return_3d_poses: bool = True
    
    # Potential exercise-specific parameters we could add:
    rep_detection_threshold: float = 0.8  # Threshold for detecting a single rep
    min_rep_duration: int = 15  # Minimum frames for a valid rep
    joint_angle_tolerance: float = 5.0  # Degrees of tolerance for joint angles
    range_of_motion_threshold: float = 0.9  # Required % of full ROM
    form_strictness: str = "medium"  # strict/medium/lenient form checking

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'InferenceConfig':
        """Load config from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('inference_params', {}))

    def validate(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

def verify_model_paths(paths: ModelPaths) -> None:
    """Verify all required model files exist."""
    for attr, path in vars(paths).items():
        if attr != 'configs_dir' and not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

def load_motionbert_model(model_base_dir: str, device: torch.device) -> DSTformer:
    """
    Load the base MotionBERT model.
    
    This model provides the foundational motion understanding capabilities,
    which can be used for general motion analysis and feature extraction.
    """
    try:
        # Fixed architecture parameters - these match the pretrained model
        model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            num_joints=17,
            maxlen=243
        ).to(device)
        
        weights_path = os.path.join(model_base_dir, "motionbert_pretrained.bin")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Failed to load MotionBERT model: {e}")
        raise

def load_pose_model(model_base_dir: str, device: torch.device, motionbert_model: Optional[DSTformer] = None) -> DSTformer:
    """
    Load the pose estimation model.
    
    This model specializes in 3D pose estimation from 2D inputs,
    useful for accurate exercise form analysis.
    """
    try:
        if motionbert_model is None:
            motionbert_model = load_motionbert_model(model_base_dir, device)
            
        # Load pose-specific weights
        weights_path = os.path.join(model_base_dir, "MB_ft_h36m.bin")
        pose_state = torch.load(weights_path, map_location=device)
        motionbert_model.load_state_dict(pose_state)
        motionbert_model.eval()
        
        return motionbert_model
    except Exception as e:
        logger.error(f"Failed to load Pose model: {e}")
        raise

def load_mesh_model(model_base_dir: str, device: torch.device) -> DSTformer:
    """
    Load the mesh generation model.
    
    This model creates detailed 3D mesh representations of the human body,
    useful for comprehensive movement visualization and analysis.
    """
    try:
        model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            num_joints=17,
            maxlen=243
        ).to(device)
        
        weights_path = os.path.join(model_base_dir, "MB_ft_pw3d.bin")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Failed to load Mesh model: {e}")
        raise

def load_all_models(model_base_dir: str, device: torch.device, config: Optional[InferenceConfig] = None) -> Dict:
    """
    Load all models with optional configuration parameters.
    
    Args:
        model_base_dir: Directory containing model weights
        device: Torch device to load models to
        config: Optional inference configuration for customizable parameters
    
    Returns:
        Dictionary containing all loaded models
    """
    try:
        paths = ModelPaths.from_base_dir(model_base_dir)
        verify_model_paths(paths)
        
        if config is None:
            config = InferenceConfig()
        config.validate()
        
        motionbert_model = load_motionbert_model(model_base_dir, device)
        pose_model = load_pose_model(model_base_dir, device, motionbert_model)
        mesh_model = load_mesh_model(model_base_dir, device)
        
        return {
            'motionbert': motionbert_model,
            'pose': pose_model,
            'mesh': mesh_model,
            'config': config
        }
    except Exception as e:
        logger.error(f"Failed to load all models: {e}")
        raise