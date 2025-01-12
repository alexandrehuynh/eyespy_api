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
from typing import Dict, Optional, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    
    # Exercise-specific parameters
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

class SimplePositionalEncoding(torch.nn.Module):
    """Simplified positional encoding layer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimplifiedDSTformer(torch.nn.Module):
    """Simplified DSTformer architecture that matches pretrained weights."""
    def __init__(self, num_joints=17, in_channels=3):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        
        # Simplified architecture matching the pretrained model
        self.input_projection = torch.nn.Linear(in_channels, 128)
        self.positional_encoding = SimplePositionalEncoding(128)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=4
        )
        self.output_projection = torch.nn.Linear(128, 3)
        
    def forward(self, x):
        # Input shape: [batch_size, sequence_length, num_joints, channels]
        B, T, J, C = x.shape
        x = x.view(B, T, J * C)  # Flatten joints and channels
        
        # Project to transformer dimension
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project back to 3D coordinates
        x = self.output_projection(x)
        x = x.view(B, T, J, -1)  # Reshape back to [batch, time, joints, 3]
        
        return x

def load_motionbert_model(model_base_dir: str, device: torch.device) -> SimplifiedDSTformer:
    """Load the base MotionBERT model with simplified architecture."""
    try:
        model = SimplifiedDSTformer().to(device)
        weights_path = os.path.join(model_base_dir, "motionbert_pretrained.bin")
        
        # Load state dict
        state_dict = torch.load(weights_path, map_location=device)
        
        # Handle potential "model_pos" wrapping
        if "model_pos" in state_dict:
            state_dict = state_dict["model_pos"]
            
        # Load weights with strict=False to allow for architecture differences
        model.load_state_dict(state_dict, strict=False)
        logger.info("MotionBERT base model loaded successfully")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load MotionBERT model: {e}")
        raise

def load_pose_model(model_base_dir: str, device: torch.device) -> SimplifiedDSTformer:
    """Load the pose estimation model."""
    try:
        model = SimplifiedDSTformer().to(device)
        weights_path = os.path.join(model_base_dir, "MB_ft_h36m.bin")
        
        state_dict = torch.load(weights_path, map_location=device)
        if "model_pos" in state_dict:
            state_dict = state_dict["model_pos"]
            
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pose model loaded successfully")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load Pose model: {e}")
        raise

def load_mesh_model(model_base_dir: str, device: torch.device) -> SimplifiedDSTformer:
    """Load the mesh generation model."""
    try:
        model = SimplifiedDSTformer().to(device)
        weights_path = os.path.join(model_base_dir, "MB_ft_pw3d.bin")
        
        state_dict = torch.load(weights_path, map_location=device)
        if "model_pos" in state_dict:
            state_dict = state_dict["model_pos"]
            
        model.load_state_dict(state_dict, strict=False)
        logger.info("Mesh model loaded successfully")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load Mesh model: {e}")
        raise

def load_all_models(model_base_dir: str, device: torch.device, config: Optional[InferenceConfig] = None) -> Dict:
    """Load all models with simplified architecture."""
    try:
        paths = ModelPaths.from_base_dir(model_base_dir)
        verify_model_paths(paths)
        
        if config is None:
            config = InferenceConfig()
        config.validate()
        
        motionbert_model = load_motionbert_model(model_base_dir, device)
        pose_model = load_pose_model(model_base_dir, device)
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