import os
import torch
import sys

# Add MotionBERT/lib to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
motionbert_path = os.path.join(script_dir, "MotionBERT", "lib")
sys.path.append(motionbert_path)
print("Python search path:", sys.path)

from model.DSTformer import DSTformer  # Backbone for MotionBERT
from model.model_action import ActionNet  # For action recognition

# Paths to the models in the params directory
motionbert_model_path = os.path.join(script_dir, "MotionBERT", "params", "motionbert_pretrained.bin")
action_xsub_model_path = os.path.join(script_dir, "MotionBERT", "params", "MB_ft_NTU60_xsub.bin")
mesh_model_path = os.path.join(script_dir, "MotionBERT", "params", "MB_ft_pw3d.bin")

# Utility to preprocess input
def preprocess_for_backbone(input_tensor):
    """Preprocess input for MotionBERT backbone."""
    return input_tensor.squeeze(1)  # Remove clips dimension

# Function to load a model
def load_model(model_path, model_class, model_kwargs=None):
    """
    Loads a PyTorch model from a given path.

    Args:
        model_path (str): Path to the model weights file.
        model_class (class): The class used to initialize the model.
        model_kwargs (dict, optional): Arguments for the model class.

    Returns:
        model: The loaded model instance.
    """
    if model_kwargs is None:
        model_kwargs = {}

    if os.path.exists(model_path):
        model = model_class(**model_kwargs)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded successfully from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

# Load models
motionbert_model = load_model(
    motionbert_model_path,
    DSTformer,
    {"dim_in": 3, "dim_out": 3, "dim_feat": 512, "dim_rep": 512, "depth": 5, "num_heads": 8, "mlp_ratio": 4, "num_joints": 17, "maxlen": 243}
)

action_model = load_model(
    action_xsub_model_path,
    ActionNet,
    {"backbone": motionbert_model, "dim_rep": 512, "num_classes": 60, "version": "class"}
)

mesh_model = load_model(
    mesh_model_path,
    DSTformer,
    {"dim_in": 3, "dim_out": 3, "dim_feat": 512, "dim_rep": 512, "depth": 5, "num_heads": 8, "mlp_ratio": 4, "num_joints": 17, "maxlen": 243}
)

# Dummy input for testing
dummy_input = torch.rand(1, 1, 243, 17, 3)  # Batch=1, Clips=1, Frames=243, Joints=17, Features=3

# Switch models to evaluation mode
motionbert_model.eval()
action_model.eval()
mesh_model.eval()

# Test models
try:
    motionbert_output = motionbert_model.get_representation(preprocess_for_backbone(dummy_input))
    print(f"MotionBERT output shape: {motionbert_output.shape} (expected: [1, 243, 17, 512])")
except Exception as e:
    print(f"Error testing MotionBERT: {e}")

try:
    action_output = action_model(dummy_input)
    print(f"Action recognition output shape: {action_output.shape} (expected: [1, 60])")
except Exception as e:
    print(f"Error testing action recognition: {e}")

try:
    mesh_output = mesh_model.get_representation(preprocess_for_backbone(dummy_input))
    print(f"3D mesh output shape: {mesh_output.shape} (expected: [1, 243, 17, 512])")
except Exception as e:
    print(f"Error testing 3D mesh creation: {e}")