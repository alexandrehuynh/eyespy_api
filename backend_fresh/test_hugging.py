import os
import torch
from lib.model.model_action import MotionBERT  # Adjust this if the class location differs

# Get the current directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the models in the params directory
motionbert_model_path = os.path.join(script_dir, "MotionBERT", "params", "motionbert_pretrained.bin")
pose_model_path = os.path.join(script_dir, "MotionBERT", "params", "MB_ft_h36m.bin")
action_xsub_model_path = os.path.join(script_dir, "MotionBERT", "params", "MB_ft_NTU60_xsub.bin")
action_xview_model_path = os.path.join(script_dir, "MotionBERT", "params", "MB_ft_NTU60_xview.bin")
mesh_model_path = os.path.join(script_dir, "MotionBERT", "params", "MB_ft_pw3d.bin")

# Initialize MotionBERT model
model = MotionBERT()

# Load the MotionBERT pretrained weights
if os.path.exists(motionbert_model_path):
    state_dict = torch.load(motionbert_model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading
    print("MotionBERT pretrained weights loaded successfully.")
else:
    print(f"Pretrained weights not found at {motionbert_model_path}. Using an untrained model.")

# Test the model with a dummy input (adjust the dimensions based on your use case)
dummy_input = torch.rand(1, 50, 17, 3)  # Example tensor: batch_size, frames, joints, channels
output = model(dummy_input)
print("Model output shape:", output.shape)