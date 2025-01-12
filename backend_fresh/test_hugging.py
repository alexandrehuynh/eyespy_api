import os
from transformers import AutoModel, AutoConfig

# Get the current directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths dynamically
motionbert_config_path = os.path.join(script_dir, "models/motionbert/config.json")
motionbert_model_path = os.path.join(script_dir, "models/motionbert/pytorch_model.bin")

pose_config_path = os.path.join(script_dir, "models/pose/config.json")
pose_model_path = os.path.join(script_dir, "models/pose/MB_ft_h36m.bin")

action_xsub_config_path = os.path.join(script_dir, "models/action/sub/config.json")
action_xsub_model_path = os.path.join(script_dir, "models/action/sub/MB_ft_NTU60_xsub.bin")

action_xview_config_path = os.path.join(script_dir, "models/action/view/config.json")
action_xview_model_path = os.path.join(script_dir, "models/action/view/MB_ft_NTU60_xview.bin")

mesh_config_path = os.path.join(script_dir, "models/mesh/config.json")
mesh_model_path = os.path.join(script_dir, "models/mesh/MB_ft_pw3d.bin")

# Load models as before
motionbert_config = AutoConfig.from_pretrained(motionbert_config_path)
motionbert_model = AutoModel.from_pretrained(motionbert_model_path, config=motionbert_config)

# Load the MotionBERT model with ignore_mismatched_sizes
motionbert_model = AutoModel.from_pretrained(
    motionbert_model_path,
    config=motionbert_config
)

print("MotionBERT model loaded successfully!")