import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import sys
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from fastapi.responses import FileResponse
from datetime import datetime
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import json
from filterpy.kalman import KalmanFilter
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
import traceback
from motionbert_models import load_motionbert_model, load_pose_model, load_mesh_model

# Add MotionBERT/lib to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
motionbert_path = os.path.join(script_dir, "MotionBERT", "lib")
sys.path.append(motionbert_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
CONFIG = {
    'MIN_DETECTION_CONFIDENCE': 0.7,
    'MIN_TRACKING_CONFIDENCE': 0.7,
    'SMOOTHING_WINDOW': 5,
    'MOTION_TRAIL_LENGTH': 10,
    'AXIS_SCALE': 0.2,
    'AXIS_OPACITY': 0.5,
    'SKELETON_SAVE_INTERVAL': 30,
}

# Define custom pose connections (excluding face landmarks)
CUSTOM_POSE_CONNECTIONS = set([
    (start, end) for start, end in POSE_CONNECTIONS
    if 11 <= start < 33 and 11 <= end < 33  # Only include body landmarks (11-32)
])

# Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed" / "app_hug"
WIREFRAME_FOLDER = BASE_DIR / PROCESSED_FOLDER / "wireframe"
MESH_FOLDER = BASE_DIR / PROCESSED_FOLDER / "3d_mesh"
META_FOLDER = PROCESSED_FOLDER / PROCESSED_FOLDER / "meta"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, META_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseProcessor:
    def __init__(self, model_base_dir, device):
        self.kalman_filters = self._init_kalman_filters()
        self.smoothing_history = defaultdict(lambda: deque(maxlen=CONFIG['SMOOTHING_WINDOW']))
        self.motion_trails = defaultdict(lambda: deque(maxlen=CONFIG['MOTION_TRAIL_LENGTH']))
        self.motion_history = defaultdict(list)
        self.previous_landmarks = None
        self.device = device

        # Load models
        try:
            self.motionbert_model = load_motionbert_model(model_base_dir, device)
            self.pose_model = load_pose_model(model_base_dir, device)
            self.mesh_model = load_mesh_model(model_base_dir, device)
            logger.info("All models loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _init_kalman_filters(self):
        """Initialize Kalman filters for each landmark"""
        filters = {}
        for i in range(33):
            kf = KalmanFilter(dim_x=6, dim_z=3)
            dt = 1.0
            kf.F = np.array([
                [1, dt, 0.5*dt**2, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5*dt**2],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1]
            ])
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            kf.R = np.eye(3) * 0.005
            kf.Q = np.eye(6) * 0.05
            kf.P = np.eye(6) * 50
            kf.x = np.zeros((6, 1))
            filters[i] = kf
        return filters

    def _apply_kalman_filter(self, landmark, index):
        """Apply Kalman filtering to landmark"""
        try:
            # Convert landmark attributes to a measurement vector (3,)
            measurement = np.array([[landmark.x],
                                [landmark.y],
                                [landmark.z]])  # Shape: (3,1)

            kf = self.kalman_filters[index]

            # Predict next state
            kf.predict()

            # Update with measurement
            kf.update(measurement)

            # Return filtered state (positions only: x, y, z)
            filtered_state = np.array([kf.x[0][0], kf.x[3][0], kf.x[5][0]])  # Extract x, y, z positions
            return filtered_state

        except Exception as e:
            logger.error(f"Kalman filter error for landmark {index}: {e}")
            # Fallback to raw values if filtering fails
            return np.array([landmark.x, landmark.y, landmark.z])

    def _calculate_angles(self, landmarks):
        """Calculate joint angles"""
        angles = {}

        # Calculate angle between three points
        def calculate_angle(p1, p2, p3):
            if not all(p is not None for p in [p1, p2, p3]):
                return None

            vector1 = np.array(p1) - np.array(p2)
            vector2 = np.array(p3) - np.array(p2)

            cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)

        # Elbows
        left_elbow = calculate_angle(
            landmarks[11][:3],  # Left shoulder
            landmarks[13][:3],  # Left elbow
            landmarks[15][:3]   # Left wrist
        )
        angles['left_elbow'] = (13, left_elbow)

        right_elbow = calculate_angle(
            landmarks[12][:3],  # Right shoulder
            landmarks[14][:3],  # Right elbow
            landmarks[16][:3]   # Right wrist
        )
        angles['right_elbow'] = (14, right_elbow)

        # Shoulders
        left_shoulder = calculate_angle(
            landmarks[13][:3],  # Left elbow
            landmarks[11][:3],  # Left shoulder
            landmarks[23][:3]   # Left hip
        )
        angles['left_shoulder'] = (11, left_shoulder)

        right_shoulder = calculate_angle(
            landmarks[14][:3],  # Right elbow
            landmarks[12][:3],  # Right shoulder
            landmarks[24][:3]   # Right hip
        )
        angles['right_shoulder'] = (12, right_shoulder)

        # Hips
        left_hip = calculate_angle(
            landmarks[11][:3],  # Left shoulder
            landmarks[23][:3],  # Left hip
            landmarks[25][:3]   # Left knee
        )
        angles['left_hip'] = (23, left_hip)

        right_hip = calculate_angle(
            landmarks[12][:3],  # Right shoulder
            landmarks[24][:3],  # Right hip
            landmarks[26][:3]   # Right knee
        )
        angles['right_hip'] = (24, right_hip)

        # Knees
        left_knee = calculate_angle(
            landmarks[23][:3],  # Left hip
            landmarks[25][:3],  # Left knee
            landmarks[27][:3]   # Left ankle
        )
        angles['left_knee'] = (25, left_knee)

        right_knee = calculate_angle(
            landmarks[24][:3],  # Right hip
            landmarks[26][:3],  # Right knee
            landmarks[28][:3]   # Right ankle
        )
        angles['right_knee'] = (26, right_knee)

        # # Optional: Calculate spine angle (mid-shoulders to mid-hips)
        # mid_shoulders = (np.array(landmarks[11][:3]) + np.array(landmarks[12][:3])) / 2
        # mid_hips = (np.array(landmarks[23][:3]) + np.array(landmarks[24][:3])) / 2
        # spine_top = np.array(landmarks[0][:3])  # Nose as reference point
        
        # spine_angle = calculate_angle(
        #     spine_top,
        #     mid_shoulders,
        #     mid_hips
        # )
        # angles['spine'] = (0, spine_angle)

        return angles

    def _calculate_velocities(self, landmarks):
        """Calculate velocities of key points"""
        velocities = {}

        if self.previous_landmarks is not None:
            for i, (curr, prev) in enumerate(zip(landmarks, self.previous_landmarks)):
                if curr is not None and prev is not None:
                    velocity = np.array(curr[:3]) - np.array(prev[:3])
                    velocities[i] = np.linalg.norm(velocity)

        return velocities

    def _update_motion_history(self, landmarks):
        """Update motion history for each landmark"""
        timestamp = datetime.now().timestamp()

        for i, landmark in enumerate(landmarks):
            if landmark is not None:
                self.motion_history[i].append({
                    'position': landmark[:3],
                    'timestamp': timestamp
                })

                # Keep only recent history
                while len(self.motion_history[i]) > 100:  # Adjust history length as needed
                    self.motion_history[i].pop(0)

    def _apply_visual_effects(self, frame, landmarks, angles, velocities):
        """Apply visual effects to the frame using CUSTOM_POSE_CONNECTIONS."""
        # Use CUSTOM_POSE_CONNECTIONS to draw motion trails
        for start_idx, end_idx in CUSTOM_POSE_CONNECTIONS:
            # Ensure valid landmarks
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx] is not None and landmarks[end_idx] is not None):

                # Get the start and end points
                start_point = tuple(map(int, landmarks[start_idx][:2]))
                end_point = tuple(map(int, landmarks[end_idx][:2]))

                # Draw motion trails for the connection
                self.motion_trails[start_idx].append(start_point)
                self.motion_trails[end_idx].append(end_point)

                # Draw the motion trail for the connection
                points = list(self.motion_trails[start_idx])
                if len(points) > 1:
                    for j in range(1, len(points)):
                        alpha = j / len(points)
                        cv2.line(frame, points[j - 1], points[j],
                                (0, int(255 * alpha), int(255 * (1 - alpha))), 2)

        # Add velocity indicators for landmarks in CUSTOM_POSE_CONNECTIONS
        for start_idx, end_idx in CUSTOM_POSE_CONNECTIONS:
            if start_idx in velocities and landmarks[start_idx] is not None:
                velocity = velocities[start_idx]
                if velocity > 5:  # Threshold for significant movement
                    pos = tuple(map(int, landmarks[start_idx][:2]))
                    cv2.circle(frame, pos, int(velocity), (0, 255, 255), 1)

        return frame

    def process_frame(self, frame, pose):
        """Process a single frame"""
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get pose landmarks
        results = pose.process(image)
        
        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not results.pose_landmarks:
            return frame, None

        height, width, _ = frame.shape

        # Process landmarks
        smoothed_landmarks = self._process_landmarks(
            results.pose_landmarks.landmark,
            width,
            height
        )

        # Calculate angles and velocities
        angles = self._calculate_angles(smoothed_landmarks)
        velocities = self._calculate_velocities(smoothed_landmarks)

        # Update motion history
        self._update_motion_history(smoothed_landmarks)

        # Apply visual effects
        frame = self._apply_visual_effects(frame, smoothed_landmarks, angles, velocities)

        # Update previous landmarks
        self.previous_landmarks = smoothed_landmarks.copy()

        return frame, {
            'landmarks': smoothed_landmarks, 
            'angles': angles,
            'velocities': velocities,
            'timestamp': datetime.now().timestamp()
        }

    def _process_landmarks(self, landmarks, width, height):
        """Process and smooth landmarks"""
        processed_landmarks = []
        for i, lm in enumerate(landmarks):
            if not isinstance(lm, NormalizedLandmark):
                logger.warning(f"Invalid landmark detected at index {i}")
                continue

            try:
                # Apply Kalman filtering
                filtered = self._apply_kalman_filter(lm, i)

                # Convert to screen coordinates
                processed = [
                    filtered[0] * width,   # x coordinate
                    filtered[1] * height,  # y coordinate
                    filtered[2] * width,   # z coordinate
                    lm.visibility
                ]

                # Apply smoothing
                smoothed = self._smooth_landmark(processed, i)
                processed_landmarks.append(smoothed)

            except Exception as e:
                logger.error(f"Error processing landmark {i}: {e}")
                # Fallback to raw values if processing fails
                processed_landmarks.append([
                    lm.x * width,
                    lm.y * height,
                    lm.z * width,
                    lm.visibility
                ])

        return processed_landmarks
    
    def _smooth_landmark(self, landmark, index):
        """
        Apply temporal smoothing to landmark coordinates using a moving average.
        
        Args:
            landmark: List containing [x, y, z, visibility] coordinates
            index: Landmark index
            
        Returns:
            Smoothed landmark coordinates
        """
        try:
            # Add current landmark to history
            self.smoothing_history[index].append(landmark)
            
            # Calculate smoothed coordinates
            if len(self.smoothing_history[index]) > 0:
                # Separate x, y, z coordinates and visibility
                coords = np.array([l[:3] for l in self.smoothing_history[index]])
                visibility = np.array([l[3] for l in self.smoothing_history[index]])
                
                # Calculate weighted average based on visibility
                weights = visibility / np.sum(visibility)
                smoothed_coords = np.average(coords, axis=0, weights=weights)
                
                # Use most recent visibility
                smoothed_visibility = landmark[3]
                
                return [*smoothed_coords, smoothed_visibility]
            
            return landmark
            
        except Exception as e:
            logger.error(f"Error smoothing landmark {index}: {str(e)}")
            return landmark

class VisualEffects:
    """Handle all visual effects and drawing operations"""

    @staticmethod
    def draw_skeleton(frame, landmarks, connections):
        """Draw the skeleton with custom styling"""
        # Draw connections in white
        for start_idx, end_idx in connections:
            start_point = tuple(map(int, landmarks[start_idx][:2]))
            end_point = tuple(map(int, landmarks[end_idx][:2]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2) 

        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            # Only draw landmarks for body (indices 11-32)
            if 11 <= i < 33:
                point = tuple(map(int, landmark[:2]))
                cv2.circle(frame, point, 4, (255, 255, 255), -1)  
                cv2.circle(frame, point, 2, (128, 128, 128), -1)  # Inner circle slightly darker

    @staticmethod
    def draw_angles(frame, landmarks, angles):
        """Draw angle measurements on the frame"""
        for joint_name, (landmark_idx, angle) in angles.items():
            if angle is not None:
                position = landmarks[landmark_idx]
                cv2.putText(
                    frame,
                    f"{joint_name}: {int(angle)}",
                    (int(position[0] + 20), int(position[1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if angle < 90 else (0, 0, 255),
                    2
                )

    @staticmethod
    def draw_pelvis_origin(frame, landmarks, mp_pose):
        """Draw pelvis origin and 3D axes"""
        height, width, _ = frame.shape

        # Calculate pelvis origin
        left_hip = np.array(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][:2])
        right_hip = np.array(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])
        pelvis_origin = tuple(map(int, (left_hip + right_hip) / 2))

        # Create overlay for axes
        overlay = frame.copy()

        # Draw extended 3D axes
        axis_length = int(min(width, height) * 0.2)
        cv2.line(overlay, pelvis_origin,
                (pelvis_origin[0] + axis_length, pelvis_origin[1]), (0, 0, 255), 3)  # X-axis
        cv2.line(overlay, pelvis_origin,
                (pelvis_origin[0], pelvis_origin[1] - axis_length), (0, 255, 0), 3)  # Y-axis
        cv2.line(overlay, pelvis_origin,
                (pelvis_origin[0], pelvis_origin[1] + axis_length), (255, 0, 0), 3)  # Z-axis

        # Add axis labels
        cv2.putText(overlay, "X", (pelvis_origin[0] + axis_length + 10, pelvis_origin[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(overlay, "Y", (pelvis_origin[0], pelvis_origin[1] - axis_length - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(overlay, "Z", (pelvis_origin[0], pelvis_origin[1] + axis_length + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Apply semi-transparent overlay
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    @staticmethod
    def draw_spine_overlay(frame, landmarks, mp_pose):
        """Draw spine overlay with midpoints"""
        # Calculate shoulder and hip midpoints
        left_shoulder = np.array(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:2])
        right_shoulder = np.array(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:2])
        left_hip = np.array(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][:2])
        right_hip = np.array(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])

        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        hip_midpoint = (left_hip + right_hip) / 2

        # Generate spine midpoints
        spine_points = []
        num_points = 5
        for i in range(num_points + 1):
            t = i / num_points
            point = tuple(map(int, (1 - t) * shoulder_midpoint + t * hip_midpoint))
            spine_points.append(point)

        # Draw spine segments
        for i in range(len(spine_points) - 1):
            cv2.line(frame, spine_points[i], spine_points[i + 1], (0, 255, 0), 2)
            cv2.circle(frame, spine_points[i], 3, (255, 255, 255), -1)
        cv2.circle(frame, spine_points[-1], 3, (255, 255, 255), -1)

class MotionBERTProcessor:
    def __init__(self, model_base_dir, device):
        self.device = device
        
        try:
            self.pose_model = load_pose_model(model_base_dir, device)
            self.mesh_model = load_mesh_model(model_base_dir, device)
            logger.info("MotionBERT models loaded successfully!")
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Fixed input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        except Exception as e:
            logger.error(f"Failed to load MotionBERT models: {e}")
            raise

    def process_frame(self, frame):
        """Process a single frame to get 3D pose and mesh features"""
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(frame_pil)
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Get pose prediction
            with torch.no_grad():
                pose_3d = self.pose_model(frame_tensor)  # Shape: [1, 1, 17, 3]
                logger.info(f"Pose3D output shape: {pose_3d.shape}")
                
                # Get mesh features if needed
                mesh_features = self.mesh_model(frame_tensor) if self.mesh_model else None
            
            return {
                'pose_3d': pose_3d[0, 0].cpu().numpy(),  # Remove batch and sequence dimensions
                'mesh_features': mesh_features[0, 0].cpu().numpy() if mesh_features is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error in MotionBERT processing: {e}")
            traceback.print_exc()
            return None

    def generate_wireframe(self, pose_3d):
        """Generate 3D wireframe visualization using pose data"""
        if pose_3d is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        try:
            # Create visualization frame with dark background
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 32
            
            # Define H36M skeleton connections
            h36m_edges = [
                (0, 1), (1, 2), (2, 3),  # Right leg
                (0, 4), (4, 5), (5, 6),  # Left leg
                (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
                (8, 11), (11, 12), (12, 13),  # Right arm
                (8, 14), (14, 15), (15, 16)  # Left arm
            ]
            
            # Project 3D points to 2D
            focal_length = 1000
            center = np.array([640, 360])
            
            points_2d = []
            for point in pose_3d:
                x = point[0] * focal_length / (point[2] + 1e-8) + center[0]
                y = point[1] * focal_length / (point[2] + 1e-8) + center[1]
                points_2d.append((int(x), int(y)))
            
            # Draw skeleton connections
            for i, (start_idx, end_idx) in enumerate(h36m_edges):
                start_point = points_2d[start_idx]
                end_point = points_2d[end_idx]
                
                # Color coding based on body part
                if i < 3:  # Right leg
                    color = (255, 0, 0)
                elif i < 6:  # Left leg
                    color = (0, 255, 0)
                elif i < 10:  # Spine and head
                    color = (0, 0, 255)
                else:  # Arms
                    color = (255, 255, 0)
                
                cv2.line(frame, start_point, end_point, color, 2, cv2.LINE_AA)
                
            # Draw joints
            for point in points_2d:
                cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating wireframe: {str(e)}")
            traceback.print_exc()
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
    def generate_mesh(self, mesh_features):
        """Generate 3D mesh visualization using mesh features"""
        if mesh_features is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        try:
            # Create visualization frame
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 32
            
            # Implement mesh visualization here
            # This will depend on the specific mesh format and visualization needs
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating mesh: {str(e)}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

def setup_paths(file: UploadFile):
    """Setup input, output and metadata paths for video processing"""
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    input_path = UPLOAD_FOLDER / file.filename
    
    # Setup output paths for each video type
    mediapipe_filename = f"mediapipe_{timestamp}_{file.filename}"
    wireframe_filename = f"wireframe_{timestamp}_{file.filename}"
    mesh_filename = f"mesh_{timestamp}_{file.filename}"
    
    mediapipe_path = PROCESSED_FOLDER / mediapipe_filename
    wireframe_path = WIREFRAME_FOLDER / wireframe_filename
    mesh_path = MESH_FOLDER / mesh_filename

    # Ensure all output directories exist
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    WIREFRAME_FOLDER.mkdir(parents=True, exist_ok=True)
    MESH_FOLDER.mkdir(parents=True, exist_ok=True)
    META_FOLDER.mkdir(parents=True, exist_ok=True)

    # Metadata path
    metadata_path = META_FOLDER / f"{timestamp}_metadata.json"

    try:
        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    return input_path, mediapipe_path, wireframe_path, mesh_path, metadata_path

def save_metadata(metadata_path: Path, metadata: dict):
    """Save metadata to JSON file with numpy array handling"""
    try:
        # Convert numpy arrays to lists in metadata
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]  # FIXED: using 'obj' instead of 'list'
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)  # Added fallback for other types
        
        processed_metadata = convert_numpy(metadata)
        
        with open(metadata_path, "w") as f:
            json.dump(processed_metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        raise

def initialize_video_writer(output_path: Path, cap, frame_shape):
    """Initialize OpenCV VideoWriter"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(frame_shape[1])
    height = int(frame_shape[0])

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to initialize video writer")

    return writer

def cleanup_resources(cap, writer):
    """Clean up video capture and writer resources"""
    try:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    """Process video and generate three output videos: MediaPipe skeleton, 3D wireframe, and 3D mesh"""
    try:
        # Validate and setup paths for all three videos
        input_path, mediapipe_path, wireframe_path, mesh_path, metadata_path = setup_paths(file)

        # Initialize processors
        model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motionbert", "params")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_processor = PoseProcessor(model_base_dir, device)
        visual_effects = VisualEffects()
        motionbert_processor = MotionBERTProcessor(model_base_dir, device)  # New MotionBERT processor


        # Process video and generate all three outputs
        metadata = process_video_frames(
            input_path,
            mediapipe_path,
            wireframe_path,
            mesh_path,
            pose_processor,
            visual_effects,
            motionbert_processor
        )

        # Save metadata
        save_metadata(metadata_path, metadata)

        return {
            "status": "success",
            "videos": {
                "mediapipe": FileResponse(
                    mediapipe_path,
                    media_type="video/mp4",
                    filename=mediapipe_path.name
                ),
                "wireframe": FileResponse(
                    wireframe_path,
                    media_type="video/mp4",
                    filename=wireframe_path.name
                ),
                "mesh": FileResponse(
                    mesh_path,
                    media_type="video/mp4",
                    filename=mesh_path.name
                )
            },
            "metadata": str(metadata_path)
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

    finally:
        # Cleanup temporary files if needed
        try:
            if input_path.exists():
                input_path.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

def process_video_frames(input_path, mediapipe_path, wireframe_path, mesh_path, pose_processor, visual_effects, motionbert_processor):
    """Main video processing function for generating three output videos"""
    metadata = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "frames": []
    }

    cap = None
    writers = {
        'mediapipe': None,
        'wireframe': None,
        'mesh': None
    }

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        # Create blank frames for cases where processing fails
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with mp_pose.Pose(
            min_detection_confidence=CONFIG['MIN_DETECTION_CONFIDENCE'],
            min_tracking_confidence=CONFIG['MIN_TRACKING_CONFIDENCE']
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Processing frame {frame_count}/{total_frames}")

                # Initialize frame variables
                mediapipe_frame = blank_frame.copy()
                wireframe_frame = blank_frame.copy()
                mesh_frame = blank_frame.copy()

                # Process frame with Mediapipe
                processed_frame, frame_data = pose_processor.process_frame(frame, pose)

                if frame_data:
                    # Generate MediaPipe visualization
                    mediapipe_frame = processed_frame.copy()
                    visual_effects.draw_skeleton(
                        mediapipe_frame,
                        frame_data['landmarks'],
                        CUSTOM_POSE_CONNECTIONS
                    )
                    visual_effects.draw_angles(
                        mediapipe_frame,
                        frame_data['landmarks'],
                        frame_data['angles']
                    )
                    visual_effects.draw_pelvis_origin(
                        mediapipe_frame,
                        frame_data['landmarks'],
                        mp_pose
                    )
                    visual_effects.draw_spine_overlay(
                        mediapipe_frame,
                        frame_data['landmarks'],
                        mp_pose
                    )

                    # Process with MotionBERT
                    motionbert_data = motionbert_processor.process_frame(frame)
                    if motionbert_data:
                        wireframe_frame = motionbert_processor.generate_wireframe(
                            motionbert_data['pose_3d']
                        )
                        mesh_frame = motionbert_processor.generate_mesh(
                            motionbert_data['mesh_features']
                        )

                # Initialize video writers if not already done
                if writers['mediapipe'] is None:
                    writers['mediapipe'] = initialize_video_writer(
                        mediapipe_path, cap, mediapipe_frame.shape
                    )
                if writers['wireframe'] is None:
                    writers['wireframe'] = initialize_video_writer(
                        wireframe_path, cap, wireframe_frame.shape
                    )
                if writers['mesh'] is None:
                    writers['mesh'] = initialize_video_writer(
                        mesh_path, cap, mesh_frame.shape
                    )

                # Write frames to respective videos
                writers['mediapipe'].write(mediapipe_frame)
                writers['wireframe'].write(wireframe_frame)
                writers['mesh'].write(mesh_frame)

                # Update metadata
                frame_metadata = {
                    "frame_number": frame_count,
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "mediapipe": {
                        "landmarks": frame_data.get('landmarks') if frame_data else None,
                        "angles": frame_data.get('angles') if frame_data else None,
                        "velocities": frame_data.get('velocities') if frame_data else None
                    },
                    "motionbert": motionbert_data if motionbert_data else None
                }
                metadata["frames"].append(frame_metadata)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        traceback.print_exc()
        raise

    finally:
        # Clean up resources
        if cap is not None:
            cap.release()
        
        for writer in writers.values():
            if writer is not None:
                writer.release()
        
        cv2.destroyAllWindows()

    return metadata