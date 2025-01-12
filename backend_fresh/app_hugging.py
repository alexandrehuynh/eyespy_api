import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
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
from transformers import AutoModel, AutoConfig
import torch
from model_architectures import MotionBERTModel, Pose3DModel, ActionRecognitionModel, MeshModel


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
    'SMOOTHING_WINDOW': 5,
}

# Define custom pose connections (excluding face landmarks)
CUSTOM_POSE_CONNECTIONS = set([
    (start, end) for start, end in POSE_CONNECTIONS
    if 11 <= start < 33 and 11 <= end < 33  # Only include body landmarks (11-32)
])

# Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"
MOTIONBERT_FOLDER = BASE_DIR / PROCESSED_FOLDER / "3d_mesh"
META_FOLDER = PROCESSED_FOLDER / "meta"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, META_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseProcessor:
    def __init__(self):
        self.kalman_filters = self._init_kalman_filters()
        self.smoothing_history = defaultdict(lambda: deque(maxlen=CONFIG['SMOOTHING_WINDOW']))
        self.motion_trails = defaultdict(lambda: deque(maxlen=CONFIG['MOTION_TRAIL_LENGTH']))
        self.motion_history = defaultdict(list)
        self.previous_landmarks = None


        # Load MotionBERT and related models
        model_base_dir = os.path.dirname(os.path.abspath(__file__)) + "/models/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load MotionBERT and related models
        model_base_dir = os.path.dirname(os.path.abspath(__file__)) + "/models/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load MotionBERT
            motionbert_folder = os.path.join(model_base_dir, "motionbert")
            self.motionbert_config = AutoConfig.from_pretrained(motionbert_folder)
            self.motionbert_model = MotionBERTModel.from_pretrained(
                motionbert_folder,
                config=self.motionbert_config,
            ).to(self.device)

            # Load 3D Pose model
            self.pose_model = Pose3DModel()
            pose_state_dict = torch.load(
                os.path.join(model_base_dir, "pose/MB_ft_h36m.bin"),
                map_location=self.device
            )
            self.pose_model.load_state_dict(pose_state_dict)
            self.pose_model.to(self.device)

            # Load Action Recognition models
            self.action_xsub_model = ActionRecognitionModel()
            self.action_xview_model = ActionRecognitionModel()
            
            action_xsub_state_dict = torch.load(
                os.path.join(model_base_dir, "action/sub/MB_ft_NTU60_xsub.bin"),
                map_location=self.device
            )
            action_xview_state_dict = torch.load(
                os.path.join(model_base_dir, "action/view/MB_ft_NTU60_xview.bin"),
                map_location=self.device
            )
            
            self.action_xsub_model.load_state_dict(action_xsub_state_dict)
            self.action_xview_model.load_state_dict(action_xview_state_dict)
            
            self.action_xsub_model.to(self.device)
            self.action_xview_model.to(self.device)

            # Load Mesh model
            self.mesh_model = MeshModel()
            mesh_state_dict = torch.load(
                os.path.join(model_base_dir, "mesh/MB_ft_pw3d.bin"),
                map_location=self.device
            )
            self.mesh_model.load_state_dict(mesh_state_dict)
            self.mesh_model.to(self.device)

            # Set all models to evaluation mode
            self.motionbert_model.eval()
            self.pose_model.eval()
            self.action_xsub_model.eval()
            self.action_xview_model.eval()
            self.mesh_model.eval()

            print("All models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _init_kalman_filters(self):
        """Initialize Kalman filters for each landmark"""
        filters = {}
        for i in range(33):
            kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x, dx, ddx, y, dy, ddy], Measurement: [x, y, z]

            # State transition matrix (6x6)
            dt = 1.0  # time step
            kf.F = np.array([
                [1, dt, 0.5*dt**2, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5*dt**2],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1]
            ])

            # Measurement matrix (3x6)
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            # Measurement noise covariance (3x3)
            kf.R = np.eye(3) * 0.005  # Reduced measurement noise

            # Process noise covariance (6x6)
            q = 0.05  # process noise
            kf.Q = np.eye(6) * q   # Adjusted process noise

            # Initial state covariance (6x6)
            kf.P = np.eye(6) * 50 # Initial state uncertainty

            # Initial state (6x1)
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
        
    def _add_3d_mesh(self, landmarks):
        """Generate a 3D mesh from landmarks."""
        if not landmarks:
            raise ValueError("Landmarks are required to generate 3D mesh.")

        try:
            # Convert landmarks to the correct format
            landmark_tensor = torch.tensor(
                [[lm[:3] for lm in landmarks]], 
                dtype=torch.float32,
                device=self.device
            )
            
            # Process through mesh model
            with torch.no_grad():
                mesh_output = self.mesh_model(landmark_tensor)
            
            # Convert output to numpy for visualization
            mesh_result = mesh_output.cpu().numpy()
            
            # Create visualization frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Add your mesh visualization code here
            # This will depend on the output format of your mesh model
            # Example placeholder visualization:
            for vertex in mesh_result[0]:  # Assuming mesh_result[0] contains vertices
                x, y = int(vertex[0] * 640 + 640), int(vertex[1] * 360 + 360)
                if 0 <= x < 1280 and 0 <= y < 720:
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating 3D mesh: {str(e)}")
            # Return a blank frame in case of error
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def process_with_motionbert(self, landmarks):
        """Process landmarks with MotionBERT for action recognition, 3D pose, and mesh."""
        if not landmarks:
            return None

        # Convert landmarks to MotionBERT input format
        sequence = []
        for lm in landmarks:
            sequence.extend(lm[:3])  # Use x, y, z coordinates

        # Create input tensor and ensure it's the correct data type
        input_tensor = torch.tensor([sequence], dtype=torch.float32)
        input_tensor = input_tensor.long()

        try:
            # Action Recognition
            motionbert_output = self.motionbert_model(input_tensor)
            action_logits = motionbert_output.last_hidden_state.detach().numpy()
            recognized_action = int(np.argmax(action_logits))

            # 3D Pose Processing
            pose_3d = self.pose_model(input_tensor)
            pose_3d_result = pose_3d.detach().numpy()

            # Mesh Reconstruction using add_3d_mesh
            mesh_result = self._add_3d_mesh(landmarks)

            return {
                "action": recognized_action,
                "pose_3d": pose_3d_result.tolist(),
                "mesh": mesh_result.tolist()
            }

        except Exception as e:
            logger.error(f"Error during MotionBERT processing: {str(e)}")
            return {
                "error": f"MotionBERT processing failed: {str(e)}"
            }

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

        # MotionBERT processing
        motionbert_results = self.process_with_motionbert(smoothed_landmarks)
    
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
            'timestamp': datetime.now().timestamp(),
            'motionbert_results': motionbert_results
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
        

    def generate_3d_mesh_frame(landmarks, motionbert_results):
        """Generate a frame visualizing the 3D mesh and action recognition."""
        # Create a blank frame for visualization
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Adjust resolution as needed

        # Draw 3D pose (example visualization)
        for lm in landmarks:
            x, y, z = lm[:3]
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Example: Draw joints

        return frame
        
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

def setup_paths(file: UploadFile):
    """Setup input, output and metadata paths for video processing"""
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    input_path = UPLOAD_FOLDER / file.filename
    output_filename = f"processed_{timestamp}_{file.filename}"
    output_path = PROCESSED_FOLDER / output_filename

    # 3D mesh video path
    motionbert_folder = MOTIONBERT_FOLDER
    motionbert_folder.mkdir(parents=True, exist_ok=True)
    motionbert_output_path = motionbert_folder / f"mesh_{timestamp}_{file.filename}"

    # Metadata path
    metadata_path = META_FOLDER / f"{timestamp}_metadata.json"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(file.file.read())

    return input_path, output_path, motionbert_output_path, metadata_path

def save_metadata(metadata_path: Path, metadata: dict):
    """Save metadata to JSON file"""
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
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
    try:
        # Validate and setup
        input_path, output_path, motionbert_output_path, metadata_path = setup_paths(file)

        # Initialize processors
        pose_processor = PoseProcessor()
        visual_effects = VisualEffects()

        # Process video
        metadata = process_video_frames(
            input_path,
            output_path,
            motionbert_output_path,
            pose_processor,
            visual_effects
        )

        # Save metadata
        save_metadata(metadata_path, metadata)

        return {
            "mediapipe_video_url": FileResponse(
                output_path,
                media_type="video/mp4",
                filename=output_path.name
            ),
            "motionbert_video_url": str(motionbert_output_path),
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

def process_video_frames(input_path, output_path, motionbert_output_path, pose_processor, visual_effects):
    """Main video processing function"""
    metadata = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "frames": []
    }

    with mp_pose.Pose(
        min_detection_confidence=CONFIG['MIN_DETECTION_CONFIDENCE'],
        min_tracking_confidence=CONFIG['MIN_TRACKING_CONFIDENCE']
    ) as pose:
        cap = cv2.VideoCapture(str(input_path))
        out = None
        out_motionbert = None
        frame_count = 0

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Processing progress: {frame_count}/{total_frames}")

                # Process frame with Mediapipe
                processed_frame, frame_data = pose_processor.process_frame(
                    frame,
                    pose
                )

                if frame_data:
                    # Apply visual effects to Mediapipe video
                    visual_effects.draw_skeleton(
                        processed_frame,
                        frame_data['landmarks'],
                        CUSTOM_POSE_CONNECTIONS
                    )
                    visual_effects.draw_angles(
                        processed_frame,
                        frame_data['landmarks'],
                        frame_data['angles']
                    )
                    visual_effects.draw_pelvis_origin(
                        processed_frame,
                        frame_data['landmarks'],
                        mp_pose
                    )
                    visual_effects.draw_spine_overlay(
                        processed_frame,
                        frame_data['landmarks'],
                        mp_pose
                    )

                    # Update metadata
                    metadata["frames"].append({
                        "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                        **frame_data
                    })

                    # Generate 3D mesh frame
                    mesh_frame = pose_processor._add_3d_mesh(
                        frame_data['landmarks'],
                    )

                # Initialize video writer if needed
                if out is None:
                    out = initialize_video_writer(
                        output_path,
                        cap,
                        processed_frame.shape
                    )

                if out_motionbert is None:
                    out_motionbert = initialize_video_writer(
                        motionbert_output_path,
                        cap,
                        mesh_frame.shape
                    )
                

                # Write frames to respective videos
                out.write(processed_frame)
                out_motionbert.write(mesh_frame)

        finally:
            cleanup_resources(cap, out)
            cleanup_resources(None, out_motionbert)  # Close second writer if open


    return metadata