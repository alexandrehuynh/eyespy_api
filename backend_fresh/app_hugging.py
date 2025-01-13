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

    def _convert_to_motionbert_format(self, landmarks):
        """Convert MediaPipe landmarks to MotionBERT input format"""
        if not landmarks:
            return None

        try:
            # Initialize output array: [batch_size, frames, joints, coords]
            converted = np.zeros((1, 1, 17, 3))
            
            # MediaPipe to MotionBERT joint mapping
            # Only including the relevant body joints (not face)
            joint_mapping = {
                11: 0,  # LEFT_SHOULDER
                12: 1,  # RIGHT_SHOULDER
                13: 2,  # LEFT_ELBOW
                14: 3,  # RIGHT_ELBOW
                15: 4,  # LEFT_WRIST
                16: 5,  # RIGHT_WRIST
                23: 6,  # LEFT_HIP
                24: 7,  # RIGHT_HIP
                25: 8,  # LEFT_KNEE
                26: 9,  # RIGHT_KNEE
                27: 10, # LEFT_ANKLE
                28: 11, # RIGHT_ANKLE
                29: 12, # LEFT_HEEL
                30: 13, # RIGHT_HEEL
                31: 14, # LEFT_FOOT_INDEX
                32: 15, # RIGHT_FOOT_INDEX
                0: 16,  # NOSE
            }

            # Convert coordinates
            for mp_idx, mb_idx in joint_mapping.items():
                if mp_idx < len(landmarks) and landmarks[mp_idx] is not None:
                    # Extract coordinates and normalize
                    coords = np.array(landmarks[mp_idx][:3])  # x, y, z
                    
                    # Normalize coordinates to [-1, 1] range
                    coords[0] = (coords[0] - 640) / 640  # Assuming 1280x720 frame
                    coords[1] = (coords[1] - 360) / 360
                    coords[2] = coords[2] / 100  # Normalize depth
                    
                    converted[0, 0, mb_idx] = coords

            # Convert to tensor and ensure correct shape
            tensor_input = torch.tensor(converted, dtype=torch.float32, device=self.device)
            
            # Add extra dummy frame dimension if needed
            if len(tensor_input.shape) == 3:
                tensor_input = tensor_input.unsqueeze(1)

            # Debug output
            logger.info(f"Input tensor shape: {tensor_input.shape}")
            logger.info(f"Input tensor device: {tensor_input.device}")
            
            return tensor_input

        except Exception as e:
            logger.error(f"Error converting landmarks to MotionBERT format: {str(e)}")
            logger.error(f"Landmark shape: {np.array(landmarks).shape if landmarks else 'None'}")
            return None

    def process_with_motionbert(self, landmarks):
        """Process landmarks with MotionBERT for 3D pose estimation and mesh generation"""
        if not landmarks:
            return None

        try:
            # Convert landmarks to MotionBERT format
            motionbert_input = self._convert_to_motionbert_format(landmarks)
            
            with torch.no_grad():
                # Get 3D pose
                pose_3d = self.pose_model(motionbert_input)
                
                # Generate mesh
                mesh_features = self.mesh_model(motionbert_input)

            return {
                "pose_3d": pose_3d.cpu().numpy().tolist(),
                "mesh_features": mesh_features.cpu().numpy().tolist()
            }

        except Exception as e:
            logger.error(f"Error during MotionBERT processing: {str(e)}")
            return {"error": f"MotionBERT processing failed: {str(e)}"}

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

    def _generate_3d_wireframe(self, landmarks):
        """Generate a 3D wireframe visualization from landmarks"""
        if not landmarks:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        try:
            # Convert landmarks to MotionBERT format
            motionbert_input = self._convert_to_motionbert_format(landmarks)
            
            # Get 3D pose estimation
            with torch.no_grad():
                pose_3d = self.pose_model(motionbert_input)
                pose_3d = pose_3d.cpu().numpy()[0, 0]  # Shape: (17, 3)
            
            # Create visualization frame with gray background
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 32  # Dark gray background
            
            # Visualization parameters
            scale_factor = 1000  # Increased scale factor
            center_x, center_y = 640, 360
            
            # Normalize and center the 3D coordinates
            pose_mean = np.mean(pose_3d, axis=0)
            pose_centered = pose_3d - pose_mean
            
            # Project 3D coordinates to 2D
            projected_coords = []
            for joint in pose_centered:
                # Simple perspective projection
                z = joint[2] * scale_factor + 1000  # Add offset to avoid division by zero
                x = int(center_x + (joint[0] * scale_factor * 1000) / (z + 1e-4))
                y = int(center_y + (joint[1] * scale_factor * 1000) / (z + 1e-4))
                projected_coords.append((x, y, joint[2]))
            
            # Draw connections between joints with depth-based coloring
            for connection in CUSTOM_POSE_CONNECTIONS:
                start_idx = connection[0] - 11  # Adjust indices for body-only landmarks
                end_idx = connection[1] - 11
                
                if (0 <= start_idx < len(projected_coords) and 
                    0 <= end_idx < len(projected_coords)):
                    
                    start = projected_coords[start_idx]
                    end = projected_coords[end_idx]
                    
                    # Check if coordinates are within frame bounds
                    if (0 <= start[0] < 1280 and 0 <= start[1] < 720 and
                        0 <= end[0] < 1280 and 0 <= end[1] < 720):
                        
                        # Calculate color based on average depth
                        avg_depth = (start[2] + end[2]) / 2
                        depth_color = int(255 * (avg_depth + 1) / 2)
                        color = (0, int(255 * (1 - abs(avg_depth))), 255)  # Cyan to blue gradient
                        
                        cv2.line(frame, 
                                (int(start[0]), int(start[1])), 
                                (int(end[0]), int(end[1])), 
                                color, 2)
            
            # Draw joints with depth-based size and color
            for i, coord in enumerate(projected_coords):
                if 0 <= coord[0] < 1280 and 0 <= coord[1] < 720:
                    # Size based on depth (closer joints are larger)
                    size = int(6 * (1 + coord[2]))
                    # Color based on depth (closer joints are brighter)
                    brightness = int(255 * (1 - abs(coord[2])))
                    color = (0, brightness, 255)  # Cyan color scheme
                    
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), size, color, -1)
            
            # Add coordinate system
            axis_length = 100
            origin = (50, 670)
            cv2.line(frame, origin, (origin[0] + axis_length, origin[1]), (0, 0, 255), 2)  # X-axis
            cv2.line(frame, origin, (origin[0], origin[1] - axis_length), (0, 255, 0), 2)  # Y-axis
            cv2.circle(frame, origin, 5, (255, 255, 255), -1)  # Origin point
            
            # Add labels and information
            cv2.putText(frame, "3D Wireframe View", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating 3D wireframe: {str(e)}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def _generate_3d_mesh_frame(self, landmarks):
        """Generate a frame visualizing the 3D mesh representation."""
        if not landmarks:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        try:
            # Convert landmarks to MotionBERT format
            motionbert_input = self._convert_to_motionbert_format(landmarks)
            
            # Get mesh representation from model
            with torch.no_grad():
                mesh_output = self.mesh_model(motionbert_input)
                mesh_vertices = mesh_output.cpu().numpy()[0, 0]  # Shape: (N, 3)
            
            # Create visualization frame with dark background
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 32
            
            # Visualization parameters
            scale_factor = 1000
            center_x, center_y = 640, 360
            
            # Center and normalize vertices
            vertices_mean = np.mean(mesh_vertices, axis=0)
            vertices_centered = mesh_vertices - vertices_mean
            
            # Project vertices to 2D space
            projected_vertices = []
            for vertex in vertices_centered:
                # Perspective projection
                z = vertex[2] * scale_factor + 1000
                x = int(center_x + (vertex[0] * scale_factor * 1000) / (z + 1e-4))
                y = int(center_y + (vertex[1] * scale_factor * 1000) / (z + 1e-4))
                
                if 0 <= x < 1280 and 0 <= y < 720:  # Check bounds
                    projected_vertices.append((x, y, vertex[2]))

            # Define body segments for mesh
            body_segments = [
                # Torso
                ([11, 12, 23, 24], (0, 255, 255)),  # Chest
                ([23, 24, 25, 26], (0, 200, 255)),  # Abdomen
                # Arms
                ([11, 13, 15], (0, 150, 255)),  # Left arm
                ([12, 14, 16], (0, 150, 255)),  # Right arm
                # Legs
                ([23, 25, 27], (0, 100, 255)),  # Left leg
                ([24, 26, 28], (0, 100, 255)),  # Right leg
            ]

            # Draw mesh segments
            for segment_indices, base_color in body_segments:
                if all(i < len(projected_vertices) for i in segment_indices):
                    points = np.array([projected_vertices[i][:2] for i in segment_indices])
                    
                    # Create alpha mask for smooth rendering
                    mask = np.zeros((720, 1280), dtype=np.uint8)
                    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
                    
                    # Calculate depth-based color
                    avg_depth = np.mean([projected_vertices[i][2] for i in segment_indices])
                    color_factor = (1 - abs(avg_depth)) * 0.7 + 0.3
                    color = tuple(int(c * color_factor) for c in base_color)
                    
                    # Draw filled segment with transparency
                    colored_segment = np.zeros_like(frame)
                    cv2.fillPoly(colored_segment, [points.astype(np.int32)], color)
                    
                    # Blend with frame
                    alpha = 0.7
                    mask_3d = np.stack([mask/255.]*3, axis=2)
                    frame = np.uint8(frame * (1 - alpha * mask_3d) + colored_segment * (alpha * mask_3d))
            
            # Draw edges for definition
            for segment_indices, _ in body_segments:
                if all(i < len(projected_vertices) for i in segment_indices):
                    points = np.array([projected_vertices[i][:2] for i in segment_indices])
                    cv2.polylines(frame, [points.astype(np.int32)], True, (255, 255, 255), 1)

            # Add labels and information
            cv2.putText(frame, "3D Mesh Visualization", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating 3D mesh: {str(e)}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
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
    """Process video and generate three output videos: MediaPipe skeleton, 3D wireframe, and 3D mesh"""
    try:
        # Validate and setup paths for all three videos
        input_path, mediapipe_path, wireframe_path, mesh_path, metadata_path = setup_paths(file)

        # Initialize processors
        model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motionbert", "params")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_processor = PoseProcessor(model_base_dir, device)
        visual_effects = VisualEffects()

        # Process video and generate all three outputs
        metadata = process_video_frames(
            input_path,
            mediapipe_path,
            wireframe_path,
            mesh_path,
            pose_processor,
            visual_effects
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

def process_video_frames(input_path, mediapipe_path, wireframe_path, mesh_path, pose_processor, visual_effects):
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

                    # Generate 3D wireframe visualization
                    wireframe_frame = pose_processor._generate_3d_wireframe(
                        frame_data['landmarks']
                    )

                    # Generate 3D mesh visualization
                    mesh_frame = pose_processor._generate_3d_mesh_frame(
                        frame_data['landmarks']
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
                        "landmarks": frame_data.get('landmarks'),
                        "angles": frame_data.get('angles'),
                        "velocities": frame_data.get('velocities'),
                        "motionbert_results": frame_data.get('motionbert_results')
                    }
                    metadata["frames"].append(frame_metadata)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
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