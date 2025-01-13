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

    def _convert_to_motionbert_format(self, landmarks):
        """Convert MediaPipe landmarks to MotionBERT's H36M format.
        
        H36M Joint Order:
        0: Hip (Pelvis center)
        1: RHip
        2: RKnee
        3: RAnkle
        4: LHip
        5: LKnee
        6: LAnkle
        7: Spine
        8: Thorax
        9: Neck/Nose
        10: Head
        11: LShoulder
        12: LElbow
        13: LWrist
        14: RShoulder
        15: RElbow
        16: RWrist
        """
        if not landmarks:
            return None

        try:
            # Initialize output array: [batch_size, frames, joints, coords]
            converted = np.zeros((1, 1, 17, 3))
            
            # Calculate mid-points for virtual joints
            if len(landmarks) > 24:  # Ensure we have the required landmarks
                # Hip center (middle of hips)
                left_hip = np.array(landmarks[23][:3])
                right_hip = np.array(landmarks[24][:3])
                hip_center = (left_hip + right_hip) / 2
                
                # Spine (between hip center and neck)
                neck = np.array(landmarks[12][:3])  # Use right shoulder level as reference
                spine = hip_center + (neck - hip_center) * 0.4
                
                # Thorax (between spine and neck)
                thorax = hip_center + (neck - hip_center) * 0.8
                
                # Head (above neck)
                nose = np.array(landmarks[0][:3])
                head = nose + np.array([0, -30, 0])  # Offset above nose
                
                # Store these virtual joints
                virtual_joints = {
                    0: hip_center,    # Hip center
                    7: spine,         # Spine
                    8: thorax,        # Thorax
                    10: head          # Head
                }
                
                # MediaPipe to H36M joint mapping
                joint_mapping = {
                    24: 1,  # RHip
                    26: 2,  # RKnee
                    28: 3,  # RAnkle
                    23: 4,  # LHip
                    25: 5,  # LKnee
                    27: 6,  # LAnkle
                    0: 9,   # Nose as Neck
                    11: 11, # LShoulder
                    13: 12, # LElbow
                    15: 13, # LWrist
                    12: 14, # RShoulder
                    14: 15, # RElbow
                    16: 16  # RWrist
                }
                
                # First, add the virtual joints
                for h36m_idx, coords in virtual_joints.items():
                    coords = np.array(coords)
                    # Normalize coordinates
                    coords[0] = (coords[0] - 640) / 640  # X coordinate
                    coords[1] = (coords[1] - 360) / 360  # Y coordinate
                    coords[2] = coords[2] / 100  # Z coordinate
                    converted[0, 0, h36m_idx] = coords
                
                # Then map the actual joints
                for mp_idx, h36m_idx in joint_mapping.items():
                    if mp_idx < len(landmarks) and landmarks[mp_idx] is not None:
                        coords = np.array(landmarks[mp_idx][:3])
                        # Normalize coordinates
                        coords[0] = (coords[0] - 640) / 640  # X coordinate
                        coords[1] = (coords[1] - 360) / 360  # Y coordinate
                        coords[2] = coords[2] / 100  # Z coordinate
                        converted[0, 0, h36m_idx] = coords
                
                # Debug logging
                logger.debug(f"Converted shape: {converted.shape}")
                logger.debug(f"Sample joint positions:\n" + 
                        "\n".join([f"Joint {i}: {converted[0, 0, i]}" 
                                    for i in range(17)]))
                
                # Convert to tensor
                return torch.tensor(converted, dtype=torch.float32, device=self.device)
                
            else:
                logger.error("Not enough landmarks for conversion")
                return None
                
        except Exception as e:
            logger.error(f"Error converting to MotionBERT format: {str(e)}")
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
        """Generate a 3D wireframe visualization using H36M format"""
        if not landmarks:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        try:
            # Convert landmarks to MotionBERT format
            motionbert_input = self._convert_to_motionbert_format(landmarks)
            
            # Get 3D pose estimation
            with torch.no_grad():
                pose_3d = self.pose_model(motionbert_input)
                pose_3d = pose_3d.cpu().numpy()[0, 0]  # Shape: (17, 3)
            
            # Create visualization frame with dark background
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 32
            
            # Define H36M skeleton connections
            h36m_skeleton = [
                # Torso
                (0, 1),    # Hip to RHip
                (0, 4),    # Hip to LHip
                (1, 2),    # RHip to RKnee
                (2, 3),    # RKnee to RAnkle
                (4, 5),    # LHip to LKnee
                (5, 6),    # LKnee to LAnkle
                (0, 7),    # Hip to Spine
                (7, 8),    # Spine to Thorax
                (8, 9),    # Thorax to Neck
                (9, 10),   # Neck to Head
                # Arms
                (8, 11),   # Thorax to LShoulder
                (11, 12),  # LShoulder to LElbow
                (12, 13),  # LElbow to LWrist
                (8, 14),   # Thorax to RShoulder
                (14, 15),  # RShoulder to RElbow
                (15, 16),  # RElbow to RWrist
            ]
            
            # Visualization parameters
            scale_factor = 200  # Adjust this value to change the size of the skeleton
            center_x, center_y = 640, 360
            
            # Normalize and center the 3D coordinates
            pose_mean = np.mean(pose_3d, axis=0)
            pose_centered = pose_3d - pose_mean
            
            # Project 3D coordinates to 2D
            projected_coords = []
            for joint in pose_centered:
                # Simple perspective projection
                z = joint[2] * scale_factor + 2000  # Increased depth offset
                x = int(center_x + (joint[0] * scale_factor * 1000) / (z + 1e-4))
                y = int(center_y + (joint[1] * scale_factor * 1000) / (z + 1e-4))
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(1279, x))
                y = max(0, min(719, y))
                depth = max(-1.0, min(1.0, joint[2]))  # Normalize depth
                projected_coords.append((x, y, depth))
            
            # Draw skeleton connections with depth-based coloring
            for connection in h36m_skeleton:
                start_idx, end_idx = connection
                start = projected_coords[start_idx]
                end = projected_coords[end_idx]
                
                # Calculate color based on depth
                avg_depth = (start[2] + end[2]) / 2
                # Use a color gradient from red (far) to blue (near)
                r = int(255 * (1 + avg_depth) / 2)
                b = int(255 * (1 - avg_depth) / 2)
                color = (b, 0, r)  # BGR format
                
                # Draw line with anti-aliasing
                cv2.line(frame, 
                        (start[0], start[1]), 
                        (end[0], end[1]), 
                        color, 2, cv2.LINE_AA)
            
            # Draw joints with depth-based size and color
            for i, (x, y, depth) in enumerate(projected_coords):
                # Size based on depth (closer joints are larger)
                size = int(4 * (1.5 - depth))  # Larger size for closer joints
                
                # Color based on joint type
                if i == 0:  # Hip center
                    color = (255, 255, 255)  # White
                elif i in [1, 2, 3, 4, 5, 6]:  # Legs
                    color = (0, 0, 255)  # Red
                elif i in [11, 12, 13, 14, 15, 16]:  # Arms
                    color = (255, 0, 0)  # Blue
                else:  # Spine and head
                    color = (0, 255, 0)  # Green
                    
                cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)
            
            # Add coordinate system
            axis_length = 50
            origin = (50, 670)
            cv2.line(frame, origin, (origin[0] + axis_length, origin[1]), (0, 0, 255), 2)  # X-axis
            cv2.line(frame, origin, (origin[0], origin[1] - axis_length), (0, 255, 0), 2)  # Y-axis
            cv2.putText(frame, "X", (origin[0] + axis_length + 5, origin[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Y", (origin[0], origin[1] - axis_length - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error generating 3D wireframe: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def _generate_3d_mesh_frame(self, landmarks):
        """Generate a 3D mesh visualization using H36M format"""
        if not landmarks:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        try:
            # Convert landmarks to MotionBERT format
            motionbert_input = self._convert_to_motionbert_format(landmarks)
            
            # Get mesh representation
            with torch.no_grad():
                mesh_output = self.mesh_model(motionbert_input)
                mesh_vertices = mesh_output.cpu().numpy()[0, 0]  # Shape: (N, 3)
            
            # Create visualization frame
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 32
            
            # Define body segments based on H36M format
            body_segments = [
                # Torso
                ([0, 1, 4, 7], (0, 255, 255)),    # Hip region
                ([7, 8, 11, 14], (0, 200, 255)),  # Upper torso
                # Left leg
                ([4, 5, 6], (0, 150, 255)),
                # Right leg
                ([1, 2, 3], (0, 150, 255)),
                # Left arm
                ([11, 12, 13], (0, 100, 255)),
                # Right arm
                ([14, 15, 16], (0, 100, 255)),
            ]
            
            # Visualization parameters
            scale_factor = 200
            center_x, center_y = 640, 360
            
            # Project vertices to 2D space
            projected_vertices = []
            for vertex in mesh_vertices:
                # Perspective projection
                z = vertex[2] * scale_factor + 2000
                x = int(center_x + (vertex[0] * scale_factor * 1000) / (z + 1e-4))
                y = int(center_y + (vertex[1] * scale_factor * 1000) / (z + 1e-4))
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(1279, x))
                y = max(0, min(719, y))
                depth = max(-1.0, min(1.0, vertex[2]))
                projected_vertices.append((x, y, depth))

            # Draw mesh segments
            for segment_indices, base_color in body_segments:
                points = []
                depths = []
                for idx in segment_indices:
                    if idx < len(projected_vertices):
                        points.append(projected_vertices[idx][:2])
                        depths.append(projected_vertices[idx][2])
                
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    points = np.array(points, dtype=np.int32)
                    avg_depth = np.mean(depths)
                    
                    # Create gradient color based on depth
                    color_factor = (1 - avg_depth) * 0.7 + 0.3
                    color = tuple(int(c * color_factor) for c in base_color)
                    
                    # Draw filled polygon with anti-aliasing
                    cv2.fillPoly(frame, [points], color)
                    # Draw edges
                    cv2.polylines(frame, [points], True, (255, 255, 255), 1, cv2.LINE_AA)

            return frame
            
        except Exception as e:
            logger.error(f"Error generating 3D mesh: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
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