import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..models import Keypoint
import math

class SkeletonDrawer:
    """Utility class for advanced skeleton drawing and visualization effects"""
    
    def __init__(
        self,
        line_thickness: int = 3,
        joint_radius: int = 4,
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        joint_color: Tuple[int, int, int] = (0, 0, 255),
        use_gradient: bool = True,
        highlight_low_confidence: bool = True
    ):
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self.skeleton_color = skeleton_color
        self.joint_color = joint_color
        self.use_gradient = use_gradient
        self.highlight_low_confidence = highlight_low_confidence
        
        # Define skeleton connections for drawing
        self.skeleton_connections = [
            # Torso connections
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),
            
            # Arms
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
            
            # Hands (if keypoints available)
            ("LEFT_WRIST", "LEFT_THUMB"),
            ("LEFT_WRIST", "LEFT_PINKY"),
            ("LEFT_WRIST", "LEFT_INDEX"),
            ("RIGHT_WRIST", "RIGHT_THUMB"),
            ("RIGHT_WRIST", "RIGHT_PINKY"),
            ("RIGHT_WRIST", "RIGHT_INDEX"),
            
            # Legs
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE"),
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),
            
            # Feet (if keypoints available)
            ("LEFT_ANKLE", "LEFT_HEEL"),
            ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
            ("RIGHT_ANKLE", "RIGHT_HEEL"),
            ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
            
            # Face connections (if keypoints available)
            ("NOSE", "LEFT_EYE"),
            ("NOSE", "RIGHT_EYE"),
            ("LEFT_EYE", "LEFT_EAR"),
            ("RIGHT_EYE", "RIGHT_EAR")
        ]
        
        # Define joint groups for colored visualization
        self.joint_groups = {
            "head": ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR"],
            "torso": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
            "arms": ["LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"],
            "hands": ["LEFT_THUMB", "LEFT_INDEX", "LEFT_PINKY", "RIGHT_THUMB", "RIGHT_INDEX", "RIGHT_PINKY"],
            "legs": ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"],
            "feet": ["LEFT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"]
        }
        
        # Predefined colors for joint groups
        self.group_colors = {
            "head": (255, 255, 0),    # Yellow
            "torso": (0, 255, 255),   # Cyan
            "arms": (0, 165, 255),    # Orange
            "hands": (255, 0, 255),   # Magenta
            "legs": (255, 0, 0),      # Blue
            "feet": (0, 255, 0)       # Green
        }
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Keypoint],
        draw_style: str = "standard"
    ) -> np.ndarray:
        """
        Draw skeleton on frame with the specified style
        
        Args:
            frame: Video frame
            keypoints: Dictionary of keypoints
            draw_style: Style of skeleton drawing ("standard", "colored", "neon", "xray")
            
        Returns:
            Frame with skeleton drawn
        """
        if draw_style == "colored":
            return self._draw_colored_skeleton(frame, keypoints)
        elif draw_style == "neon":
            return self._draw_neon_skeleton(frame, keypoints)
        elif draw_style == "xray":
            return self._draw_xray_skeleton(frame, keypoints)
        else:
            return self._draw_standard_skeleton(frame, keypoints)
    
    def _draw_standard_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Keypoint]
    ) -> np.ndarray:
        """Draw standard skeleton"""
        height, width = frame.shape[:2]
        
        # Draw lines
        for p1_name, p2_name in self.skeleton_connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.3 or p2.confidence < 0.3:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Adjust line color based on confidence
                color = self.skeleton_color
                if self.highlight_low_confidence:
                    avg_confidence = (p1.confidence + p2.confidence) / 2
                    if avg_confidence < 0.5:
                        # Use red for low confidence
                        red_factor = (0.5 - avg_confidence) * 2  # 0 to 1
                        color = (
                            int(self.skeleton_color[0] * (1 - red_factor) + 0 * red_factor),
                            int(self.skeleton_color[1] * (1 - red_factor) + 0 * red_factor),
                            int(self.skeleton_color[2] * (1 - red_factor) + 255 * red_factor)
                        )
                
                # Adjust thickness based on confidence
                thickness = max(1, int(self.line_thickness * ((p1.confidence + p2.confidence) / 2)))
                
                cv2.line(frame, p1_px, p2_px, color, thickness)
        
        # Draw joints
        for name, kp in keypoints.items():
            if kp.confidence < 0.3:
                continue
                
            x_px = int(kp.x * width)
            y_px = int(kp.y * height)
            
            # Adjust radius based on confidence
            radius = max(1, int(self.joint_radius * kp.confidence))
            
            # Adjust color based on confidence
            color = self.joint_color
            if self.highlight_low_confidence and kp.confidence < 0.5:
                # Use yellow for low confidence
                red_factor = (0.5 - kp.confidence) * 2  # 0 to 1
                color = (
                    int(self.joint_color[0] * (1 - red_factor) + 0 * red_factor),
                    int(self.joint_color[1] * (1 - red_factor) + 255 * red_factor),
                    int(self.joint_color[2] * (1 - red_factor) + 255 * red_factor)
                )
            
            cv2.circle(frame, (x_px, y_px), radius, color, -1)
        
        return frame
    
    def _draw_colored_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Keypoint]
    ) -> np.ndarray:
        """Draw skeleton with color-coded segments"""
        height, width = frame.shape[:2]
        
        # Draw skeleton segments with group colors
        for p1_name, p2_name in self.skeleton_connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                
                if p1.confidence < 0.3 or p2.confidence < 0.3:
                    continue
                
                # Convert to pixel coordinates
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Determine color based on joint group
                color = self.skeleton_color
                
                # Find the group for this connection
                for group_name, group_points in self.joint_groups.items():
                    if p1_name in group_points and p2_name in group_points:
                        color = self.group_colors[group_name]
                        break
                
                # Draw line with adaptive thickness
                thickness = max(1, int(self.line_thickness * ((p1.confidence + p2.confidence) / 2)))
                cv2.line(frame, p1_px, p2_px, color, thickness)
        
        # Draw joints
        for name, kp in keypoints.items():
            if kp.confidence < 0.3:
                continue
                
            x_px = int(kp.x * width)
            y_px = int(kp.y * height)
            
            # Determine color based on joint group
            color = self.joint_color
            for group_name, group_points in self.joint_groups.items():
                if name in group_points:
                    color = self.group_colors[group_name]
                    break
            
            radius = max(1, int(self.joint_radius * kp.confidence))
            cv2.circle(frame, (x_px, y_px), radius, color, -1)
        
        return frame
    
    def _draw_neon_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Keypoint]
    ) -> np.ndarray:
        """Draw skeleton with neon glow effect"""
        height, width = frame.shape[:2]
        
        # Create a black canvas for the skeleton
        skeleton_canvas = np.zeros_like(frame)
        
        # Draw skeleton on the canvas
        for p1_name, p2_name in self.skeleton_connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                
                if p1.confidence < 0.3 or p2.confidence < 0.3:
                    continue
                
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Draw with thicker lines for glow effect
                thickness = max(2, int(self.line_thickness * 2))
                cv2.line(skeleton_canvas, p1_px, p2_px, (0, 255, 0), thickness)
        
        # Draw joints
        for name, kp in keypoints.items():
            if kp.confidence < 0.3:
                continue
                
            x_px = int(kp.x * width)
            y_px = int(kp.y * height)
            
            # Draw larger circles for glow effect
            radius = max(2, int(self.joint_radius * 1.5))
            cv2.circle(skeleton_canvas, (x_px, y_px), radius, (0, 0, 255), -1)
        
        # Apply blur for glow effect
        glow = cv2.GaussianBlur(skeleton_canvas, (15, 15), 0)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1.0, glow, 0.8, 0)
        
        return result
    
    def _draw_xray_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Keypoint]
    ) -> np.ndarray:
        """Draw skeleton with X-ray effect"""
        height, width = frame.shape[:2]
        
        # Create a dark overlay
        overlay = np.zeros_like(frame)
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        
        # Apply dark overlay to frame
        dark_frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Create a canvas for the skeleton
        skeleton_canvas = np.zeros_like(frame)
        
        # Draw skeleton with bright colors
        for p1_name, p2_name in self.skeleton_connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                
                if p1.confidence < 0.3 or p2.confidence < 0.3:
                    continue
                
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                
                # Draw with bright cyan color
                thickness = max(2, int(self.line_thickness * 1.5))
                cv2.line(skeleton_canvas, p1_px, p2_px, (255, 255, 0), thickness)
        
        # Draw joints with bright colors
        for name, kp in keypoints.items():
            if kp.confidence < 0.3:
                continue
                
            x_px = int(kp.x * width)
            y_px = int(kp.y * height)
            
            radius = max(2, int(self.joint_radius * 1.5))
            cv2.circle(skeleton_canvas, (x_px, y_px), radius, (0, 255, 255), -1)
        
        # Apply slight blur for glow
        skeleton_canvas = cv2.GaussianBlur(skeleton_canvas, (5, 5), 0)
        
        # Combine dark frame with skeleton
        result = cv2.addWeighted(dark_frame, 1.0, skeleton_canvas, 1.0, 0)
        
        return result
    
    def draw_motion_trail(
        self,
        frame: np.ndarray,
        keypoint_history: List[Dict[str, Keypoint]],
        trail_length: int = 10,
        trail_opacity: float = 0.7
    ) -> np.ndarray:
        """
        Draw motion trails for tracked keypoints
        
        Args:
            frame: Video frame
            keypoint_history: List of keypoint dictionaries for previous frames
            trail_length: Number of previous frames to show in trail
            trail_opacity: Opacity factor for trail
            
        Returns:
            Frame with motion trails
        """
        height, width = frame.shape[:2]
        
        # Create a transparent overlay for the trails
        trail_canvas = np.zeros_like(frame)
        
        # Limit history to trail length
        history = keypoint_history[-trail_length:]
        
        # Key points to track for trails
        trail_points = [
            "LEFT_WRIST", "RIGHT_WRIST",  # Hands
            "LEFT_ANKLE", "RIGHT_ANKLE",  # Feet
            "NOSE"                        # Head
        ]
        
        # Draw trails for each tracked point
        for point_name in trail_points:
            points = []
            
            # Collect point positions from history
            for frame_keypoints in history:
                if point_name in frame_keypoints and frame_keypoints[point_name].confidence > 0.5:
                    kp = frame_keypoints[point_name]
                    points.append((int(kp.x * width), int(kp.y * height)))
            
            # Draw trail if we have enough points
            if len(points) >= 2:
                # Create gradient colors for trail
                for i in range(1, len(points)):
                    # Calculate opacity based on position in trail
                    alpha = (i / len(points)) * trail_opacity
                    
                    # Calculate color based on point type
                    if point_name.endswith("WRIST"):
                        color = (0, int(255 * alpha), int(255 * alpha))  # Cyan for wrists
                    elif point_name.endswith("ANKLE"):
                        color = (int(255 * alpha), int(255 * alpha), 0)  # Yellow for ankles
                    else:
                        color = (int(255 * alpha), 0, int(255 * alpha))  # Magenta for head
                    
                    # Draw line with variable thickness
                    thickness = max(1, int(self.line_thickness * (i / len(points))))
                    cv2.line(trail_canvas, points[i-1], points[i], color, thickness)
        
        # Blend trail canvas with original frame
        result = cv2.addWeighted(frame, 1.0, trail_canvas, 1.0, 0)
        
        return result
    
    def draw_heatmap(
        self,
        frame: np.ndarray,
        keypoint_history: List[Dict[str, Keypoint]],
        radius: int = 15,
        opacity: float = 0.6
    ) -> np.ndarray:
        """
        Generate a heatmap of movement intensity
        
        Args:
            frame: Video frame
            keypoint_history: List of keypoint dictionaries for previous frames
            radius: Radius of heat spots
            opacity: Opacity of heatmap overlay
            
        Returns:
            Frame with heatmap overlay
        """
        height, width = frame.shape[:2]
        
        # Create a blank heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Key points to use for heatmap
        heat_points = [
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ELBOW", "RIGHT_ELBOW"
        ]
        
        # Add heat to the map for each keypoint position
        for frame_idx, frame_keypoints in enumerate(keypoint_history):
            # Weight recent frames more
            recency_weight = frame_idx / len(keypoint_history)
            
            for point_name in heat_points:
                if point_name in frame_keypoints and frame_keypoints[point_name].confidence > 0.5:
                    kp = frame_keypoints[point_name]
                    x, y = int(kp.x * width), int(kp.y * height)
                    
                    # Create a Gaussian heat spot
                    for dy in range(-radius, radius+1):
                        for dx in range(-radius, radius+1):
                            if 0 <= y+dy < height and 0 <= x+dx < width:
                                # Calculate distance from center
                                distance = math.sqrt(dx*dx + dy*dy)
                                
                                if distance <= radius:
                                    # Add heat with Gaussian falloff
                                    heat_value = math.exp(-(distance*distance)/(2*(radius/2)*(radius/2)))
                                    heat_value *= recency_weight * kp.confidence
                                    heatmap[y+dy, x+dx] += heat_value
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Apply colormap to create RGB heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create mask for blending
        mask = (heatmap * 255 * opacity).astype(np.uint8)
        mask = cv2.merge([mask, mask, mask])
        
        # Blend heatmap with original frame
        result = np.where(mask > 0, cv2.addWeighted(frame, 1.0, heatmap_colored, opacity, 0), frame)
        
        return result.astype(np.uint8)
    
    def highlight_joint_angles(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Keypoint],
        angle_definitions: Dict[str, Tuple[str, str, str]],
        normal_ranges: Dict[str, Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Highlight and measure joint angles
        
        Args:
            frame: Video frame
            keypoints: Dictionary of keypoints
            angle_definitions: Dictionary mapping angle names to triplets of keypoint names
            normal_ranges: Dictionary mapping angle names to normal angle ranges
            
        Returns:
            Tuple of (rendered frame, dictionary of measured angles)
        """
        height, width = frame.shape[:2]
        angle_results = {}
        
        if normal_ranges is None:
            normal_ranges = {
                "left_elbow": (10, 160),
                "right_elbow": (10, 160),
                "left_knee": (10, 170),
                "right_knee": (10, 170),
                "left_hip": (80, 160),
                "right_hip": (80, 160),
                "left_shoulder": (50, 170),
                "right_shoulder": (50, 170)
            }
        
        # Process each angle
        for angle_name, (p1_name, p2_name, p3_name) in angle_definitions.items():
            if all(p_name in keypoints for p_name in [p1_name, p2_name, p3_name]):
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                p3 = keypoints[p3_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.5 or p2.confidence < 0.5 or p3.confidence < 0.5:
                    continue
                
                # Calculate angle
                angle = self._calculate_angle(
                    (p1.x, p1.y),
                    (p2.x, p2.y),
                    (p3.x, p3.y)
                )
                angle_results[angle_name] = angle
                
                # Convert to pixel coordinates
                p1_px = (int(p1.x * width), int(p1.y * height))
                p2_px = (int(p2.x * width), int(p2.y * height))
                p3_px = (int(p3.x * width), int(p3.y * height))
                
                # Determine color based on normal range
                color = (0, 255, 0)  # Default: green
                if angle_name in normal_ranges:
                    min_angle, max_angle = normal_ranges[angle_name]
                    if angle < min_angle:
                        color = (0, 0, 255)  # Red for below range
                    elif angle > max_angle:
                        color = (255, 165, 0)  # Orange for above range
                
                # Draw angle arc
                self._draw_angle_arc(frame, p1_px, p2_px, p3_px, angle, color)
                
                # Add angle text
                text_x = p2_px[0] + 10
                text_y = p2_px[1] + 10
                
                cv2.putText(
                    frame,
                    f"{angle:.1f}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
        
        return frame, angle_results
    
    def _calculate_angle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        """Calculate angle between three points"""
        # Convert to numpy arrays
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _draw_angle_arc(
        self,
        frame: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        p3: Tuple[int, int],
        angle: float,
        color: Tuple[int, int, int],
        radius: int = 30
    ) -> None:
        """Draw arc showing the angle"""
        # Calculate vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # Calculate start and end angles
        start_angle = np.arctan2(v1_norm[1], v1_norm[0])
        end_angle = np.arctan2(v2_norm[1], v2_norm[0])
        
        # Convert to degrees
        start_angle_deg = np.degrees(start_angle) % 360
        end_angle_deg = np.degrees(end_angle) % 360
        
        # Ensure correct arc direction
        if abs(end_angle_deg - start_angle_deg) > 180:
            if start_angle_deg < end_angle_deg:
                start_angle_deg += 360
            else:
                end_angle_deg += 360
        
        # Draw arc
        cv2.ellipse(
            frame,
            p2,
            (radius, radius),
            0,  # Angle of ellipse
            min(start_angle_deg, end_angle_deg),  # Start angle
            max(start_angle_deg, end_angle_deg),  # End angle
            color,
            2,
            cv2.LINE_AA
        )