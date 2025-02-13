import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List, Optional, Dict, Tuple
from ..models import Keypoint
import asyncio
from concurrent.futures import ThreadPoolExecutor

class MovenetEstimator:
    def __init__(self):
        # Load MoveNet (Thunder is more accurate but slower than Lightning)
        model_name = "movenet_thunder"
        self.model = hub.load(f"https://tfhub.dev/google/movenet/{model_name}/4")
        self.movenet = self.model.signatures['serving_default']
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # MoveNet keypoint mapping
        self.keypoint_map = {
            0: "NOSE",
            1: "LEFT_EYE",
            2: "RIGHT_EYE",
            3: "LEFT_EAR",
            4: "RIGHT_EAR",
            5: "LEFT_SHOULDER",
            6: "RIGHT_SHOULDER",
            7: "LEFT_ELBOW",
            8: "RIGHT_ELBOW",
            9: "LEFT_WRIST",
            10: "RIGHT_WRIST",
            11: "LEFT_HIP",
            12: "RIGHT_HIP",
            13: "LEFT_KNEE",
            14: "RIGHT_KNEE",
            15: "LEFT_ANKLE",
            16: "RIGHT_ANKLE"
        }

    async def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 5
    ) -> List[Optional[List[Keypoint]]]:
        """Process multiple frames with batching"""
        all_keypoints = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_keypoints = []
            
            # Process batch in parallel
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self.executor, self._process_single_frame, frame)
                for frame in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            all_keypoints.extend(batch_results)
            
            await asyncio.sleep(0)
        
        return all_keypoints

    def _process_single_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame with MoveNet"""
        try:
            # Prepare image for model
            input_image = self._prepare_image(frame)
            
            # Run model inference
            outputs = self.movenet(input_image)
            keypoints = outputs['output_0'].numpy().squeeze()
            
            # Convert to our keypoint format
            return self._convert_keypoints(keypoints)
            
        except Exception as e:
            print(f"MoveNet error: {str(e)}")
            return None

    def _prepare_image(self, frame: np.ndarray) -> tf.Tensor:
        """Prepare image for MoveNet input"""
        # MoveNet expects 192x192 or 256x256 images
        input_size = 256
        
        # Resize and normalize image
        image = tf.convert_to_tensor(frame)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_pad(image, input_size, input_size)
        image = tf.cast(image, dtype=tf.int32)
        
        return image

    def _convert_keypoints(self, keypoints: np.ndarray) -> List[Keypoint]:
        """Convert MoveNet keypoints to our format"""
        converted_keypoints = []
        
        for idx, keypoint in enumerate(keypoints):
            if idx in self.keypoint_map:
                y, x, confidence = keypoint
                
                # Convert to normalized coordinates
                converted_keypoints.append(
                    Keypoint(
                        x=float(x),
                        y=float(y),
                        confidence=float(confidence),
                        name=self.keypoint_map[idx]
                    )
                )
        
        return converted_keypoints

    def __del__(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=False)