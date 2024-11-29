import numpy as np

class Keyframe:
    """
     set data for identified key frame

        1. id  # Unique identifier for the keyframe
        2. pose  # Pose of the keyframe (e.g., [x, y, theta])
        3. descriptors  # Descriptors extracted from the keyframe
        4. keypoints  # Keypoints extracted from the keyframe

    """
    def __init__(self, id, pose, descriptors, keypoints):
        self.id = id  # Unique identifier for the keyframe
        self.pose = pose  # Pose of the keyframe (e.g., [x, y, theta])
        self.descriptors = descriptors  # Descriptors extracted from the keyframe
        self.keypoints = keypoints  # Keypoints extracted from the keyframe


    # Add a method to convert Keyframe to a dictionary
    def to_dict(self):
        # Convert NumPy arrays to lists for JSON serialization
        pose_list = self.pose.tolist() if isinstance(self.pose, np.ndarray) else self.pose
        # Convert keypoints and descriptors to lists of lists
        keypoints_list = self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints
        descriptors_list = self.descriptors.tolist() if isinstance(self.descriptors, np.ndarray) else self.descriptors

        return {
            'id': self.id,
            'pose': pose_list
        }