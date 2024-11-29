import json
import os
from datetime import datetime

from utils.keypoints import KeyPointUtils

class KeyframeUtils:
    def __init__(self) -> None:
        self.keypoints = KeyPointUtils()
        
    def flatten_data(self,data):
        """Flattens nested lists and dictionaries for Neo4j compatibility."""
        if isinstance(data, dict):
            # Convert dictionaries to a JSON string
            return json.dumps(data)
        elif isinstance(data, list):
            # Convert list elements to JSON strings if they are dicts or lists
            return [json.dumps(item) if isinstance(item, (dict, list)) else item for item in data]
        return data

    def add_and_update_keyframes(self,
                                frame_count,
                                current_keyframe,
                                next_keyframe,
                                current_frame_grey,
                                next_frame_grey,
                                camera_matrix,
                                average_mconf,
                                current_depth_image,
                                next_depth_image,
                                next_imu_data,
                                current_imu_data,
                                spatial_db,
                                rgb_files,
                                graph_database,
                                pose_graph):
        """Add new keyframes to the graph and update them with feature points."""

        timestamp = datetime.now().time()

        # Insert current keyframe if it does not exist in the spatial database
        # Check if the keyframe already exists in the spatial database
        spatial_db.cursor.execute("SELECT 1 FROM keyframes WHERE keyframe_id = %s", (frame_count,))
        not_exists = spatial_db.cursor.fetchone()

        if not_exists is None:  # Only insert if it doesn't exist
            current_orientation = current_imu_data.iloc[0][['qx', 'qy', 'qz', 'qw']].to_dict()
            current_filename = os.path.basename(rgb_files[frame_count])
            spatial_db.insert_keyframe(
                frame_count, timestamp, 0.0, 0.0, 0.0,
                current_orientation['qx'], current_orientation['qy'], current_orientation['qz'], current_orientation['qw'],
                current_filename
            )
        else:
            print(f"Keyframe with ID {frame_count} already exists or insertion failed.")

        # Insert next keyframe if it does not exist in the spatial database
        spatial_db.cursor.execute("SELECT 1 FROM keyframes WHERE keyframe_id = %s", (frame_count + 1,))
        not_exists = spatial_db.cursor.fetchone()

        if not_exists is None:
            next_orientation = next_imu_data.iloc[0][['qx', 'qy', 'qz', 'qw']].to_dict()
            next_filename = os.path.basename(rgb_files[frame_count + 1])
            spatial_db.insert_keyframe(
                frame_count + 1, timestamp, 0.0, 0.0, 0.0,
                next_orientation['qx'], next_orientation['qy'], next_orientation['qz'], next_orientation['qw'],
                next_filename
            )
        else:
            print(f"Keyframe with ID {frame_count + 1} already exists or insertion failed.")

        # Extract and assign feature points and descriptors
        keypoints1, descriptors1 = self.keypoints.extract_keypoints(current_frame_grey)
        keypoints2, descriptors2 = self.keypoints.extract_keypoints(next_frame_grey)

        # Save keyframes and relationship in Neo4j
        with graph_database.session(database="neo4j") as session:
            session.execute_write(pose_graph.save_keyframe, current_keyframe)
            session.execute_write(pose_graph.save_keyframe, next_keyframe)
            session.execute_write(pose_graph.add_relationship, current_keyframe.id, next_keyframe.id)

        # Convert keypoints to 3D points
        current_3d_points = self.keypoints.convert_keypoints_to_3d(keypoints1, current_depth_image, camera_matrix)
        next_3d_points = self.keypoints.convert_keypoints_to_3d(keypoints2, next_depth_image, camera_matrix)

        return current_3d_points, next_3d_points

    def update_keyframe(self,frame_count, existing_keyframe, new_keyframe, new_frame_gray, camera_matrix, average_mconf,
                        new_keyframe_depth, new_imu_data, rgb_files, spatial_db, graph_database, pose_graph):
        """Update an existing keyframe and connect it to a new keyframe."""

        timestamp = datetime.now().time()

        # Extract features from the new frame
        keypoints_new, descriptors_new = self.keypoints.extract_keypoints(new_frame_gray)
        new_keyframe.keypoints, new_keyframe.descriptors = keypoints_new, descriptors_new

        # Check if the keyframe already exists in the spatial database
        spatial_db.cursor.execute("SELECT 1 FROM keyframes WHERE keyframe_id = %s", (frame_count,))
        not_exists = spatial_db.cursor.fetchone()

        if not_exists is None:  # Only insert if it doesn't exist
            next_orientation = new_imu_data.iloc[0][['qx', 'qy', 'qz', 'qw']].to_dict()
            next_filename = os.path.basename(rgb_files[frame_count])
            spatial_db.insert_keyframe(
                frame_count, timestamp, 0.0, 0.0, 0.0,
                next_orientation['qx'], next_orientation['qy'], next_orientation['qz'], next_orientation['qw'],
                next_filename
            )
        else:
            print(f"Keyframe with ID {frame_count} already exists or insertion failed. Skipping insertion.")

        # Save the keyframe and relationship in Neo4j
        with graph_database.session(database="neo4j") as session:
            session.execute_write(pose_graph.save_keyframe, new_keyframe)
            session.execute_write(pose_graph.add_relationship, existing_keyframe.id, new_keyframe.id)

        # Convert keypoints to 3D points
        points_3d = self.keypoints.convert_keypoints_to_3d(keypoints_new, new_keyframe_depth, camera_matrix)

        return points_3d
