import io
import matplotlib.pyplot as plt
import numpy as np
from database.spatial import SpatialDatabase
from database.timeseries import TimeseriesDatabase
from neo4j import GraphDatabase
import os
import cv2
import logging
from entities.pose import Pose
from entities.imu import IMU
from entities.keyframe import Keyframe
from utils.keyframes import KeyframeUtils
from utils.file import FileUtils
from utils.image import ImageUtils
import boto3
from botocore.exceptions import ClientError
import tempfile
from PIL import Image
import re
from datetime import datetime




# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Generator:
  
  def __init__(self):
    self.s3_client = boto3.client("s3")

  def read_s3_image(self, s3_folder, s3_file):
        """
        Reads an image from an S3 path without downloading it to local storage.

        :param s3_folder: S3 folder path
        :param s3_file: S3 file name
        :return: OpenCV image (numpy array) or None if error
        """
        try:
            bucket_name, _ = s3_folder.replace("s3://", "").split("/", 1)
            s3_object = self.s3_client.get_object(Bucket=bucket_name, Key=s3_file)
            img_data = s3_object['Body'].read()
            img = Image.open(io.BytesIO(img_data))
            return np.array(img)
        except ClientError as e:
            print(f"Error reading image {s3_file} from S3: {e}")
            return None
        
  def list_s3_files(self, s3_folder):
    """
    Lists files in an S3 folder.

    :param s3_folder: The S3 folder path (e.g., 's3://bucket-name/folder/')
    :return: List of file names in the folder
    """
    bucket_name, prefix = s3_folder.replace("s3://", "").split("/", 1)
    files = []
    try:
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                files.extend([file['Key'] for file in page['Contents']])
    except ClientError as e:
        print(f"Error listing S3 files: {e}")
    return files




  
  def validate_local_or_s3_path(self,path, description):
        print(path)
        if not path:
            raise ValueError(f"{description} is not set or empty.")

        if path.startswith("s3://"):
            bucket_name, key = path.replace("s3://", "").split("/", 1)
            s3_client = boto3.client("s3")
            try:
                s3_client.head_object(Bucket=bucket_name, Key=key)
                print(f"{description} exists in S3: {path}")
            except ClientError as e:
                raise ValueError(f"{description} not found in S3: {path}. Error: {e}")
        elif not os.path.exists(path):
            raise ValueError(f"Invalid or missing {description}: {path}")
        else:
            print(f"{description} exists locally: {path}")

  def validate_paths(self,rgb_folder,depth_folder,imu_data,camera_matrix):
      """
      Validates the provided paths for RGB, depth, IMU data, and camera matrix using environment variables.

      :raises ValueError: If any path is invalid or missing, or if the camera matrix is not provided.
      """

      print("path"+os.environ.get("RGB_FOLDER"))

      if camera_matrix:
          # Parse the camera matrix from a string, e.g., "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
          camera_matrix = eval(camera_matrix)

      # Validate paths
      self.validate_local_or_s3_path(rgb_folder, "RGB folder")
      self.validate_local_or_s3_path(depth_folder, "Depth folder")
      self.validate_local_or_s3_path(imu_data, "IMU data file")

      if camera_matrix is None:
          raise ValueError("Camera matrix must be provided.")
      else:
          print(f"Camera matrix is valid: {camera_matrix}")
  
  def visualize_occupancy_grid(self,occupancy_grid):
    """Visualizes the occupancy grid using matplotlib."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of occupied cells
    x, y, z = np.where(occupancy_grid == 1)

    # Plot the occupied cells as points
    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Occupancy Grid')
    plt.show()

  # Function to read an image from S3
  def read_image_from_s3(self,bucket, key, mode=cv2.IMREAD_COLOR):
    response = self.s3_client.get_object(Bucket=bucket, Key=key)
    image_data = response['Body'].read()
    np_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_array, mode)
  
  def create_point_cloud(self, rgb_folder, depth_folder, imu_data, camera_matrix):
    """
    Create a point cloud from RGB and depth images, and save the result in a spatial database.

    Parameters:
    - rgb_folder: Path to the folder containing RGB images.
    - depth_folder: Path to the folder containing depth images.
    - imu_data: path to the imu data csv file
    - camera_matrix: Camera intrinsic matrix used to convert depth images to 3D coordinates.

    Returns:
    - None
    """
    # Validate input parameters (ensure folders exist, camera_matrix is valid, etc.)
    if not rgb_folder:
        raise ValueError("RGB folder path is missing.")
    if not depth_folder:
        raise ValueError("Depth folder path is missing.")
    if not imu_data:
        raise ValueError("IMU data file path is missing.")
    if camera_matrix is None:
        raise ValueError("Camera matrix must be provided.")
    
    try:
        # Create spatial database connection
        spatial_db = SpatialDatabase(
            db_name=os.getenv("SPATIAL_DBNAME"),
            user=os.getenv("SPATIAL_DB_USER"),
            password=os.getenv("SPATIAL_DB_PASSWORD"),
            host=os.getenv("SPATIAL_DB_HOST"),
            port=os.getenv("SPATIAL_DB_PORT")
        )
        print("Connected to Spatial Database")

        # Initialize the InfluxDB Client
        timeseries_db = TimeseriesDatabase(
            host=os.getenv("TIME_SERIES_DB_HOST"),
            token=os.getenv("TIME_SERIES_DB_TOKEN"),
            org=os.getenv("TIME_SERIES_DB_ORG"),
            database=os.getenv("TIME_SERIES_DB_NAME"),
            file="imu_data"  # Assuming the file argument is needed as per your original code
        )
        print("Connected to InfluxDB")

        # Create a keyframe graph
        graph_database = GraphDatabase.driver(
            os.getenv("GRAPH_DB_URI"),
            auth=(os.getenv("GRAPH_DB_USERNAME"), os.getenv("GRAPH_DB_PASSWORD"))
        )
        print("Connected to Neo4j")

        # Initiate the helper functions for the graph
        pose_graph = Pose()

        # Initiate the helper for imu data
        imu = IMU()

        # Initialize the graph database
        with graph_database.session(database="neo4j") as session:
            session.execute_write(pose_graph.add_unique_constraint)

        # Get the imu data from influxdb
        imu_data = imu.get_imu_data()

        # Get lists of RGB and depth files
        rgb_files = self.list_s3_files(rgb_folder)
        depth_files = self.list_s3_files(depth_folder)

        print(f"Number of RGB files: {len(rgb_files)}")
        print(f"Number of depth files: {len(depth_files)}")
        print(f"Number of poses: {len(imu_data)}")

        # Throw error if no files
        if len(rgb_files) == 0 or len(depth_files) == 0 or len(rgb_files) != len(depth_files):
            print("Error: No images found or RGB and depth image counts do not match.")
            return None

        # Initiate the image utils
        image = ImageUtils()

        # Create a feature matcher using LoFTR
        matcher = image.set_image_type('outdoor')

        # Point cloud to be returned
        point_cloud = []

        # Rename the columns to the expected names
        column_mapping = {
            'orientation_x': 'qx',
            'orientation_y': 'qy',
            'orientation_z': 'qz',
            'orientation_w': 'qw'
        }

        imu_data = imu_data.rename(columns=column_mapping).sort_values(by='time')

        # Read RGB and depth frames
        bucket_name = os.getenv("SIMULATION_BUCKET_NAME")

        # Process all frames
        for frame_count in range(len(rgb_files)-1):

            current_rgb_frame = self.read_image_from_s3(bucket_name, rgb_files[frame_count], cv2.IMREAD_COLOR)
            current_depth_frame = self.read_image_from_s3(bucket_name, depth_files[frame_count], cv2.IMREAD_UNCHANGED)

            # Check that the frame is not empty
            if current_rgb_frame is None or current_depth_frame is None:
                print(f"Error reading frame {frame_count}")
                continue

            # Convert depth image to float (assuming depth is in mm)
            current_depth_image = current_depth_frame.astype(np.float32) / 1000.0

            file = FileUtils()

            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", rgb_files[frame_count])

            if match:
                timestamp_str = match.group(1)
                # Convert the extracted string to a datetime object
                current_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            else:
                print("No timestamp found in filename.")

            # Get the current imu data from influxdb
            current_imu_data = imu.get_imu_data_by_timestamp(current_timestamp)

            if current_imu_data.empty:
                print(f"No IMU data found for current timestamp: {current_timestamp}")
                continue

            # Get the current imu data from influxdb
            current_imu_data = imu.get_imu_data_by_timestamp(current_timestamp)

            # Read next RGB and depth frames
            next_rgb_frame = self.read_image_from_s3(bucket_name, rgb_files[frame_count+1], cv2.IMREAD_COLOR)
            next_depth_frame = self.read_image_from_s3(bucket_name, depth_files[frame_count+1], cv2.IMREAD_UNCHANGED)

            # Check that the frame is not empty
            if next_rgb_frame is None or next_depth_frame is None:
                print(f"Error reading frame {frame_count}")
                continue

            # Convert depth image to float (assuming depth is in mm)
            next_depth_image = next_depth_frame.astype(np.float32) / 1000.0

            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", rgb_files[frame_count+1])

            if match:
                timestamp_str = match.group(1)
                # Convert the extracted string to a datetime object
                next_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            else:
                print("No timestamp found in filename.")

            # Get the current imu data from influxdb
            next_imu_data = imu.get_imu_data_by_timestamp(next_timestamp)

            # Get the imu data for the next frame
            if next_imu_data.empty:
                print(f"No IMU data found for next timestamp: {next_timestamp}")
                continue

            # Process frames
            current_frame_grey = cv2.cvtColor(current_rgb_frame, cv2.COLOR_BGR2GRAY)
            next_frame_grey = cv2.cvtColor(next_rgb_frame, cv2.COLOR_BGR2GRAY)

            # Convert to PyTorch tensors
            current_image, next_image = image.convert_to_tensors(current_frame_grey, next_frame_grey)

            # Inference with LoFTR
            average_mconf = image.perform_loftr_inference(matcher, current_image, next_image)

            # Add new keyframes to the graph and update them with feature points
            if average_mconf >= 0.5:

                # Check if imu data exists
                if current_imu_data is None or current_imu_data.empty:
                    print(f"No IMU data found for current timestamp: {current_timestamp}")
                else:
                    # Convert to pandas dataframe and sort by timestamp
                    current_imu_data = current_imu_data.rename(columns=column_mapping).sort_values(by='time')

                    # Save the keyframe to postgres
                    current_keyframe = Keyframe(id=frame_count, pose=current_imu_data[['qx', 'qy', 'qz', 'qw']].values, keypoints=None, descriptors=None)

                    key_frames = KeyframeUtils()

                    # Get the next frame
                    if frame_count + 1 < len(imu_data):
                        next_keyframe_imu_data = imu.get_imu_data_by_timestamp(next_timestamp)

                        if next_imu_data is None or next_keyframe_imu_data.empty:
                            print(f"No IMU data found for next timestamp: {next_timestamp}")
                        else:
                            # Save the keyframe to postgres
                            next_imu_data = next_imu_data.rename(columns=column_mapping).sort_values(by='time')
                            next_keyframe = Keyframe(id=frame_count+1, pose=next_imu_data[['qx', 'qy', 'qz', 'qw']].values, keypoints=None, descriptors=None)

                            # Add nodes to the graph
                            current_node_exists = pose_graph.node_exists(current_keyframe.id, driver=graph_database)
                            next_node_exists = pose_graph.node_exists(next_keyframe.id, driver=graph_database)

                            if not current_node_exists and not next_node_exists:
                                points_3d_for_current_frame, points_3d_for_next_frame = key_frames.add_and_update_keyframes(
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
                                    pose_graph
                                )
                                point_cloud.append(points_3d_for_current_frame)
                                point_cloud.append(points_3d_for_next_frame)

                            elif current_node_exists:
                                points_3d_for_keyframe2 = key_frames.update_keyframe(
                                    frame_count,
                                    current_keyframe,
                                    next_keyframe,
                                    next_frame_grey,
                                    camera_matrix,
                                    average_mconf,
                                    next_depth_image,
                                    next_imu_data,
                                    rgb_files,
                                    spatial_db,
                                    graph_database,
                                    pose_graph
                                )
                                point_cloud.append(points_3d_for_keyframe2)
                            else:
                                points_3d_for_keyframe1 = key_frames.update_keyframe(
                                    frame_count + 1,
                                    next_keyframe,
                                    current_keyframe,
                                    current_frame_grey,
                                    camera_matrix,
                                    average_mconf,
                                    current_depth_image,
                                    current_imu_data,
                                    rgb_files,
                                    spatial_db,
                                    graph_database,
                                    pose_graph
                                )
                                point_cloud.append(points_3d_for_keyframe1)

        print("Mapping completed.")
        print(len(point_cloud))
        return point_cloud, graph_database  # Return the 3D point cloud as a NumPy array

    except Exception as e:
        print(f"Error occurred while creating point cloud: {e}")
    finally:
        # Make sure to close the database connection after operation
        spatial_db.close()
