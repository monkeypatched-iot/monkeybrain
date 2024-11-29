import tempfile
import numpy as np
import os
import boto3

class CameraHelper:

    def __init__(self):
        # Get the S3 URL from the environment variable
        self.camera_info_file = os.getenv('CAMERA_INFO_FILE')
        if not self.camera_info_file:
            raise ValueError("CAMERA_INFO_FILE environment variable is not set.")
        
    def download_camera_info(self):
        """
        Download the camera info file from S3 to a temporary location.
        """
        # Parse the S3 path
        if not self.camera_info_file.startswith("s3://"):
            raise ValueError("The camera info file path must start with 's3://'.")

        s3_url = self.camera_info_file[5:]  # Remove "s3://" prefix
        bucket_name, key = s3_url.split('/', 1)

        # Create a boto3 S3 client
        s3_client = boto3.client('s3')

        # Download to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3_client.download_file(bucket_name, key, temp_file.name)
            temp_file.close()
            return temp_file.name  # Return the path of the downloaded file

    def read_camera_info(self):
        camera_info_file = self.download_camera_info()
        
        camera_info = {}
        with open(camera_info_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(':')
                camera_info[key.strip()] = value.strip()
        return camera_info

    def calculate_projection_matrix(self, fx, fy, cx, cy, R, t):
        """
        Calculate the camera projection matrix.

        :param fx: Focal length in x
        :param fy: Focal length in y
        :param cx: Optical center x
        :param cy: Optical center y
        :param R: Rotation matrix (3x3)
        :param t: Translation vector (3x1)
        :return: Projection matrix (3x4)
        """
        # Intrinsic matrix
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        # Create the extrinsic matrix [R|t]
        extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))

        # Camera projection matrix
        P = K @ extrinsic_matrix
        return P

    def get_camera_matrix(self):
        """
        Compute the full camera matrix (intrinsics @ extrinsics).

        :return: A 3x3 camera matrix.
        """
        camera_info = self.read_camera_info()

        # Parse Intrinsics and Extrinsics
        intrinsics = np.array([float(x) for x in camera_info['Intrinsics'].strip('[] ').split()])
        extrinsics = np.array([float(x) for x in camera_info['Extrinsics'].strip('[] ').split()])

        # Reshape to appropriate dimensions
        intrinsics = intrinsics.reshape(3, 3)  # 3x3 intrinsic matrix
        extrinsics = extrinsics.reshape(3, 3)  # 3x3 extrinsic matrix

        # Compute the camera matrix
        camera_matrix = intrinsics @ extrinsics

        return camera_matrix
