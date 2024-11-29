import cv2
import matplotlib.pyplot as plt
import numpy as np 

class KeyPointUtils:
  def __init__(self) -> None:
     pass
  
  def extract_keypoints(self,image):
      """
        extract all keypoints and descriptor for the given image

      """
      # Create ORB detector
      orb = cv2.ORB_create()

      # Ensure the image is grayscale
      if len(image.shape) == 3:  # Check if the image has 3 channels (color)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale


      # Detect keypoints and compute descriptors
      keypoints, descriptors = orb.detectAndCompute(image, None)

      if keypoints is None or descriptors is None:
          print("Error: Keypoints or descriptors are None.")
          return None, None

      return keypoints, descriptors

  def convert_keypoints_to_3d(self,keypoints, depth, camera_matrix):
      """
      Converts 2D keypoints to 3D points using depth information and camera matrix.

      Args:
          keypoints: List of cv2.KeyPoint objects.
          depth: Depth image corresponding to the keypoints.
          camera_matrix: Camera intrinsic matrix.

      Returns:
          np.ndarray: A list of 3D points.
      """
      points_3d = []
      # Assuming depth has same resolution as keypoints image
      for i, kp in enumerate(keypoints):
          # Get the 2D point coordinates using kp.pt
          x, y = kp.pt  # Access x and y using .pt

          # Convert to normalized camera coordinates
          x_normalized = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
          y_normalized = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]

          # Calculate 3D point using depth and normalized coordinates
          depth_value = depth[int(y), int(x)]

          # if the depth value is greater than 0
          if depth_value > 0:  # Check for valid depth value
              z = depth_value
              point_3d = [x_normalized * z, y_normalized * z, z]
              points_3d.append(point_3d)

          # since we are using comarative depth
          if depth_value == 0:
              z = 0
              point_3d = [x_normalized, y_normalized, z]
              points_3d.append(point_3d)

      return np.array(points_3d)


  def visualize_point_cloud(self,points):
    """Visualizes a point cloud using Matplotlib.

    Args:
      points: A NumPy array of shape (N, 3) representing the point cloud.
    """
    if len(points) == 0:
      print("Warning: Point cloud is empty. Nothing to visualize.")
      return

    if points.ndim == 1:  # Check if the array is 1-dimensional
      # Reshape to (N, 3) if necessary, assuming points are stored as [x1, y1, z1, x2, y2, z2, ...]
      points = points.reshape(-1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Now you can safely access elements using two indices
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

  # Define a simple orthographic projection function
  def orthographic_projection(self,points):
      # Discard the z-coordinate for orthographic projection (X, Y plane)
      # Iterate through each point in the list
      projected_points = []
      for point_array in points:
          # Check if the point array has at least 3 dimensions
          if point_array.shape[1] >= 3:
              # Extract x, y coordinates and append to the projected points
              projected_points.extend([(x, y) for x, y, *_ in point_array])
          else:
              # Handle cases where point array has fewer than 3 dimensions
              print(f"Warning: Point array with shape {point_array.shape} has fewer than 3 dimensions. Skipping.")
      return projected_points
  
  # Function to scale values
  def scale_values(self,value, min_val, max_val, new_min, new_max):
      """Scales a value from a given range to a new range."""
      scaled_value = ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
      return scaled_value