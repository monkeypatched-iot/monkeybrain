from helpers.generator import Generator
from helpers.camera import CameraHelper
from utils.keypoints import KeyPointUtils
from utils.file import FileUtils
import numpy as np
import os
import numpy as np
import pandas as pd
import boto3
from io import StringIO

generator = Generator()
file = FileUtils()
camera = CameraHelper()
keypoints = KeyPointUtils()

rgb_folder = os.getenv("RGB_FOLDER")
depth_folder = os.getenv("DEPTH_FOLDER")
imu_data = os.getenv("IMU_DATA_FILE")
camera_matrix = os.getenv("CAMERA_MATRIX")

camera_matrix = camera.get_camera_matrix()

points,graph = generator.create_point_cloud(
    rgb_folder,
    depth_folder,
    imu_data,
    camera_matrix=camera_matrix
)

# Assuming 'points' is your list of points
# Find the maximum dimensions across all arrays in the list
max_dims = [0, 0]  # [max_rows, max_cols]
for p in points:
    max_dims[0] = max(max_dims[0], p.shape[0])
    max_dims[1] = max(max_dims[1], p.shape[1])

# Pad the arrays to the maximum dimensions
padded_points = []
for p in points:
    pad_width = [(0, max_dims[0] - p.shape[0]), (0, max_dims[1] - p.shape[1])]
    padded_p = np.pad(p, pad_width, 'constant')
    padded_points.append(padded_p)

# Now try to create the numpy array
points_array = np.array(padded_points)

# Assuming you have the 'points' array from the previous code
points_2d = keypoints.orthographic_projection(points)

# Calculate the min and max for x and y values from points_2d
x_values_raw = [point[0] for point in points_2d]
y_values_raw = [point[1] for point in points_2d]

min_x, max_x = min(x_values_raw), max(x_values_raw)
min_y, max_y = min(y_values_raw), max(y_values_raw)

# Define the new range
new_min = 0
new_max = 10

# Now you can apply the scaling to the x and y values in points_2d
x_values = [keypoints.scale_values(point[0], min_x, max_x, new_min, new_max) for point in points_2d]
y_values = [keypoints.scale_values(point[1], min_y, max_y, new_min, new_max) for point in points_2d]

# Retrieve all points from the scatter plot
points = list(zip(x_values, y_values))

# Sample scatter plot data
scatter_data = points

# Define the grid size
grid_size = 10
map_grid = np.zeros((grid_size, grid_size))

# Populate the grid
for point in scatter_data:
    x, y = point
    grid_x = int(x)
    grid_y = int(y)
    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        map_grid[grid_y, grid_x] = 1  # Set point to black

# Create the black and white map
bw_map = np.where(map_grid > 0, 1, 0)
rotated_bw_map = np.rot90(bw_map, 2)  # Rotate 180 degrees using np.rot90
flipped_rotated_bw_map = np.fliplr(rotated_bw_map)

print(flipped_rotated_bw_map)

# Define the matrix
matrix = np.array(flipped_rotated_bw_map)

# Convert the NumPy array to a DataFrame (pandas is better for CSV operations)
df = pd.DataFrame(matrix)

# Specify S3 bucket name and the file path for the CSV
bucket_name = os.getenv("SIMULATION_BUCKET_NAME")  # Replace with your actual bucket name
file_path = 'map/map.csv' # Replace 'folder_name/' with your desired path in the bucket

# Initialize an in-memory file-like buffer for the CSV data
csv_buffer = StringIO()

# Save DataFrame to CSV format in-memory (no header or index)
df.to_csv(csv_buffer, index=False, header=False)

# Initialize S3 client
s3 = boto3.client('s3')

# Upload the CSV directly from memory to S3
s3.put_object(Bucket=bucket_name, Key=file_path, Body=csv_buffer.getvalue())
print("CSV file successfully uploaded to S3.")
