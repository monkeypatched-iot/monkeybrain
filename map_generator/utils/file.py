from datetime import datetime
import zipfile
import os

class FileUtils:
    def extract_timestamp_from_filename(self,filename):
        # Remove the prefix (e.g., 'rgb_image_') and split by space to get the timestamp part
        timestamp_str = filename.split('_')[2]  # Get the second part which is the timestamp

        # Convert to a datetime object
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

        # return timestamp
        return timestamp
    
    def unzip_file(self,zip_file_path, extract_to_folder):
        # Create the output directory if it doesn't exist
        os.makedirs(extract_to_folder, exist_ok=True)

        # Open the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all the contents into the specified folder
            zip_ref.extractall(extract_to_folder)
            print(f'Extracted all files to: {extract_to_folder}')