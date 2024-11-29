import torch
import cv2
import numpy as np
from lib.loftr.src.loftr import LoFTR, default_cfg
import matplotlib.pyplot as plt

class ImageUtils:
    def set_image_type(self, image_type):
        """
        Set the environment type for LoFTR
        1. image_type: indoor or outdoor
        """

        # Assuming LoFTR and config are defined somewhere in your code
        matcher = LoFTR(config=default_cfg)

        # Load model weights based on the image type
        if image_type == 'indoor':
            matcher.load_state_dict(torch.load("weights/indoor_ds.ckpt", weights_only=True)['state_dict'])
        elif image_type == 'outdoor':
            matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt", weights_only=True)['state_dict'])
        else:
            raise ValueError("Wrong image_type is given.")

        # Determine the device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the selected device (either CPU or GPU)
        matcher = matcher.eval().to(device)

        return matcher

    def process_frames(self, frame, frame2):
        """Convert frames to grayscale and resize."""
        frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(cv2.resize(frame2, (640, 480)), cv2.COLOR_BGR2GRAY)
        return frame_gray, frame2_gray

    def convert_to_tensors(self, frame_gray, frame2_gray):
        """Convert images to PyTorch tensors."""
        
        # Determine device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the numpy arrays to PyTorch tensors and then to the appropriate device
        image1 = torch.from_numpy(frame_gray)[None][None].to(device) / 255.
        image2 = torch.from_numpy(frame2_gray)[None][None].to(device) / 255.

        return image1, image2

    def perform_loftr_inference(self, matcher, image1, image2):
        """Run LoFTR inference and return average matching confidence."""
        with torch.no_grad():
            batch = {'image0': image1, 'image1': image2}
            matcher(batch)
            mconf = batch['mconf'].cpu().numpy()
            return np.mean(mconf) if mconf.size > 0 else None

    def display_frames(self, frame1, frame2):
        """Display two frames side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB))
        axes[0].axis('off')
        axes[0].set_title('Frame 1')
        axes[1].imshow(cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB))
        axes[1].axis('off')
        axes[1].set_title('Frame 2')
        plt.show()
