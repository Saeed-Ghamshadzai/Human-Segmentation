import os
import cv2
import numpy as np
import base64
from torchvision import transforms
from moviepy.editor import VideoFileClip
from model_pytorch.model import predict
from data_pytorch.preprocessor import PreprocessImage

class InvTransform(object):
    """
    Inverse the standardization applied to images.
    
    Args:
        inv_mean (list of float): Mean values used to denormalize images.
        inv_std (list of float): Standard deviation values used to denormalize images.
    """
    def __init__(self, inverse_mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], inverse_std=[1/0.229, 1/0.224, 1/0.225]):
        # Define the transformer
        self.inverse_transform = transforms.Normalize(
            mean=inverse_mean,
            std=inverse_std,
        )
        
    def __call__(self, image):
        """
        Denormalize the image to its original form.
        
        Args:
            image (torch.Tensor): The normalized image to be denormalized.
            
        Returns:
            image (torch.Tensor): The denormalized image.
        """
        image = self.inverse_transform(image) * 255
        
        return image

def segment_image(model, image_array, device, preprocessor):
    proccessed_image = preprocessor(image_array)
    mask = predict(model, proccessed_image, device) * 255

    return mask

def segment_video(model, video_path, device, processor):
    # Extract video directory, name, and extension
    video_dir, video_name = os.path.split(video_path)
    video_base, video_ext = os.path.splitext(video_name)
    
    # Load the video
    video_clip = VideoFileClip(video_path)
    
    # Get video properties
    fps = video_clip.fps
    width, height = video_clip.size
    audio_clip = video_clip.audio

    # Define the output video path
    segmented_video_path = os.path.join(video_dir, f"{video_base}_segmented{video_ext}")

    # Process each frame to grayscale
    def process_frame(frame):
        inv_transform = InvTransform()

        processed_frame = processor(frame)
        segmented_image = segment_image(model, processed_frame, device)

        original_image = inv_transform(processed_frame.squeeze(0)).permute(1, 2, 0).numpy()
        overlayed_image = apply_mask_overlay(original_image, segmented_image, format='RGB')

        return overlayed_image
    
    # Apply the frame processing
    processed_clip = video_clip.fl_image(process_frame)

    # Write the processed video with the original audio
    processed_clip = processed_clip.set_audio(audio_clip)
    processed_clip.write_videofile(segmented_video_path, codec='libx264', fps=fps)

    print(f"Processed video saved at: {segmented_video_path}")

    return segmented_video_path


def apply_mask_overlay(image, mask, color=(0,0,255), alpha=0.65, format='BGR'):
    """
    Overlays a binary mask on an image with a specified color and transparency.

    Args:
        image (np.ndarray): Images.
        mask (np.ndarray): Segmentation masks.
        color (tuple of int): The color for the mask overlay
        alpha (float): Transparency factor for the overlay.
        
    Returns:
        overlayed_image (np.ndarray): The image with the mask overlay.

    """
    if format == 'RGB':
        color = (255,0,0)
        
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    masked = masked.filled()

    overlayed_image = cv2.addWeighted(image, 1 - alpha, masked, alpha, 0)

    return overlayed_image

def image_to_base64(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str