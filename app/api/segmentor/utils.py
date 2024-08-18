from torchvision import transforms
import numpy as np
import cv2

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