import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the necessary transformations
class PreprocessImage:
    def __init__(self, resize=(256, 256), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Compose([
            transforms.Resize(resize),          # Resize the image
            transforms.ToTensor(),              # Convert the image to a tensor
            transforms.Normalize(mean, std)     # Normalize the image with mean and std
        ])

    def __call__(self, image_array):
        """
        Preprocess the image array and convert it into a tensor suitable for model input.
        
        Args:
            image_array (numpy.ndarray): Input image array (H, W, C).
        
        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W).
        """
        # Convert the numpy array to a PIL image
        image = Image.fromarray(image_array)
        
        # Apply the transformations
        image_tensor = self.transform(image)
        
        # Add a batch dimension (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor

# Example usage
if __name__ == "__main__":
    # Load your image array (for example, using OpenCV, PIL, or other libraries)
    example_image_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Initialize the preprocessing class
    preprocess = PreprocessImage()
    
    # Preprocess the image
    processed_image_tensor = preprocess(example_image_array)
    
    # The processed_image_tensor is now ready for model input
    print(processed_image_tensor.shape)  # Should output: torch.Size([1, 3, 256, 256])
