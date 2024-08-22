import torch
import torch.nn as nn
from torchvision import models, transforms

class ModifiedDeepLabV3(nn.Module):
    """
    Modified pre-trained DeepLabV3 model with a ResNet-101 backbone for binary segmentation (Fine-Tuning).
    """
    def __init__(self):
        super(ModifiedDeepLabV3, self).__init__()
        # Load pre-trained DeepLabV3
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        
        # Modify classifier for binary segmentation
        self.model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, 1)
                                            
        # Remove the auxiliary classifier layers (we only need the Primary output)
        self.model.aux_classifier = None
                                            
        # Add Sigmoid for binary segmentation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the modified model to obtain the output mask.
        
        Args:
            x (torch.Tensor): Input tensor of images.
            
        Returns:
            x (torch.Tensor): Output mask after applying Sigmoid activation for binary classification.
        """
        x = self.model(x)['out'] # Get primary output
        x = self.sigmoid(x)  # Apply Sigmoid
        return x

def load_model(weights_path, device):
    """
    Initialize the model, load the weights, and move it to the specified device.
    
    Args:
        weights_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model onto (CPU or GPU).
    
    Returns:
        nn.Module: The loaded model.
    """
    model = ModifiedDeepLabV3()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model

def predict(model, image_tensor, device, threshold=0.5):
    """
    Make a prediction on the preprocessed image tensor.
    
    Args:
        model (nn.Module): The loaded model.
        image_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W).
        device (torch.device): Device to perform the prediction on (CPU or GPU).
    
    Returns:
        torch.Tensor: The predicted binary mask (1, H, W).
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = (output > threshold).float()
        pred_mask = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Remove the batch dimension and move to CPU

    return pred_mask

# Example usage
if __name__ == "__main__":
    # Path to the saved model weights
    weights_path = "app\\api\\segmentor\\DeepLabV3-Model-V1.0.pth.tar"
    
    # Initialize the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(weights_path, device)
    
    # Example image tensor (this should come from your preprocessing step)
    example_image_tensor = torch.rand((1, 3, 256, 256))  # Dummy tensor for example
    
    # Perform prediction
    prediction = predict(model, example_image_tensor, device)
    
    # The prediction is now a tensor with the predicted mask
    print(prediction.shape)  # Should output: torch.Size([1, 256, 256])
