from PIL import Image
from torchvision import transforms
import numpy as np

def process_image(image_path):
    """Adjusts a PIL image for input into a PyTorch model, 
       returning a Numpy array.
    """
    
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Define the transformations: resize, crop, convert to tensor, and normalize
    transformation_pipeline = transforms.Compose([
        transforms.Resize(226),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply the transformations
    image_tensor = transformation_pipeline(image)
    
    # Convert the tensor to a Numpy array
    processed_image = np.array(image_tensor)
    
    # Rearrange dimensions to match (C, H, W) format
    processed_image = processed_image.transpose((1, 2, 0))
    
    return processed_image
