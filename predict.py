from load_my_model import load_my_checkpoint
from pre_process import process_image
from my_label_mapping import get_label_map

import torch
import torch.nn.functional as F
import argparse
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

def predict_image(image_path, checkpoint_path, category_names, top_k=5, use_gpu=False):
    model = None  # Initialize the model variable

    # Load the model from checkpoint
    try:
        model, _, _ = load_my_checkpoint(checkpoint_path, use_gpu)
        if model is None:
            print("Failed to load model. Exiting...")
            return
    except Exception as e:
        print(f"Error loading the model: {e}")
        return  # Exit if the model cannot be loaded

    # Set the model to evaluation mode
    model.eval()

    # Process the image
    img_tensor = process_image(image_path)

    # Add a batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Move the tensor to GPU if necessary
    if use_gpu:
        img_tensor = img_tensor.to('cuda')

    # Perform prediction
    with torch.no_grad():
        outputs = model(img_tensor)

    # Get top k predictions
    probs = F.softmax(outputs, dim=1)
    top_p, top_class = probs.topk(top_k, dim=1)

    print(f"Top {top_k} probabilities: {top_p}")
    print(f"Top {top_k} classes: {top_class}")

    # Save the model state if it has been defined
    try:
        torch.save(model.state_dict(), 'model.pth')  # Save the model state
        print("Model state saved to 'model.pth'")
    except Exception as e:
        print(f"Error saving the model state: {e}")

# Example usage
# predict_image(image_path='path/to/image.jpg', checkpoint_path='path/to/checkpoint.pth', category_names='path/to/category_names.json')
