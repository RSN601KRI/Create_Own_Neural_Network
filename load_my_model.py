import torch
from my_model import create_nn_model  # Ensure this function is correctly defined in 'my_model.py'

def load_my_checkpoint(filepath, gpu=False):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location="cuda" if gpu else "cpu")
    
    # Retrieve architecture and other parameters
    my_arch = checkpoint['architecture']
    hidden = checkpoint['hidden']
    lr = checkpoint['learning_rate']
    
    # Create the model
    model, criterion, optimizer = create_nn_model(my_arch, hidden, lr, gpu)

    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model, optimizer, criterion  # Ensure you return the model
