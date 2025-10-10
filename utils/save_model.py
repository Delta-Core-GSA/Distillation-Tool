import os
import torch


def save_model_to_pt(model, output_path):
    """
    Save PyTorch model to .pt file.
    
    Args:
        model: PyTorch model to save
        output_path: Path where the model will be saved (should end with .pt)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model, output_path)
    print("Model conversion to .pt completed")


def save_model(model, output_path):
    """
    Save PyTorch model with path validation.
    Ensures output path ends with .pt extension and directory exists.
    
    Args:
        model: PyTorch model to save
        output_path: Path where the model will be saved (must end with .pt)
        
    Raises:
        ValueError: If output_path does not end with .pt extension
    """
    # Ensure parent directory exists
    dir_name = os.path.dirname(output_path)
    os.makedirs(dir_name, exist_ok=True)

    # Validate output path has correct extension
    if not output_path.endswith(".pt"):
        raise ValueError(f"Invalid output path: {output_path} (missing .pt extension?)")

    print(f"Saving model to: {output_path}")
    torch.save(model, output_path)