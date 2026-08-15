import os
import argparse
import numpy as np
import torch
import sys

# Ensure we can import from src directory
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.resunet_advanced import AdvancedResUNet

def load_model(weights_path, device):
    """Loads the frozen E06-D model."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
    model = AdvancedResUNet(use_se=True, use_dilated_bottleneck=True)
    
    # Verify parameter count to ensure it's exactly E06-D
    num_params = sum(p.numel() for p in model.parameters())
    if num_params != 1026766:
        raise ValueError(f"CRITICAL ERROR: Loaded architecture has {num_params} parameters instead of the expected 1,026,766 for E06-D.")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(img_path, model, device):
    """Processes a single raw .npy float32 image."""
    # Load raw input (expected shape: [H, W] or [1, H, W] or [1, 1, H, W])
    img_np = np.load(img_path)
    
    # Ensure shape is [1, 1, H, W] for the model
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, np.newaxis, :, :]
    elif img_np.ndim == 3:
        img_np = img_np[np.newaxis, :, :, :]
        
    # Maintain raw float32 representation
    img_tensor = torch.from_numpy(img_np).float().to(device)
    
    with torch.no_grad():
        pred_tensor = model(img_tensor)
        
        # Clamp prediction strictly to [0, 1] domain
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
        
    # Convert back to float32 numpy array
    pred_np = pred_tensor.cpu().numpy()
    
    # Squeeze back to [H, W] for output
    return np.squeeze(pred_np)

def save_image(img_np, output_path, out_format):
    """Saves the float32 array in the requested format."""
    if out_format == 'npy':
        # Preserve native float32
        np.save(output_path, img_np)
    else:
        # If competition requires image formats (PNG/TIFF)
        from PIL import Image
        # Scale [0, 1] to [0, 255] uint8
        img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')
        img_pil.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluator for E06-D (KLA Competition)")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to test images directory containing .npy files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output directory where restored images will be saved.")
    parser.add_argument('--output_format', type=str, default='npy', choices=['npy', 'png', 'tif'], help="Format to save the restored outputs. Default is npy to preserve float32.")
    parser.add_argument('--weights', type=str, default=os.path.join(project_root, 'weights', 'best_model.pth'), help="Path to E06-D checkpoint.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading E06-D model...")
    model = load_model(args.weights, device)
    
    valid_extensions = ('.npy',)
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(valid_extensions)]
    
    if len(input_files) == 0:
        print(f"Warning: No valid test files found in {args.input_dir}")
        return
        
    print(f"Found {len(input_files)} test files. Beginning inference...")
    
    for filename in input_files:
        in_path = os.path.join(args.input_dir, filename)
        
        # Determine output filename
        base_name = os.path.splitext(filename)[0]
        out_filename = f"{base_name}.{args.output_format}"
        out_path = os.path.join(args.output_dir, out_filename)
        
        # Predict
        restored_np = process_image(in_path, model, device)
        
        # Save
        save_image(restored_np, out_path, args.output_format)
        
    print(f"Inference complete. Restored outputs saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
