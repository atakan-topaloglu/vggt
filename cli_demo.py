import os
import glob
import argparse
import torch
import torch.nn as nn # Added for nn.DataParallel
import numpy as np
from tqdm.auto import tqdm
import cv2

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation will not work.")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import predictions_to_glb, download_file_from_url # segment_sky is used within predictions_to_glb
from vggt.utils.geometry import unproject_depth_map_to_point_map


def main():
    parser = argparse.ArgumentParser(description="VGGT Inference Script")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to the directory to save the GLB file."
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=50.0,
        help="Confidence threshold for filtering points (percentage).",
    )
    parser.add_argument(
        "--remove_sky", action="store_true", help="Whether to remove sky points."
    )
    parser.add_argument(
        "--use_point_map", action="store_true", help="Use point map instead of depth-based points"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading VGGT model from pretrained weights...")
    # Load model to CPU first to ensure correct DataParallel wrapping
    model = VGGT.from_pretrained("facebook/VGGT-1B", map_location='cpu')
    model.eval()
    
    # --- Alternative manual loading (if from_pretrained fails or for custom checkpoints) ---
    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
    # model.load_state_dict(state_dict)
    # model.eval()
    # ------------------------------------------------------------------------------------

    if device == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference via DataParallel.")
        model = nn.DataParallel(model) # Wrap model for multi-GPU
    
    model = model.to(device) # Move the (potentially wrapped) model to the target device
   
    print(f"Loading images from {args.image_dir}...")
    # Corrected image path: assumes args.image_dir is the folder containing images
    image_names = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    print(f"Found {len(image_names)} images")

    if not image_names:
        print(f"No images found in {args.image_dir}. Exiting.")
        return

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    # Determine dtype for autocast based on CUDA device capability (if CUDA is used)
    dtype = torch.float16 
    if device == "cuda" and torch.cuda.is_available():
        if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8: # Ampere or newer for bfloat16
            dtype = torch.bfloat16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype): # Enable autocast only for CUDA
            predictions = model(images)

    # Re-collation logic for DataParallel output
    if isinstance(model, nn.DataParallel):
        print("Re-collating outputs from DataParallel.")
        re_collated_predictions = {}
        num_gpus_used = torch.cuda.device_count() # Number of GPUs DataParallel actually used
        
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == num_gpus_used and value.dim() > 1:
                # This tensor was gathered by DataParallel from multiple GPUs.
                # Original shape from each replica: (1, S_chunk, *dims)
                # Gathered shape by DataParallel: (num_gpus_used, S_chunk, *dims)
                # We need to reshape it to (1, S_total, *dims)
                s_chunk = value.shape[1] 
                s_total = num_gpus_used * s_chunk
                
                re_collated_predictions[key] = value.contiguous().view(1, s_total, *value.shape[2:])
            else:
                re_collated_predictions[key] = value
        predictions = re_collated_predictions

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:] # images.shape[-2:] is (H_original, W_original)
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            # remove batch dimension (which should be 1 after re-collation) and convert to numpy
            predictions[key] = predictions[key].cpu().numpy().squeeze(0) 

    # Create world points from depth
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    print(f"Using {'Point Map' if args.use_point_map else 'Depth from camera'} for world points")

    print("Generating GLB file...")
    scene_3d = predictions_to_glb(
        predictions,
        conf_thres=args.conf_threshold,
        target_dir=args.image_dir,  # Pass the image directory to predictions_to_glb
        mask_sky=args.remove_sky, #Enable sky removal
        prediction_mode = "Predicted Pointmap" if args.use_point_map else "Depthmap and Camera"
    )
    output_path = os.path.join(args.output_dir, "scene.glb")
    scene_3d.export(output_path)
    print(f"Saved GLB file to {output_path}")

    print("Inference complete.")


if __name__ == "__main__":
    main()