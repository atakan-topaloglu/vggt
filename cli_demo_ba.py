# demo_cli_ba.py
import os
import glob
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm # Not used in current script, but can be added for loops
import cv2
import sys # For sys.path
import logging
import warnings

# Ensure vggt and evaluation modules can be found
# Assuming this script is in the root of the vggt repository
sys.path.append(".") 
from PIL import Image

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation will not work if --remove_sky is used.")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb, download_file_from_url, segment_sky # Assuming visual_util.py is in the same dir or accessible
from evaluation.ba import run_vggt_with_ba # Import the BA function

# Suppress DINO v2 logs and other warnings
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision (optional, but good practice)
torch.set_float32_matmul_precision('highest')
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = False


def load_vggt_model(model_path_or_hf_id="facebook/VGGT-1B", device="cuda"):
    """Loads the VGGT model either from Hugging Face or a local .pt file."""
    print(f"Loading VGGT model from: {model_path_or_hf_id} on device: {device}")
    if os.path.exists(model_path_or_hf_id) and model_path_or_hf_id.endswith(".pt"):
        model = VGGT()
        model.load_state_dict(torch.load(model_path_or_hf_id, map_location=torch.device(device)))
    else:
        model = VGGT.from_pretrained(model_path_or_hf_id)
    
    model.eval()
    model = model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="VGGT Inference Script with optional Bundle Adjustment")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to the directory containing images (e.g., examples/kitchen/)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output_cli_ba", help="Path to the directory to save the GLB file."
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=50.0,
        help="Confidence threshold for filtering points for GLB visualization (percentage).",
    )
    parser.add_argument(
        "--remove_sky", action="store_true", help="Whether to remove sky points during GLB generation."
    )
    parser.add_argument(
        "--use_point_map_viz", action="store_true", help="For GLB: Use point map branch output instead of depth-based points."
    )
    
    # BA specific arguments
    parser.add_argument(
        "--use_ba", action="store_true", help="Enable Bundle Adjustment to refine camera poses."
    )
    parser.add_argument(
        '--ba_model_path', type=str, default=None,
        help='Path to VGGT model for BA (e.g., tracker_fixed). Defaults to main model if not set.'
    )
    parser.add_argument(
        '--ba_max_query_num', type=int, default=1024, 
        help='Max keypoints per query frame for BA tracking.'
    )
    parser.add_argument(
        '--ba_det_thres', type=float, default=0.01, 
        help='Detection threshold for BA keypoint extraction.'
    )
    parser.add_argument(
        '--ba_query_frame_num', type=int, default=3, # Default from evaluation/ba.py
        help='Number of query frames for BA keypoint extraction.'
    )
    parser.add_argument(
        '--ba_track_iters', type=int, default=3,
        help='Number of tracking iterations within BA.'
    )
    parser.add_argument(
        '--ba_no_global_attn', action='store_true',
        help='Disable global attention in aggregator during BA tracking to save memory.'
    )
    parser.add_argument(
        '--ba_min_inliers', type=int, default=16,
        help='Minimum number of inlier tracks per frame required for BA to proceed.'
    )


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Determine which model to load for the initial pass
    # If BA is used and a specific BA model is given, the BA function might reload it or use the passed one.
    # For simplicity, we load one model here. If ba_model_path is given, BA will use that.
    # Otherwise, run_vggt_with_ba will use the `model` object passed to it.
    
    main_model_identifier = "facebook/VGGT-1B"
    if args.use_ba and args.ba_model_path:
        print(f"BA is enabled. Main model for initial pass: {main_model_identifier}.")
        print(f"BA will use model from: {args.ba_model_path}.")
        # BA logic will handle its own model loading if ba_model_path is different,
        # or you can pass the loaded main_model to run_vggt_with_ba if paths are same or ba_model_path is None.
        # For this script, let's assume run_vggt_with_ba uses the model object we pass.
        # So, if a specific ba_model_path is given, we load *that* one for everything.
        if args.ba_model_path:
            model = load_vggt_model(args.ba_model_path, device)
        else: # use_ba is true, but no specific ba_model_path, so use default for BA too.
            model = load_vggt_model(main_model_identifier, device)
    else: # Not using BA, or using BA with the default model
        model = load_vggt_model(main_model_identifier, device)
   

    print(f"Loading images from {args.image_dir}...")
    # The example paths in README are like "examples/kitchen/images/"
    # If user gives "examples/kitchen", append "/images"
    image_subdir = "images"
    actual_image_dir = args.image_dir
    if not os.path.basename(args.image_dir).lower() == image_subdir.lower() and \
       os.path.isdir(os.path.join(args.image_dir, image_subdir)):
        actual_image_dir = os.path.join(args.image_dir, image_subdir)
        print(f"Searching for images in subdir: {actual_image_dir}")

    image_names = sorted(glob.glob(os.path.join(actual_image_dir, "*")))
    image_names = [f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(image_names)} images.")

    if not image_names:
        print(f"No images found in {actual_image_dir}. Please check the --image_dir path. Exiting.")
        return

    # images_tensor_for_model is (S, C, H, W) or (B,S,C,H,W) if B=1 from load_and_preprocess
    images_tensor_for_model = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images tensor shape: {images_tensor_for_model.shape}")

    print("Running initial VGGT inference...")
    dtype_inference = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype_inference):
            # model expects batched input
            input_for_model = images_tensor_for_model if images_tensor_for_model.ndim == 4 else images_tensor_for_model.unsqueeze(0)
            predictions = model(input_for_model)

    print("Converting pose encoding to initial extrinsic and intrinsic matrices...")
    # images.shape[-2:] should be from the tensor fed to the model
    extrinsic_init, intrinsic_init = pose_encoding_to_extri_intri(
        predictions["pose_enc"], input_for_model.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic_init
    predictions["intrinsic"] = intrinsic_init

    # Convert all tensor predictions to numpy arrays on CPU, remove batch dim
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0) 
    
    # Compute initial world points (will be recomputed if BA is used)
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    
    # BA step
    if args.use_ba:
        print("-" * 30)
        print("Attempting Bundle Adjustment...")
        try:
            # run_vggt_with_ba expects images tensor (S, C, H, W)
            ba_input_images = images_tensor_for_model # Already (S,C,H,W) if B=1, or needs squeeze if B was added
            if ba_input_images.ndim == 5 and ba_input_images.shape[0] == 1:
                 ba_input_images = ba_input_images.squeeze(0)
            elif ba_input_images.ndim != 4:
                 raise ValueError(f"Unexpected shape for BA input images: {ba_input_images.shape}")

            pred_extrinsic_ba = run_vggt_with_ba(
                model, # Pass the loaded model
                ba_input_images,
                image_names=image_names,
                dtype=dtype_inference, # Use same dtype as main inference for consistency
                max_query_num=args.ba_max_query_num,
                det_thres=args.ba_det_thres,
                query_frame_num=args.ba_query_frame_num,
                camera_type="PINHOLE"
            )
            
            print("BA successful. Updating extrinsics and recomputing 3D points.")
            predictions["extrinsic"] = pred_extrinsic_ba.cpu().numpy() # Update with BA results
            
            # Recompute world_points_from_depth using BA-refined extrinsics
            predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
                predictions["depth"], # Original depth
                predictions["extrinsic"], # BA-refined extrinsics
                predictions["intrinsic"]  # Original intrinsics (BA typically refines extrinsics & 3D pts)
            )
        except Exception as e:
            print(f"Bundle Adjustment failed: {e}. Proceeding with initial VGGT poses.")
            # Predictions will use the initial non-BA poses.
        print("-" * 30)

    # Prepare `images` for predictions_to_glb: (S, H_pred, W_pred, 3) in [0,1]
    # This needs to match the resolution of the depth/point maps.
    # `predictions["depth"]` is (S, H_proc, W_proc, 1)
    S_pred, H_pred, W_pred = predictions["depth"].shape[:3]
    
    # We need original images resized to H_pred, W_pred for coloring
    images_for_glb_coloring_list = []
    for img_name in image_names:
        pil_img = Image.open(img_name).convert("RGB")
        resized_img_np = np.array(pil_img.resize((W_pred, H_pred), Image.Resampling.LANCZOS))
        images_for_glb_coloring_list.append(resized_img_np)
    
    # `predictions["images"]` for `predictions_to_glb`
    predictions["images"] = (np.stack(images_for_glb_coloring_list) / 255.0).astype(np.float32)

    print(f"Using {'Point Map branch output' if args.use_point_map_viz else 'Depth-derived points'} for GLB world points.")
    
    print("Generating GLB file...")
    scene_3d = predictions_to_glb(
        predictions,
        conf_thres=args.conf_threshold,
        target_dir=args.image_dir, # Pass the original image directory for sky segmentation
        mask_sky=args.remove_sky,
        prediction_mode = "Pointmap Branch" if args.use_point_map_viz else "Depthmap and Camera Branch"
    )
    output_path = os.path.join(args.output_dir, "scene_ba.glb" if args.use_ba else "scene_no_ba.glb")
    scene_3d.export(output_path)
    print(f"Saved GLB file to {output_path}")

    print("Inference complete.")

if __name__ == "__main__":
    main()