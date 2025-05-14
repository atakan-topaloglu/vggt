import os
import argparse
import numpy as np
import torch
import glob
import struct
from scipy.spatial.transform import Rotation
import sys
from PIL import Image
import cv2
import requests
import tempfile
import logging # For suppressing DINOv2/xFormers logs
import warnings # For suppressing DINOv2/xFormers logs

sys.path.append(".") # Assuming run from root, so evaluation.ba can be found

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3 # Import closed_form_inverse_se3
from evaluation.ba import run_vggt_with_ba # Import the BA function

# Suppress DINO v2 logs and other warnings
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available") # From vggt.layers.block
warnings.filterwarnings("ignore", message="xFormers is not available") # From vggt.layers.swiglu_ffn
warnings.filterwarnings("ignore", message="dinov2") # From evaluation.test_co3d

# Set computation precision (optional, but good practice from test_co3d.py)
torch.set_float32_matmul_precision('highest')
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = False


def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Using from_pretrained for simplicity, adjust if using a local .pt file for BA
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    
    model.eval()
    model = model.to(device)
    return model, device

def process_images(image_dir, model, device):
    """Process images with VGGT and return predictions and the image tensor."""
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_names)} images")
    
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")

    original_images_pil = [] # Store PIL images for original dimensions if needed
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images_pil.append(img)
    
    # `images_tensor` is the preprocessed tensor for the model
    images_tensor = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images tensor shape: {images_tensor.shape}")
    
    print("Running initial VGGT inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Pass images_tensor which has batch dim already
            predictions = model(images_tensor if images_tensor.ndim == 4 else images_tensor.unsqueeze(0))


    print("Converting pose encoding to initial camera parameters...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors in predictions to numpy arrays and remove batch dim
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0) 
    
    print("Computing initial 3D points from depth maps...")
    depth_map = predictions["depth"] 
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Store original images (resized to match depth map resolution for coloring points)
    S_pred, H_pred, W_pred = world_points.shape[:3] # S, H, W from prediction output
    
    # Use PIL images for original dimensions for COLMAP camera.txt, then resize for coloring
    original_images_for_coloring = []
    for pil_img in original_images_pil:
        # Resize to match the H_pred, W_pred for consistent coloring later
        resized_img_np = np.array(pil_img.resize((W_pred, H_pred), Image.Resampling.LANCZOS))
        original_images_for_coloring.append(resized_img_np)

    # `predictions["images"]` should be the color source for 3D points, matching H_pred, W_pred
    predictions["images"] = (np.stack(original_images_for_coloring) / 255.0).astype(np.float32)
    
    # Store original image dimensions for COLMAP cameras.txt
    # Assuming all original images intended for COLMAP have same original size,
    # or COLMAP handles varying sizes if per-image camera models are written.
    # For simplicity, using the first image's original dimensions.
    original_H, original_W = original_images_pil[0].height, original_images_pil[0].width
    predictions["original_image_dimensions"] = (original_W, original_H) # W, H for COLMAP

    return predictions, image_names, images_tensor # Return the images_tensor for BA

def extrinsic_to_colmap_format(extrinsics_cam_from_world):
    """
    Convert extrinsic matrices (camera_from_world) to COLMAP format (world_from_camera: quaternion + translation).
    Input: extrinsics_cam_from_world (N, 3, 4) - R_cam_from_world | t_cam_from_world
    Output: COLMAP quaternions (N, 4) as [qw, qx, qy, qz], COLMAP translations (N, 3) as t_world_from_cam
    """
    num_cameras = extrinsics_cam_from_world.shape[0]
    colmap_quaternions = []
    colmap_translations = []
    
    for i in range(num_cameras):
        R_cam_from_world = extrinsics_cam_from_world[i, :3, :3]
        t_cam_from_world = extrinsics_cam_from_world[i, :3, 3]

        # COLMAP expects world_from_camera pose
        R_world_from_cam = R_cam_from_world.T
        t_world_from_cam = -R_world_from_cam @ t_cam_from_world
        
        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R_world_from_cam)
        quat_scipy = rot.as_quat()  # scipy returns [x, y, z, w]
        quat_colmap = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        colmap_quaternions.append(quat_colmap)
        colmap_translations.append(t_world_from_cam)
    
    return np.array(colmap_quaternions), np.array(colmap_translations)


def download_file_from_url(url, filename):
    """Downloads a file from a URL, handling redirects."""
    try:
        response = requests.get(url, allow_redirects=False, timeout=30) # Added timeout
        response.raise_for_status() 

        if response.status_code == 302:  
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True, timeout=30)
            response.raise_for_status()
        # If not 302, and not an error, assume it's a direct download or an error handled by raise_for_status
        elif response.status_code != 200: # Check for other non-successful statuses
             response = requests.get(url, stream=True, timeout=30) # Try again with stream=True for direct download
             response.raise_for_status()


        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    """
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255 

    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
    
    return output_mask

def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.
    """
    import copy
    
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    if max_value > min_value: # Avoid division by zero
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
    else:
        onnx_result = np.zeros_like(onnx_result) # Handle case where all values are the same
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result

def filter_and_prepare_points(predictions, conf_threshold_percent, image_names_for_skymask,
                              mask_sky=False, mask_black_bg=False, 
                              mask_white_bg=False, stride=1, prediction_mode="Depthmap and Camera Branch"):
    """
    Filter points based on confidence and prepare for COLMAP format.
    """
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch for 3D points.")
        pred_world_points = predictions.get("world_points", predictions["world_points_from_depth"])
        pred_world_points_conf = predictions.get("world_points_conf", predictions.get("depth_conf"))
    else:
        print("Using Depthmap and Camera Branch for 3D points.")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    colors_rgb_normalized = predictions["images"] # Should be (S, H, W, 3) in [0,1]
    colors_rgb_uint8 = (colors_rgb_normalized * 255).astype(np.uint8)
    
    S, H, W = pred_world_points.shape[:3]
    assert colors_rgb_normalized.shape == (S, H, W, 3), f"Color map shape {colors_rgb_normalized.shape} mismatch with points {S,H,W,3}"
    
    if mask_sky:
        print("Applying sky segmentation mask...")
        try:
            import onnxruntime
            with tempfile.TemporaryDirectory() as temp_dir:
                sky_masks_dir = os.path.join(temp_dir, "sky_masks") # Not strictly needed if not saving masks
                os.makedirs(sky_masks_dir, exist_ok=True)
                
                skyseg_path = os.path.join(temp_dir, "skyseg.onnx")
                if not os.path.exists("skyseg.onnx"): 
                    print("Downloading skyseg.onnx...")
                    download_success = download_file_from_url(
                        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
                        skyseg_path
                    )
                    if not download_success:
                        print("Failed to download skyseg model, skipping sky filtering.")
                        mask_sky = False # Disable if download fails
                else: # Use local copy if available
                    import shutil
                    shutil.copy("skyseg.onnx", skyseg_path)
                
                if mask_sky: # Check again in case download failed
                    skyseg_session = onnxruntime.InferenceSession(skyseg_path)
                    sky_mask_list = []
                    
                    for img_path_orig in image_names_for_skymask: # Use original image paths for sky segmentation
                        # We don't save the mask here, just use it
                        sky_mask_for_frame = segment_sky(img_path_orig, skyseg_session, mask_filename=None) 
                        
                        if sky_mask_for_frame.shape[0] != H or sky_mask_for_frame.shape[1] != W:
                            sky_mask_for_frame = cv2.resize(sky_mask_for_frame, (W, H), interpolation=cv2.INTER_NEAREST)
                        
                        sky_mask_list.append(sky_mask_for_frame)
                    
                    sky_mask_array = np.stack(sky_mask_list, axis=0) # (S, H, W)
                    sky_mask_binary = (sky_mask_array > 128).astype(np.float32) # Sky is 0, non-sky is 1 (after inversion in segment_sky)
                                                                            # Here we expect sky_mask_binary to be 1 for non-sky, 0 for sky.
                                                                            # segment_sky returns 255 for non-sky, 0 for sky. So >128 works.
                    pred_world_points_conf = pred_world_points_conf * sky_mask_binary 
                    print(f"Applied sky mask, shape: {sky_mask_binary.shape}. Non-sky area ratio: {sky_mask_binary.mean():.3f}")

        except (ImportError, Exception) as e:
            print(f"Sky segmentation failed or onnxruntime not found: {e}. Skipping sky filtering.")
            mask_sky = False
    
    vertices_3d_flat = pred_world_points.reshape(-1, 3)
    conf_flat = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb_uint8.reshape(-1, 3)
    
    if len(conf_flat) != len(colors_rgb_flat) or len(conf_flat) != len(vertices_3d_flat):
        print(f"WARNING: Shape mismatch. Conf: {len(conf_flat)}, Colors: {len(colors_rgb_flat)}, Vertices: {len(vertices_3d_flat)}")
        min_len = min(len(conf_flat), len(colors_rgb_flat), len(vertices_3d_flat))
        conf_flat = conf_flat[:min_len]
        vertices_3d_flat = vertices_3d_flat[:min_len]
        colors_rgb_flat = colors_rgb_flat[:min_len]
    
    if conf_threshold_percent == 0.0:
        conf_thres_value = -np.inf # Include all points if threshold is 0
    else:
        conf_thres_value = np.percentile(conf_flat[conf_flat > 1e-5], conf_threshold_percent) # Percentile on valid confs
    
    print(f"Using confidence threshold: {conf_threshold_percent}% (value: {conf_thres_value:.4f})")
    
    # Initial confidence mask
    valid_conf_mask = (conf_flat >= conf_thres_value) & (conf_flat > 1e-5) # Basic confidence filter
    
    if mask_black_bg:
        print("Filtering black background points...")
        # Consider black if sum of RGB is low, e.g., < 3*10 = 30
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 30 
        valid_conf_mask = valid_conf_mask & black_bg_mask
    
    if mask_white_bg:
        print("Filtering white background points...")
        # Consider white if all RGB values are high, e.g., > 245
        white_bg_mask = ~((colors_rgb_flat[:, 0] > 245) & (colors_rgb_flat[:, 1] > 245) & (colors_rgb_flat[:, 2] > 245))
        valid_conf_mask = valid_conf_mask & white_bg_mask
    
    # Apply stride to the valid_conf_mask indices before creating points3D entries
    # This part needs careful handling to map strided 2D points to their 3D counterparts and observations.
    
    points3D = []
    point_hash_to_idx = {} # Using a dictionary to map a quantized 3D point to its index in points3D list
    image_points2D = [[] for _ in range(S)] # List of lists to store (x, y, point3D_idx) for each image

    print(f"Preparing points for COLMAP format with stride {stride}...")
    num_valid_points_before_stride = np.sum(valid_conf_mask)
    
    current_point3D_idx = 0
    for s_idx in range(S):
        for r_idx in range(0, H, stride):
            for c_idx in range(0, W, stride):
                flat_idx = s_idx * H * W + r_idx * W + c_idx
                if flat_idx >= len(valid_conf_mask): continue

                if valid_conf_mask[flat_idx]:
                    point_xyz = vertices_3d_flat[flat_idx]
                    point_rgb = colors_rgb_flat[flat_idx]

                    if not np.all(np.isfinite(point_xyz)): continue # Skip non-finite points

                    # Quantize point_xyz to handle minor floating point variations for hashing
                    quantized_xyz_key = tuple(np.round(point_xyz * 1000).astype(int)) # Scale by 1000 before rounding

                    if quantized_xyz_key not in point_hash_to_idx:
                        point_hash_to_idx[quantized_xyz_key] = current_point3D_idx
                        points3D.append({
                            "id": current_point3D_idx,
                            "xyz": point_xyz,
                            "rgb": point_rgb,
                            "error": 1.0, # Default error
                            "track": [] 
                        })
                        current_point3D_idx += 1
                    
                    # Add observation: (image_id, point2D_idx in that image's observation list)
                    # COLMAP point2D_idx is specific to the image, not global.
                    # Here, (c_idx, r_idx) are the 2D coordinates in the (potentially downsampled) feature map.
                    # For COLMAP, these should be pixel coordinates in the *original* image if possible,
                    # or at least consistent. `load_and_preprocess_images` standardizes output size.
                    # `filter_and_prepare_points` gets H, W from `pred_world_points`, which matches `images_tensor`.
                    # So (c_idx, r_idx) are pixel coords in the processed image.
                    p3d_idx_for_obs = point_hash_to_idx[quantized_xyz_key]
                    points3D[p3d_idx_for_obs]["track"].append((s_idx, len(image_points2D[s_idx])))
                    image_points2D[s_idx].append((c_idx, r_idx, p3d_idx_for_obs))
    
    # Filter out 3D points that don't have at least 2 observations (COLMAP requirement for triangulation)
    final_points3D = []
    final_point_map = {} # old_idx -> new_idx
    new_idx_counter = 0
    for old_idx, p_entry in enumerate(points3D):
        if len(p_entry["track"]) >= 2:
            final_point_map[old_idx] = new_idx_counter
            p_entry["id"] = new_idx_counter # Update ID to be sequential for final list
            final_points3D.append(p_entry)
            new_idx_counter +=1

    # Update point3D_idx in image_points2D
    final_image_points2D = [[] for _ in range(S)]
    for img_idx, observations in enumerate(image_points2D):
        for x, y, old_p3d_idx in observations:
            if old_p3d_idx in final_point_map:
                new_p3d_idx = final_point_map[old_p3d_idx]
                final_image_points2D[img_idx].append((x,y,new_p3d_idx))

    print(f"Total valid points before stride: {num_valid_points_before_stride}")
    print(f"Prepared {len(final_points3D)} unique 3D points with at least 2 observations.")
    print(f"Total 2D observations: {sum(len(obs) for obs in final_image_points2D)}")
    
    return final_points3D, final_image_points2D


def hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)

def write_colmap_cameras_txt(file_path, intrinsics_batch, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics_batch)}\n")
        
        for i, intrinsic_matrix in enumerate(intrinsics_batch):
            camera_id = i + 1 
            model = "PINHOLE" 
            
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            
            f.write(f"{camera_id} {model} {image_width} {image_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_colmap_images_txt(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        num_total_observations = sum(len(points) for points in image_points2D)
        avg_points = num_total_observations / len(image_points2D) if image_points2D and len(image_points2D) > 0 else 0
        f.write(f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n")
        
        for i in range(len(quaternions)):
            image_id = i + 1 
            camera_id = i + 1 # Assuming one camera model per image, sequentially
          
            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]
            
            # Use only basename for image name in COLMAP file
            img_name_for_colmap = os.path.basename(image_names[i])
            f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {camera_id} {img_name_for_colmap}\n")
            
            # POINT3D_ID in image_points2D is 0-indexed from our prep, COLMAP is 1-indexed for POINT3D_ID
            points_line = " ".join([f"{x_coord:.2f} {y_coord:.2f} {p3d_idx+1}" for x_coord, y_coord, p3d_idx in image_points2D[i]])
            f.write(f"{points_line}\n")


def write_colmap_points3D_txt(file_path, points3D_list):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        avg_track_length = sum(len(point["track"]) for point in points3D_list) / len(points3D_list) if points3D_list else 0
        f.write(f"# Number of points: {len(points3D_list)}, mean track length: {avg_track_length:.4f}\n")
        
        for point_entry in points3D_list:
            # POINT3D_ID is 1-indexed in COLMAP text files
            point_id_colmap = point_entry["id"] + 1 
            x, y, z = point_entry["xyz"]
            r_val, g_val, b_val = point_entry["rgb"]
            error_val = point_entry["error"]
            
            # IMAGE_ID in track is 1-indexed, POINT2D_IDX is 0-indexed within the image's observation list
            track_str_parts = []
            for img_idx_0based, point2d_idx_0based in point_entry["track"]:
                track_str_parts.append(f"{img_idx_0based + 1}") # IMAGE_ID
                track_str_parts.append(f"{point2d_idx_0based}") # POINT2D_IDX
            track_str = " ".join(track_str_parts)
            
            f.write(f"{point_id_colmap} {x:.6f} {y:.6f} {z:.6f} {int(r_val)} {int(g_val)} {int(b_val)} {error_val:.6f} {track_str}\n")


def write_colmap_cameras_bin(file_path, intrinsics_batch, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.bin format."""
    with open(file_path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(intrinsics_batch))) # num_cameras
        for i, intrinsic_matrix in enumerate(intrinsics_batch):
            camera_id = i + 1 
            model_id = 1 # PINHOLE model
            
            fx = float(intrinsic_matrix[0, 0])
            fy = float(intrinsic_matrix[1, 1])
            cx = float(intrinsic_matrix[0, 2])
            cy = float(intrinsic_matrix[1, 2])
            
            fid.write(struct.pack('<i', camera_id)) # camera_id (int32)
            fid.write(struct.pack('<i', model_id))  # model_id (int32)
            fid.write(struct.pack('<Q', image_width)) # width (uint64)
            fid.write(struct.pack('<Q', image_height))# height (uint64)
            fid.write(struct.pack('<dddd', fx, fy, cx, cy)) # params (double)

def write_colmap_images_bin(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.bin format."""
    with open(file_path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(quaternions))) # num_reg_images
        
        for i in range(len(quaternions)):
            image_id = i + 1 
            camera_id = i + 1 
            
            qw, qx, qy, qz = quaternions[i].astype(float)
            tx, ty, tz = translations[i].astype(float)
            
            image_name_bytes = os.path.basename(image_names[i]).encode('utf-8') + b'\x00' # Null-terminated
            observations = image_points2D[i] # List of (x, y, point3D_id_0based)
            
            fid.write(struct.pack('<i', image_id)) # image_id (int32)
            fid.write(struct.pack('<dddd', qw, qx, qy, qz)) # qvec (double)
            fid.write(struct.pack('<ddd', tx, ty, tz))    # tvec (double)
            fid.write(struct.pack('<i', camera_id))     # camera_id (int32)
            fid.write(image_name_bytes)                 # name (variable length)
            
            fid.write(struct.pack('<Q', len(observations))) # num_points2D (uint64)
            for x_coord, y_coord, p3d_idx_0based in observations:
                fid.write(struct.pack('<dd', float(x_coord), float(y_coord))) # xy (double)
                fid.write(struct.pack('<Q', p3d_idx_0based + 1)) # point3D_id (uint64, 1-based)

def write_colmap_points3D_bin(file_path, points3D_list):
    """Write 3D points and tracks to COLMAP points3D.bin format."""
    with open(file_path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(points3D_list))) # num_points (uint64)
        
        for point_entry in points3D_list:
            point_id_colmap = point_entry["id"] + 1 # 1-based ID
            x, y, z = point_entry["xyz"].astype(float)
            r_val, g_val, b_val = point_entry["rgb"].astype(np.uint8)
            error_val = float(point_entry["error"])
            track = point_entry["track"] # List of (image_id_0based, point2D_idx_0based)
            
            fid.write(struct.pack('<Q', point_id_colmap)) # point3D_id (uint64)
            fid.write(struct.pack('<ddd', x, y, z))     # xyz (double)
            fid.write(struct.pack('<BBB', int(r_val), int(g_val), int(b_val))) # rgb (uint8)
            fid.write(struct.pack('<d', error_val))      # error (double)
            
            fid.write(struct.pack('<Q', len(track)))     # track_len (uint64)
            for img_idx_0based, point2d_idx_0based in track:
                fid.write(struct.pack('<II', img_idx_0based + 1, point2d_idx_0based)) # image_id, point2D_idx (uint32)

def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT, with optional BA")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="colmap_output", 
                        help="Directory to save COLMAP files")
    parser.add_argument("--conf_threshold", type=float, default=50.0, 
                        help="Confidence threshold (0-100 percentile) for including points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter out points likely to be sky (requires onnxruntime and skyseg.onnx)")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter out points with very dark/black color")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter out points with very bright/white color")
    parser.add_argument("--binary", action="store_true", 
                        help="Output binary COLMAP files instead of text")
    parser.add_argument("--stride", type=int, default=1, 
                        help="Stride for point sampling (higher = fewer points, e.g., 2 means sample every 2nd pixel)")
    parser.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Which VGGT prediction branch to use for 3D points")
    parser.add_argument("--use_ba", action="store_true",
                        help="Enable Bundle Adjustment to refine camera poses.")
    # Add BA specific arguments if needed, e.g., path to BA-specific model
    parser.add_argument('--ba_model_path', type=str, default=None,
                        help='Path to the VGGT model checkpoint specifically for BA (e.g., tracker_fixed). Defaults to main model.')


    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_load_path = args.ba_model_path if args.use_ba and args.ba_model_path else "facebook/VGGT-1B" # Default or BA specific
    
    if args.use_ba and args.ba_model_path:
        print(f"Loading BA-specific model from: {args.ba_model_path}")
        # Assuming local .pt loading logic for BA model if path is given
        device_for_model = "cuda" if torch.cuda.is_available() else "cpu"
        model = VGGT() 
        model.load_state_dict(torch.load(args.ba_model_path, map_location=torch.device(device_for_model)))
        model.eval()
        model = model.to(device_for_model)
        device = device_for_model
    else:
        model, device = load_model() # Loads from_pretrained by default now


    predictions, image_names, images_tensor_for_ba = process_images(args.image_dir, model, device)
    
    if args.use_ba:
        print("Running VGGT with Bundle Adjustment...")
        dtype_for_ba = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        try:
            # run_vggt_with_ba expects images tensor (S, C, H, W)
            # images_tensor_for_ba is (B, S, C, H, W) or (S,C,H,W) if B=1 after load_and_preprocess
            if images_tensor_for_ba.ndim == 4: # If S,C,H,W -> S,C,H,W (expected by ba.py)
                 ba_input_images = images_tensor_for_ba
            elif images_tensor_for_ba.ndim == 5 and images_tensor_for_ba.shape[0] == 1: # B,S,C,H,W with B=1 -> S,C,H,W
                 ba_input_images = images_tensor_for_ba.squeeze(0)
            else:
                raise ValueError(f"Unexpected shape for images_tensor_for_ba: {images_tensor_for_ba.shape}")

            pred_extrinsic_ba = run_vggt_with_ba(model, ba_input_images, image_names=image_names, dtype=dtype_for_ba)
            
            print("Updating predictions with BA-refined extrinsics...")
            predictions["extrinsic"] = pred_extrinsic_ba.cpu().numpy() # This is (S, 3, 4)
            
            # Recompute world points using BA-refined extrinsics and original depth
            print("Re-computing 3D points from depth maps using BA-refined extrinsics...")
            depth_map_np = predictions["depth"] # Already numpy (S, H, W, 1) or (S, H, W)
            intrinsic_np = predictions["intrinsic"] # Already numpy (S, 3, 3)
            
            world_points_ba = unproject_depth_map_to_point_map(
                depth_map_np, predictions["extrinsic"], intrinsic_np
            )
            predictions["world_points_from_depth"] = world_points_ba
            print("3D points recomputed.")

        except Exception as e:
            print(f"Bundle Adjustment failed: {e}. Proceeding with initial VGGT poses.")
            # Fallback: predictions["extrinsic"] remains the initial VGGT estimate
            # world_points_from_depth based on initial extrinsics is already in predictions


    print("Converting camera parameters to COLMAP format...")
    # Ensure predictions["extrinsic"] is (S, 3, 4)
    # Note: extrinsic_to_colmap_format expects camera_from_world, VGGT provides this.
    # COLMAP format itself is world_from_camera, the conversion handles this.
    quaternions_colmap, translations_colmap = extrinsic_to_colmap_format(predictions["extrinsic"])
    
    print(f"Filtering points with confidence threshold {args.conf_threshold}% and stride {args.stride}...")
    points3D_colmap, image_points2D_colmap = filter_and_prepare_points(
        predictions, 
        args.conf_threshold, 
        image_names_for_skymask=image_names, # Pass original image names for skymask
        mask_sky=args.mask_sky, 
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        stride=args.stride,
        prediction_mode=args.prediction_mode
    )
    
    # Get original image dimensions for COLMAP cameras file
    # This uses the W, H from the *original* images, not the processed ones.
    original_width, original_height = predictions["original_image_dimensions"]

    print(f"Writing {'binary' if args.binary else 'text'} COLMAP files to {args.output_dir}...")
    # Intrinsics from predictions are based on the processed image size.
    # If COLMAP is to be run on original images, intrinsics might need scaling.
    # Here, we assume COLMAP will use images of the size VGGT processed (or we provide images of that size to COLMAP).
    # For cameras.txt/bin, width/height should be of the images COLMAP will see.
    # If using original images with COLMAP, and VGGT processed resized ones, this needs care.
    # The current `load_and_preprocess_images` and `process_images` makes `predictions["intrinsic"]`
    # correspond to `images_tensor` (e.g., 518x518 or 518xH_new).
    # If you provide original images to COLMAP, you must provide original intrinsics.
    # Simplest: give COLMAP the *same images VGGT processed*.
    # The current `original_width, original_height` are from the *first original image*.
    # This is suitable if you intend to use original images with COLMAP and assume all have same original size.
    # If images are resized by `load_and_preprocess_images`, then `predictions["intrinsic"]` matches that resized H, W.
    # Let's use the H, W from the processed predictions for intrinsics consistency.
    _, processed_H, processed_W = predictions["depth"].shape[:3] # H, W of the depth map / processed images

    if args.binary:
        write_colmap_cameras_bin(
            os.path.join(args.output_dir, "cameras.bin"), 
            predictions["intrinsic"], processed_W, processed_H) # Use processed W, H
        write_colmap_images_bin(
            os.path.join(args.output_dir, "images.bin"), 
            quaternions_colmap, translations_colmap, image_points2D_colmap, image_names)
        write_colmap_points3D_bin(
            os.path.join(args.output_dir, "points3D.bin"), 
            points3D_colmap)
    else:
        write_colmap_cameras_txt(
            os.path.join(args.output_dir, "cameras.txt"), 
            predictions["intrinsic"], processed_W, processed_H) # Use processed W, H
        write_colmap_images_txt(
            os.path.join(args.output_dir, "images.txt"), 
            quaternions_colmap, translations_colmap, image_points2D_colmap, image_names)
        write_colmap_points3D_txt(
            os.path.join(args.output_dir, "points3D.txt"), 
            points3D_colmap)
    
    print(f"COLMAP files successfully written to {args.output_dir}")
    print("To use with COLMAP GUI: create an empty database.db, then 'File > Import model' and select the folder.")
    print("You'll also need to provide the actual image files to COLMAP, matching the names in images.txt/bin.")

if __name__ == "__main__":
    main()