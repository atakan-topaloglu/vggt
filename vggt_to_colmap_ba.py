# VGGT_to_COLMAP_BA.py
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

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from evaluation.ba import run_vggt_with_ba


def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    
    model.eval()
    model = model.to(device)
    return model, device

def process_images_with_ba(image_dir, model, device, ba_args):
    """Process images with VGGT (with Bundle Adjustment) and return predictions."""
   
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")

    images_tensor_S3HW = load_and_preprocess_images(image_names, mode=ba_args.preprocess_mode).to(device)
    print(f"Preprocessed images shape: {images_tensor_S3HW.shape}")

    images_tensor_BS3HW = images_tensor_S3HW.unsqueeze(0)
    print("Running initial VGGT inference for depth, intrinsics, etc...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            initial_predictions_dict = model(images_tensor_BS3HW)

    _, initial_intrinsics_tensor_BS33 = pose_encoding_to_extri_intri(
        initial_predictions_dict["pose_enc"], images_tensor_BS3HW.shape[-2:] 
    )

    initial_intrinsics_np_S33 = initial_intrinsics_tensor_BS33.cpu().numpy().squeeze(0)

    print("Running VGGT with Bundle Adjustment...")
    refined_extrinsics_tensor_S34 = run_vggt_with_ba(
        model,
        images_tensor_S3HW, 
        image_names=image_names, 
        dtype=dtype,
        max_query_num=ba_args.max_query_num,
        det_thres=ba_args.det_thres,
        query_frame_num=ba_args.query_frame_num,
        extractor_method=ba_args.extractor_method,
        max_reproj_error=ba_args.max_reproj_error,
        shared_camera=ba_args.shared_camera,
        camera_type=ba_args.camera_type,
    )

    refined_extrinsics_np_S34 = refined_extrinsics_tensor_S34.cpu().numpy() 

    predictions_for_colmap = {}
    predictions_for_colmap["extrinsic"] = refined_extrinsics_np_S34
    predictions_for_colmap["intrinsic"] = initial_intrinsics_np_S33


    for key in ["depth", "depth_conf", "world_points", "world_points_conf"]:
        if key in initial_predictions_dict and isinstance(initial_predictions_dict[key], torch.Tensor):
            predictions_for_colmap[key] = initial_predictions_dict[key].cpu().numpy().squeeze(0)
    
        elif key in initial_predictions_dict and isinstance(initial_predictions_dict[key], np.ndarray):
             predictions_for_colmap[key] = initial_predictions_dict[key].squeeze(0)
        elif key not in initial_predictions_dict:
            print(f"Warning: Key '{key}' not found in initial_predictions_dict.")
            if key.endswith("_conf"):
                 if key.replace("_conf","") in predictions_for_colmap:
                    base_key_shape = predictions_for_colmap[key.replace("_conf","")].shape
            
                    conf_shape = base_key_shape[:-1] if base_key_shape[-1] == 3 or base_key_shape[-1] == 1 else base_key_shape
                    predictions_for_colmap[key] = np.ones(conf_shape, dtype=np.float32)
                    print(f"Created default confidence for {key} with shape {conf_shape}")


    print("Re-computing 3D points from depth maps using refined extrinsics...")
    depth_map_np = predictions_for_colmap.get("depth")
    if depth_map_np is None:
        raise ValueError("Depth map not found in predictions, cannot recompute world_points_from_depth.")
    if depth_map_np.ndim == 3:
        depth_map_np = np.expand_dims(depth_map_np, axis=-1) 

    world_points_from_depth_refined_SHW3 = unproject_depth_map_to_point_map(
        depth_map_np, refined_extrinsics_np_S34, initial_intrinsics_np_S33
    )
    predictions_for_colmap["world_points_from_depth"] = world_points_from_depth_refined_SHW3

    original_images_pil_list = []
    for img_path in image_names:
        img_pil = Image.open(img_path).convert('RGB')
        original_images_pil_list.append(np.array(img_pil))
  
    S_refined, H_refined, W_refined = world_points_from_depth_refined_SHW3.shape[:3]
    normalized_images_for_coloring_SHW3 = np.zeros((S_refined, H_refined, W_refined, 3), dtype=np.float32)
    
    for i, img_np_orig in enumerate(original_images_pil_list):
        resized_img = cv2.resize(img_np_orig, (W_refined, H_refined)) 
        normalized_images_for_coloring_SHW3[i] = resized_img / 255.0
    
    predictions_for_colmap["images"] = normalized_images_for_coloring_SHW3
    
    return predictions_for_colmap, image_names


def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []
    
    for i in range(num_cameras):
        R_wc = extrinsics[i, :3, :3] 
        t_wc = extrinsics[i, :3, 3] 
        
        rot = Rotation.from_matrix(R_wc)
        quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        quaternions.append(quat_wxyz)
        translations.append(t_wc)
    
    return np.array(quaternions), np.array(translations)

def download_file_from_url(url, filename):
    """Downloads a file from a URL, handling redirects."""
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status() 

        if response.status_code == 302:  
            redirect_url = response.headers["Location"]
            print(f"Redirecting to {redirect_url}")
            response = requests.get(redirect_url, stream=True)
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
    if image is None:
        print(f"Warning: Could not read image {image_path} for sky segmentation.")
        return np.zeros((100,100), dtype=np.uint8) # return a dummy small mask

    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original >= 32] = 1 # Non-sky = 1
    # output_mask[result_map_original < 32] = 0 is implicit due to zeros_like

    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask * 255)

    return output_mask # Returns 0 for sky, 1 for non-sky

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
    if max_value == min_value: # Avoid division by zero
        onnx_result = np.zeros_like(onnx_result, dtype=np.uint8)
    else:
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype("uint8")

    return onnx_result

def filter_and_prepare_points(predictions, conf_threshold_percent, mask_sky=False, mask_black_bg=False, 
                             mask_white_bg=False, stride=1, prediction_mode="Depthmap and Camera Branch",
                             image_dir_for_skymask=None): 
    """
    Filter points based on confidence and prepare for COLMAP format.
    """
    
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch for 3D points")
        if "world_points" in predictions and predictions["world_points"] is not None:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
            if pred_world_points_conf is None: 
                pred_world_points_conf = np.ones_like(pred_world_points[...,0])
        else:
            print("Warning: world_points not found or is None in predictions for Pointmap Branch, falling back to depth-based points.")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
            if pred_world_points_conf is None: 
                 pred_world_points_conf = np.ones_like(pred_world_points[...,0])
    else: 
        print("Using Depthmap and Camera Branch for 3D points (world_points_from_depth)")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
        if pred_world_points_conf is None:
            pred_world_points_conf = np.ones_like(pred_world_points[...,0])


    colors_rgb_normalized = predictions["images"] 
    
    S, H, W = pred_world_points.shape[:3]

    if colors_rgb_normalized.shape[1:3] != (H, W):
        print(f"Resizing colors_rgb_normalized from {colors_rgb_normalized.shape[1:3]} to match point map {(H, W)}")
        resized_colors_rgb_normalized = np.zeros((S, H, W, 3), dtype=np.float32)
        for i in range(S):
            if i < len(colors_rgb_normalized):
                resized_colors_rgb_normalized[i] = cv2.resize(colors_rgb_normalized[i], (W, H))
        colors_rgb_normalized = resized_colors_rgb_normalized
    
    colors_rgb_uint8 = (colors_rgb_normalized * 255).astype(np.uint8)
    
    if mask_sky:
        print("Applying sky segmentation mask...")
        if image_dir_for_skymask is None:
            print("Warning: image_dir_for_skymask not provided for sky segmentation. Skipping.")
            mask_sky = False
        else:
            try:
                import onnxruntime
                with tempfile.TemporaryDirectory() as temp_sky_process_dir:
                    print(f"Using temporary directory for sky segmentation processing: {temp_sky_process_dir}")
                    
                    skyseg_model_path = os.path.join(temp_sky_process_dir, "skyseg.onnx")
                    if not os.path.exists("skyseg.onnx"): 
                        print("Downloading skyseg.onnx...")
                        download_success = download_file_from_url(
                            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
                            skyseg_model_path
                        )
                        if not download_success:
                            print("Failed to download skyseg model, skipping sky filtering.")
                            mask_sky = False
                    else:
                        import shutil
                        shutil.copy("skyseg.onnx", skyseg_model_path)
                        print("Using local skyseg.onnx.")

                    if mask_sky:
                        skyseg_session = onnxruntime.InferenceSession(skyseg_model_path)
                        
                        image_paths_orig = sorted([
                            f for f in glob.glob(os.path.join(image_dir_for_skymask, "*")) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                        ])

                        if len(image_paths_orig) != S:
                            print(f"Warning: Number of original images ({len(image_paths_orig)}) "
                                  f"does not match number of frames in prediction ({S}). Sky mask might be misaligned.")
                        
                        sky_mask_frames = []
                        for i in range(S):
                            if i < len(image_paths_orig):
                                img_path_orig = image_paths_orig[i]
                               
                                sky_mask_output_path = os.path.join(temp_sky_process_dir, f"skymask_{i:04d}.png")
                                sky_mask_single_frame = segment_sky(img_path_orig, skyseg_session, sky_mask_output_path) 
                            else: 
                                sky_mask_single_frame = np.ones((H,W), dtype=np.float32)

                            if sky_mask_single_frame.shape[0] != H or sky_mask_single_frame.shape[1] != W:
                                sky_mask_single_frame = cv2.resize(sky_mask_single_frame, (W, H), interpolation=cv2.INTER_NEAREST)
                            
                            sky_mask_frames.append(sky_mask_single_frame)
                        
                        sky_mask_all_frames_SHW = np.array(sky_mask_frames).astype(np.float32) 
                        pred_world_points_conf = pred_world_points_conf * sky_mask_all_frames_SHW 
                        print(f"Applied sky mask to confidence, original shape: {pred_world_points_conf.shape}")
                    
            except (ImportError, RuntimeError, Exception) as e: 
                print(f"Error during sky segmentation: {e}. Skipping sky filtering.")
                mask_sky = False
    
    vertices_3d_flat = pred_world_points.reshape(-1, 3)
    conf_flat = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb_uint8.reshape(-1, 3) 

    min_len = min(len(vertices_3d_flat), len(conf_flat), len(colors_rgb_flat))
    if len(vertices_3d_flat) != min_len:
        print(f"Trimming vertices_3d_flat from {len(vertices_3d_flat)} to {min_len}")
        vertices_3d_flat = vertices_3d_flat[:min_len]
    if len(conf_flat) != min_len:
        print(f"Trimming conf_flat from {len(conf_flat)} to {min_len}")
        conf_flat = conf_flat[:min_len]
    if len(colors_rgb_flat) != min_len:
        print(f"Trimming colors_rgb_flat from {len(colors_rgb_flat)} to {min_len}")
        colors_rgb_flat = colors_rgb_flat[:min_len]

    if conf_threshold_percent == 0.0:
        conf_thres_value = 0.0
    elif len(conf_flat) > 0 :
        conf_thres_value = np.percentile(conf_flat, conf_threshold_percent)
    else:
        conf_thres_value = 0.0  

    print(f"Using confidence threshold: {conf_threshold_percent}% (value: {conf_thres_value:.4f})")
    
    if len(conf_flat) > 0:
        combined_mask = (conf_flat >= conf_thres_value) & (conf_flat > 1e-5) 
    else:
        combined_mask = np.array([], dtype=bool)

    if mask_black_bg and len(colors_rgb_flat) > 0:
        print("Filtering black background")
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16 # Sum of R,G,B
        combined_mask = combined_mask & black_bg_mask
    
    if mask_white_bg and len(colors_rgb_flat) > 0:
        print("Filtering white background")
        white_bg_mask = ~((colors_rgb_flat[:, 0] > 240) & (colors_rgb_flat[:, 1] > 240) & (colors_rgb_flat[:, 2] > 240))
        combined_mask = combined_mask & white_bg_mask
    

    points3D_colmap = []
    point_indices_map = {}  
    image_points2D_tracks = [[] for _ in range(S)] 
    
    print(f"Preparing points for COLMAP format with stride {stride}...")
    
    for img_idx in range(S):
        for r_idx in range(0, H, stride): 
            for c_idx in range(0, W, stride): 
                flat_array_idx = img_idx * H * W + r_idx * W + c_idx
                
                if flat_array_idx >= min_len:
                    continue
                
                current_conf = conf_flat[flat_array_idx]
                current_color_rgb = colors_rgb_flat[flat_array_idx]

                if current_conf < conf_thres_value or current_conf <= 1e-5:
                    continue
                
                if mask_black_bg and current_color_rgb.sum() < 16:
                    continue
                
                if mask_white_bg and all(c > 240 for c in current_color_rgb):
                    continue
                

                point3D_xyz = vertices_3d_flat[flat_array_idx]
                
                if not np.all(np.isfinite(point3D_xyz)):
                    continue
                
                point_hash_val = hash_point(point3D_xyz, scale=100) 
                
                if point_hash_val not in point_indices_map:
                    new_point3d_idx = len(points3D_colmap)
                    point_indices_map[point_hash_val] = new_point3d_idx
                    
                    colmap_point_entry = {
                        "id": new_point3d_idx, 
                        "xyz": point3D_xyz,
                        "rgb": current_color_rgb,
                        "error": 1.0, 
                        "track": [] 
                    }
                    points3D_colmap.append(colmap_point_entry)
                
                point3d_idx_in_colmap_list = point_indices_map[point_hash_val]
                point2d_idx_in_current_image_track = len(image_points2D_tracks[img_idx])
                
                image_points2D_tracks[img_idx].append((c_idx, r_idx, point3d_idx_in_colmap_list))
                points3D_colmap[point3d_idx_in_colmap_list]["track"].append((img_idx, point2d_idx_in_current_image_track))
    
    num_observations = sum(len(pts) for pts in image_points2D_tracks)
    print(f"Prepared {len(points3D_colmap)} unique 3D points with {num_observations} observations for COLMAP.")
    return points3D_colmap, image_points2D_tracks

def hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)


def write_colmap_cameras_txt(file_path, intrinsics_S33, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    # intrinsics_S33 is a numpy array of shape (S, 3, 3)
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics_S33)}\n")
        
        for i, intrinsic_matrix in enumerate(intrinsics_S33):
            camera_id = i + 1  # COLMAP uses 1-indexed camera IDs
            model = "PINHOLE" # Assumes fx, fy, cx, cy
            
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            
            f.write(f"{camera_id} {model} {image_width} {image_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

def write_colmap_images_txt(file_path, quaternions_wxyz_S4, translations_S3, image_points2D_tracks, image_names):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        num_observations = sum(len(points) for points in image_points2D_tracks)
        avg_points_per_image = num_observations / len(image_points2D_tracks) if image_points2D_tracks else 0
        f.write(f"# Number of images: {len(quaternions_wxyz_S4)}, mean observations per image: {avg_points_per_image:.1f}\n")
        
        for i in range(len(quaternions_wxyz_S4)):
            image_id = i + 1 
            camera_id = i + 1  
          
            qw, qx, qy, qz = quaternions_wxyz_S4[i]
            tx, ty, tz = translations_S3[i]
            
            image_filename = os.path.basename(image_names[i])
            f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {camera_id} {image_filename}\n")
        
            points_line_parts = []
            for x_coord, y_coord, point3d_id_colmap_0indexed in image_points2D_tracks[i]:
                points_line_parts.extend([f"{x_coord:.2f}", f"{y_coord:.2f}", f"{point3d_id_colmap_0indexed + 1}"])
            f.write(" ".join(points_line_parts) + "\n")


def write_colmap_points3D_txt(file_path, points3D_colmap_list):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        avg_track_len = sum(len(p["track"]) for p in points3D_colmap_list) / len(points3D_colmap_list) if points3D_colmap_list else 0
        f.write(f"# Number of points: {len(points3D_colmap_list)}, mean track length: {avg_track_len:.4f}\n")
        
        for point_entry in points3D_colmap_list:
            point_id_colmap_1indexed = point_entry["id"] + 1
            x, y, z = point_entry["xyz"]
            r_val, g_val, b_val = point_entry["rgb"] 
            error_val = point_entry["error"]
            
            track_parts = []
            for img_id_0indexed, pt2d_idx_0indexed in point_entry["track"]:
                track_parts.extend([f"{img_id_0indexed + 1}", f"{pt2d_idx_0indexed}"]) 
            
            track_str = " ".join(track_parts)
            f.write(f"{point_id_colmap_1indexed} {x:.6f} {y:.6f} {z:.6f} {int(r_val)} {int(g_val)} {int(b_val)} {error_val:.6f} {track_str}\n")


def write_colmap_cameras_bin(file_path, intrinsics_S33, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.bin format."""
    with open(file_path, 'wb') as fid:
        num_cameras = len(intrinsics_S33)
        fid.write(struct.pack('<Q', num_cameras)) # uint64_t
        
        for i, intrinsic_matrix in enumerate(intrinsics_S33):
            camera_id = i + 1 
            model_id = 1 
            
            fx = float(intrinsic_matrix[0, 0])
            fy = float(intrinsic_matrix[1, 1])
            cx = float(intrinsic_matrix[0, 2])
            cy = float(intrinsic_matrix[1, 2])
            
            fid.write(struct.pack('<I', camera_id))        # camera_id (uint32_t)
            fid.write(struct.pack('<i', model_id))         # model_id (int)
            fid.write(struct.pack('<Q', image_width))      # width (uint64_t)
            fid.write(struct.pack('<Q', image_height))     # height (uint64_t)
            fid.write(struct.pack('<dddd', fx, fy, cx, cy)) # params (double)

def write_colmap_images_bin(file_path, quaternions_wxyz_S4, translations_S3, image_points2D_tracks, image_names):
    """Write camera poses and keypoints to COLMAP images.bin format."""
    with open(file_path, 'wb') as fid:
        num_images = len(quaternions_wxyz_S4)
        fid.write(struct.pack('<Q', num_images))
        
        for i in range(num_images):
            image_id = i + 1
            camera_id = i + 1 
            
            qw, qx, qy, qz = quaternions_wxyz_S4[i].astype(float)
            tx, ty, tz = translations_S3[i].astype(float)
            
            image_filename_bytes = os.path.basename(image_names[i]).encode('utf-8')
            current_image_tracks = image_points2D_tracks[i] # List of (x,y, point3d_id_0indexed)
            
            fid.write(struct.pack('<I', image_id))          # image_id (uint32_t)
            fid.write(struct.pack('<dddd', qw, qx, qy, qz)) # qvec (double[4])
            fid.write(struct.pack('<ddd', tx, ty, tz))      # tvec (double[3])
            fid.write(struct.pack('<I', camera_id))        # camera_id (uint32_t)
            
            # Image name: null-terminated string. Length first, then chars.
            fid.write(image_filename_bytes)
            fid.write(b'\x00') # Null terminator

            fid.write(struct.pack('<Q', len(current_image_tracks))) # num_points2D (uint64_t)
            
            for x_coord, y_coord, point3d_id_colmap_0indexed in current_image_tracks:
                fid.write(struct.pack('<dd', float(x_coord), float(y_coord))) 
                fid.write(struct.pack('<Q', point3d_id_colmap_0indexed + 1))

def write_colmap_points3D_bin(file_path, points3D_colmap_list):
    """Write 3D points and tracks to COLMAP points3D.bin format."""
    with open(file_path, 'wb') as fid:
        num_points = len(points3D_colmap_list)
        fid.write(struct.pack('<Q', num_points)) 
        
        for point_entry in points3D_colmap_list:
            point_id_colmap_1indexed = point_entry["id"] + 1
            xyz = point_entry["xyz"].astype(float)
            rgb = point_entry["rgb"].astype(np.uint8) # R, G, B
            error_val = float(point_entry["error"])
            track = point_entry["track"] # List of (image_id_0indexed, point2d_idx_in_image_track_0indexed)
            
            fid.write(struct.pack('<Q', point_id_colmap_1indexed)) # point3D_id (uint64_t)
            fid.write(struct.pack('<ddd', xyz[0], xyz[1], xyz[2]))  # xyz (double[3])
            fid.write(struct.pack('<BBB', rgb[0], rgb[1], rgb[2]))  # color (uint8_t[3])
            fid.write(struct.pack('<d', error_val))                 # error (double)
            
            fid.write(struct.pack('<Q', len(track))) # track_len (uint64_t)
            for img_id_0indexed, pt2d_idx_0indexed in track:
                fid.write(struct.pack('<II', img_id_0indexed + 1, pt2d_idx_0indexed)) # image_id (uint32_t), point2D_idx (uint32_t)

def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT with Bundle Adjustment")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="colmap_output_ba", 
                        help="Directory to save COLMAP files")
    parser.add_argument("--conf_threshold", type=float, default=50.0, 
                        help="Confidence threshold (percentage, 0-100) for including points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter out points likely to be sky")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter out points with very dark/black color")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter out points with very bright/white color")
    parser.add_argument("--binary", action="store_true", 
                        help="Output binary COLMAP files instead of text")
    parser.add_argument("--stride", type=int, default=1, 
                        help="Stride for point sampling from depth/point maps (higher = fewer points)")
    parser.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Which VGGT prediction branch to use for 3D points (before BA an re-projection)")
    parser.add_argument("--preprocess_mode", type=str, default="crop", choices=["crop", "pad"],
                        help="Preprocessing mode for images ('crop' or 'pad') for load_and_preprocess_images.")

    # BA specific arguments
    parser.add_argument("--max_query_num", type=int, default=2048, help="BA: Max number of query points for tracking")
    parser.add_argument("--det_thres", type=float, default=0.005, help="BA: Detection threshold for keypoint extraction")
    parser.add_argument("--query_frame_num", type=int, default=3, help="BA: Number of frames to select for feature extraction by DINO")
    parser.add_argument("--extractor_method", type=str, default="aliked+sp+sift", help="BA: Feature extraction method (e.g., 'aliked', 'sp+sift')")
    parser.add_argument("--max_reproj_error", type=int, default=12, help="BA: Max reprojection error for COLMAP BA")
    parser.add_argument("--no_shared_camera", action="store_false", dest="shared_camera",
                        help="BA: Do not use shared camera parameters during COLMAP BA (default is to share)")
    parser.set_defaults(shared_camera=True)
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", 
                        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL"], # Add more if supported by tensor_to_pycolmap
                        help="BA: Camera model type for COLMAP BA internal representation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, device = load_model()

    predictions, image_names = process_images_with_ba(args.image_dir, model, device, args)
    
    print("Converting camera parameters to COLMAP format...")
    quaternions_S4_wxyz, translations_S3 = extrinsic_to_colmap_format(predictions["extrinsic"])
    
    print(f"Filtering points with confidence threshold {args.conf_threshold}% and stride {args.stride}...")
    points3D, image_points2D = filter_and_prepare_points(
        predictions, 
        args.conf_threshold, 
        mask_sky=args.mask_sky, 
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        stride=args.stride,
        prediction_mode=args.prediction_mode,
        image_dir_for_skymask=args.image_dir # Pass original image dir for sky mask
    )
    

    if "depth" in predictions and predictions["depth"] is not None:
        height, width = predictions["depth"].shape[1:3] 
    elif "world_points_from_depth" in predictions and predictions["world_points_from_depth"] is not None:
        height, width = predictions["world_points_from_depth"].shape[1:3]
    elif "world_points" in predictions and predictions["world_points"] is not None:
        height, width = predictions["world_points"].shape[1:3]
    else:
        # Fallback: try to get from preprocessed images fed to model
        img_temp = Image.open(image_names[0])
        img_tensor_temp = load_and_preprocess_images([image_names[0]], mode=args.preprocess_mode)
        height, width = img_tensor_temp.shape[-2:]
        print(f"Warning: Could not determine H,W from prediction keys, using H={height}, W={width} from a sample preprocessed image.")


    print(f"Writing {'binary' if args.binary else 'text'} COLMAP files to {args.output_dir}...")
    if args.binary:
        write_colmap_cameras_bin(
            os.path.join(args.output_dir, "cameras.bin"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_bin(
            os.path.join(args.output_dir, "images.bin"), 
            quaternions_S4_wxyz, translations_S3, image_points2D, image_names)
        write_colmap_points3D_bin(
            os.path.join(args.output_dir, "points3D.bin"), 
            points3D)
    else:
        write_colmap_cameras_txt(
            os.path.join(args.output_dir, "cameras.txt"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_txt(
            os.path.join(args.output_dir, "images.txt"), 
            quaternions_S4_wxyz, translations_S3, image_points2D, image_names)
        write_colmap_points3D_txt(
            os.path.join(args.output_dir, "points3D.txt"), 
            points3D)
    
    print(f"COLMAP files with Bundle Adjustment results successfully written to {args.output_dir}")

if __name__ == "__main__":
    main()