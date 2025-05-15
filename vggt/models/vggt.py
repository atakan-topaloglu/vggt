# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from typing import List, Dict, Optional, Union


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        heads_to_run: Optional[List[str]] = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            heads_to_run (Optional[List[str]]): List of head names to run (e.g., ['camera', 'depth']). If None, runs default set.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if heads_to_run is None: # Default: run all relevant heads
            heads_to_run = ['camera', 'depth', 'point']
            if query_points is not None:
                 heads_to_run.append('track')

        aggregated_tokens_list_full, patch_start_idx = self.aggregator(images)
        predictions = {}
        required_dpt_layers_indices = set()
        active_dpt_heads = []

        if 'depth' in heads_to_run and self.depth_head is not None:
            active_dpt_heads.append(self.depth_head)
        if 'point' in heads_to_run and self.point_head is not None:
            active_dpt_heads.append(self.point_head)
        # track_head.feature_extractor is a DPTHead
        if 'track' in heads_to_run and self.track_head is not None and self.track_head.feature_extractor is not None:
             active_dpt_heads.append(self.track_head.feature_extractor)

        for dpt_head_instance in active_dpt_heads:
            if hasattr(dpt_head_instance, 'get_required_inter_layers'):
                required_dpt_layers_indices.update(dpt_head_instance.get_required_inter_layers())
        
        max_depth_idx = len(aggregated_tokens_list_full) - 1
        
        # Prepare inputs for heads
        # CameraHead uses the last layer from aggregated_tokens_list
        tokens_for_camera = [aggregated_tokens_list_full[max_depth_idx]] if 'camera' in heads_to_run and self.camera_head is not None else None
        
        # DPT-based heads use specific intermediate layers, passed as a dictionary
        tokens_for_dpt = {
            idx: aggregated_tokens_list_full[idx] 
            for idx in sorted(list(required_dpt_layers_indices))
            if idx <= max_depth_idx  # Ensure index is valid
        } if active_dpt_heads else None



        with torch.cuda.amp.autocast(enabled=False):
            if 'camera' in heads_to_run and self.camera_head is not None:
                pose_enc_list = self.camera_head(tokens_for_camera)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if 'depth' in heads_to_run and self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    tokens_for_dpt, images=images, patch_start_idx=patch_start_idx
                 )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if 'point' in heads_to_run and self.point_head is not None: 
                pts3d, pts3d_conf = self.point_head(
                    tokens_for_dpt, images=images, patch_start_idx=patch_start_idx
                 )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if 'track' in heads_to_run and self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                tokens_for_dpt, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf
        
        # Clean up the full aggregated tokens list to save memory if parts of it were selected
        if tokens_for_camera is not None or tokens_for_dpt is not None:
            del aggregated_tokens_list_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Store original images if any head was run (often needed for visualization or by DPT heads)
        if heads_to_run: 
            predictions["images"] = images

        return predictions
