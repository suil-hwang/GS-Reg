import os
import sys
import argparse
import numpy as np
import torch
from scene import GaussianModel
import open3d as o3d
from utils.registration_utils import AlignmentController

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and merge two Gaussian Splatting models.")
    parser.add_argument("source_ply", type=str, help="Path to the source PLY file")
    parser.add_argument("target_ply", type=str, help="Path to the target PLY file")
    parser.add_argument("output_ply", type=str, help="Path to the output merged PLY file")
    parser.add_argument(
        "--align_ground_plane", 
        action='store_true', 
        help="Perform ground plane detection and alignment. This step is skipped if not specified."
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=0.05,
        help="Coefficient used for threshold setting (default: 0.05)"
    )
    parser.add_argument(
        "--icp_method",
        type=str,
        choices=['point_to_point', 'point_to_plane'],
        default='point_to_point',
        help="ICP alignment method: 'point_to_point' or 'point_to_plane' (default: 'point_to_point')"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Maximum number of ICP iterations (default: 50)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="RMSE change tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--threshold_decay",
        type=float,
        default=0.8,
        help="Threshold decay coefficient (default: 0.8)"
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.01,
        help="Minimum threshold value (default: 0.01)"
    )
    args = parser.parse_args()

    controller = AlignmentController(
        source_path=args.source_ply,
        target_path=args.target_ply,
        output_path=args.output_ply,
        align_ground_plane=args.align_ground_plane,
        multiplier=args.multiplier,
        icp_method=args.icp_method,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        threshold_decay=args.threshold_decay,
        min_threshold=args.min_threshold
    )

    controller.run()
