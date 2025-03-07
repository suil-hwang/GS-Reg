import numpy as np
import open3d as o3d
import torch
import copy
import os
import sys
import logging
from scipy.spatial.transform import Rotation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GaussianModelHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.gaussian_model = None
        self.point_cloud = None

    def load_gaussian_model(self):
        try:
            self.point_cloud = o3d.io.read_point_cloud(self.file_path)
            if not self.point_cloud.has_points():
                logging.error(f"Failed to load point cloud or empty point cloud: {self.file_path}")
                return False
            logging.info(f"Loaded point cloud with {len(self.point_cloud.points)} points")
            return True
        except Exception as e:
            logging.error(f"Error loading point cloud: {str(e)}")
            return False

    def convert_to_point_cloud(self):
        # This function assumes the model is already loaded as a point cloud
        return self.point_cloud

    def apply_transformation(self, transformation):
        if self.point_cloud is None:
            logging.error("No point cloud loaded")
            return False
        
        try:
            self.point_cloud.transform(transformation)
            logging.info("Transformation applied successfully")
            return True
        except Exception as e:
            logging.error(f"Error applying transformation: {str(e)}")
            return False

    def save_ply(self, output_path):
        if self.point_cloud is None:
            logging.error("No point cloud to save")
            return False
        
        try:
            o3d.io.write_point_cloud(output_path, self.point_cloud)
            logging.info(f"Point cloud saved to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving point cloud: {str(e)}")
            return False

class KeypointSelector:
    def __init__(self, gaussian_model_handler, point_size=2.0):
        self.gaussian_model_handler = gaussian_model_handler
        self.point_size = point_size
        self.keypoints = []

    def run(self):
        if self.gaussian_model_handler.point_cloud is None:
            logging.error("No point cloud loaded for keypoint selection")
            return []

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Select Keypoints (Press 'Q' to exit)", width=1024, height=768)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        # Add point cloud to visualizer
        vis.add_geometry(self.gaussian_model_handler.point_cloud)
        
        # Set view control
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Instructions
        print("\n===== KEYPOINT SELECTION =====")
        print("1. Shift+Left click to select points")
        print("2. Press 'Q' to close the window when done")
        print("===============================\n")
        
        # Run visualization and get indices of picked points
        vis.run()
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        
        # Convert indices to 3D coordinates
        if picked_indices:
            points = np.asarray(self.gaussian_model_handler.point_cloud.points)
            self.keypoints = [points[idx] for idx in picked_indices]
            logging.info(f"Selected {len(self.keypoints)} keypoints")
        else:
            logging.warning("No keypoints selected")
            
        return self.keypoints

class ICPAligner:
    def __init__(self, multiplier=0.05, icp_method='point_to_point', max_iterations=50, 
                 tolerance=1e-6, threshold_decay=0.8, min_threshold=0.01):
        self.multiplier = multiplier
        self.icp_method = icp_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.threshold_decay = threshold_decay
        self.min_threshold = min_threshold

    def estimate_initial_threshold(self, point_cloud):
        if not point_cloud.has_points():
            return 0.05  # Default threshold
            
        # Get point cloud bounds
        points = np.asarray(point_cloud.points)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        
        # Calculate diagonal length
        diagonal = np.linalg.norm(max_bound - min_bound)
        
        # Return threshold as a proportion of diagonal
        threshold = diagonal * self.multiplier
        logging.info(f"Estimated initial threshold: {threshold:.6f}")
        return threshold

    def perform_icp(self, source_pcd, target_pcd, initial_transformation=np.eye(4)):
        if not source_pcd.has_points() or not target_pcd.has_points():
            logging.error("Source or target point cloud is empty")
            return np.eye(4), np.eye(6), 0.0, float('inf')
        
        # Make deep copies to avoid modifying the originals
        source = copy.deepcopy(source_pcd)
        target = copy.deepcopy(target_pcd)
        
        # Apply initial transformation
        source.transform(initial_transformation)
        
        # Select ICP method
        if self.icp_method == 'point_to_plane':
            # Estimate normals if using point-to-plane
            if not target.has_normals():
                target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            if not source.has_normals():
                source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Start with an initial threshold
        current_threshold = self.estimate_initial_threshold(target)
        best_fitness = 0
        best_rmse = float('inf')
        best_transformation = np.eye(4)
        best_information = np.eye(6)
        
        # Initialize progress
        logging.info("Starting ICP alignment...")
        
        # Loop until threshold becomes too small
        iteration = 0
        while current_threshold >= self.min_threshold and iteration < self.max_iterations:
            # Perform ICP for current threshold
            if self.icp_method == 'point_to_plane':
                result = o3d.pipelines.registration.registration_icp(
                    source, target, current_threshold, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20, relative_fitness=self.tolerance, relative_rmse=self.tolerance)
                )
            else:  # point_to_point
                result = o3d.pipelines.registration.registration_icp(
                    source, target, current_threshold, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20, relative_fitness=self.tolerance, relative_rmse=self.tolerance)
                )
            
            # Check if this result is better
            if result.fitness > best_fitness or (result.fitness == best_fitness and result.inlier_rmse < best_rmse):
                best_fitness = result.fitness
                best_rmse = result.inlier_rmse
                best_transformation = result.transformation
                best_information = result.information
                
                # Apply the transformation for the next iteration
                source.transform(result.transformation)
            
            # Reduce threshold for next iteration
            current_threshold *= self.threshold_decay
            iteration += 1
            
            # Log progress
            logging.info(f"ICP Iteration {iteration}: Threshold={current_threshold:.6f}, Fitness={result.fitness:.6f}, RMSE={result.inlier_rmse:.6f}")
        
        # Compose the final transformation (initial * best)
        final_transformation = np.matmul(best_transformation, initial_transformation)
        
        logging.info(f"ICP completed with fitness={best_fitness:.6f}, RMSE={best_rmse:.6f}")
        return final_transformation, best_information, best_fitness, best_rmse

class PLYMerger:
    @staticmethod
    def read_ply(file_path):
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                logging.error(f"Empty point cloud read from {file_path}")
            return pcd
        except Exception as e:
            logging.error(f"Error reading PLY file {file_path}: {str(e)}")
            return None

    @staticmethod
    def merge_ply(file1, file2, output_file):
        try:
            # Read point clouds
            pcd1 = PLYMerger.read_ply(file1)
            pcd2 = PLYMerger.read_ply(file2)
            
            if pcd1 is None or pcd2 is None:
                return False
                
            # Merge point clouds
            merged_points = np.vstack([np.asarray(pcd1.points), np.asarray(pcd2.points)])
            
            # Merge colors if available
            if pcd1.has_colors() and pcd2.has_colors():
                merged_colors = np.vstack([np.asarray(pcd1.colors), np.asarray(pcd2.colors)])
            elif pcd1.has_colors():
                # Create default color for pcd2
                default_color = np.ones((len(pcd2.points), 3)) * 0.7
                merged_colors = np.vstack([np.asarray(pcd1.colors), default_color])
            elif pcd2.has_colors():
                # Create default color for pcd1
                default_color = np.ones((len(pcd1.points), 3)) * 0.7
                merged_colors = np.vstack([default_color, np.asarray(pcd2.colors)])
            else:
                merged_colors = None
            
            # Create merged point cloud
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
            if merged_colors is not None:
                merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            
            # Save merged point cloud
            o3d.io.write_point_cloud(output_file, merged_pcd)
            logging.info(f"Merged point cloud saved to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error merging PLY files: {str(e)}")
            return False

class AlignmentController:
    def __init__(self, source_path, target_path, output_path, align_ground_plane=True, 
                 multiplier=0.05, icp_method='point_to_point', max_iterations=50, 
                 tolerance=1e-6, threshold_decay=0.8, min_threshold=0.01,
                 source_keypoints_file='source_keypoints.txt', target_keypoints_file='target_keypoints.txt'):
        self.source_path = source_path
        self.target_path = target_path
        self.output_path = output_path
        self.align_ground_plane = align_ground_plane
        self.source_keypoints_file = source_keypoints_file
        self.target_keypoints_file = target_keypoints_file
        
        # Initialize handlers and aligners
        self.source_handler = GaussianModelHandler(source_path)
        self.target_handler = GaussianModelHandler(target_path)
        self.icp_aligner = ICPAligner(multiplier, icp_method, max_iterations, tolerance, threshold_decay, min_threshold)

    def save_keypoints(self, keypoints, file_path):
        try:
            with open(file_path, 'w') as f:
                for point in keypoints:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
            logging.info(f"Keypoints saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving keypoints: {str(e)}")
            return False

    def load_keypoints(self, file_path):
        keypoints = []
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 3:
                            point = [float(values[0]), float(values[1]), float(values[2])]
                            keypoints.append(point)
                logging.info(f"Loaded {len(keypoints)} keypoints from {file_path}")
            else:
                logging.warning(f"Keypoint file not found: {file_path}")
        except Exception as e:
            logging.error(f"Error loading keypoints: {str(e)}")
        
        return keypoints

    def detect_and_align_ground_plane(self):
        if not self.source_handler.load_gaussian_model() or not self.target_handler.load_gaussian_model():
            return np.eye(4)
            
        source_pcd = self.source_handler.point_cloud
        target_pcd = self.target_handler.point_cloud
        
        # Detect ground plane in source and target
        source_plane = detect_ground_plane_ransac(source_pcd)
        target_plane = detect_ground_plane_ransac(target_pcd)
        
        if source_plane is not None and target_plane is not None:
            # Compute alignment transformation
            source_transform = compute_alignment_transformation(source_pcd, source_plane)
            target_transform = compute_alignment_transformation(target_pcd, target_plane)
            
            # Compute the combined transformation to align source to target
            combined_transform = np.matmul(np.linalg.inv(target_transform), source_transform)
            logging.info("Ground plane alignment computed successfully")
            return combined_transform
        else:
            logging.warning("Ground plane could not be detected, using identity transformation")
            return np.eye(4)

    def perform_icp_alignment(self):
        if not self.source_handler.load_gaussian_model() or not self.target_handler.load_gaussian_model():
            return np.eye(4)
            
        # Perform ICP alignment
        transformation, _, fitness, rmse = self.icp_aligner.perform_icp(
            self.source_handler.point_cloud, 
            self.target_handler.point_cloud
        )
        
        logging.info(f"ICP alignment completed with fitness={fitness:.4f}, RMSE={rmse:.4f}")
        return transformation

    def compute_keypoint_alignment(self, keypoints_source, keypoints_target):
        if len(keypoints_source) < 3 or len(keypoints_target) < 3:
            logging.warning("Not enough keypoints for alignment (need at least 3)")
            return np.eye(4)
            
        if len(keypoints_source) != len(keypoints_target):
            logging.warning(f"Keypoint count mismatch: source={len(keypoints_source)}, target={len(keypoints_target)}")
            min_points = min(len(keypoints_source), len(keypoints_target))
            keypoints_source = keypoints_source[:min_points]
            keypoints_target = keypoints_target[:min_points]
        
        transformation = compute_keypoint_alignment(keypoints_source, keypoints_target)
        logging.info(f"Keypoint alignment computed with {len(keypoints_source)} point pairs")
        return transformation

    def save_and_merge(self, temp_source_ply, temp_target_ply, merged_ply):
        # Save aligned point clouds
        self.source_handler.save_ply(temp_source_ply)
        self.target_handler.save_ply(temp_target_ply)
        
        # Merge point clouds
        result = PLYMerger.merge_ply(temp_source_ply, temp_target_ply, merged_ply)
        return result

    def run(self):
        logging.info("Starting alignment process")
        
        # Step 1: Load models
        if not self.source_handler.load_gaussian_model() or not self.target_handler.load_gaussian_model():
            logging.error("Failed to load models")
            return False
        
        # Initial transformation (identity)
        transformation = np.eye(4)
        
        # Step 2: Ground plane alignment (if enabled)
        if self.align_ground_plane:
            logging.info("Performing ground plane alignment")
            ground_transform = self.detect_and_align_ground_plane()
            self.source_handler.apply_transformation(ground_transform)
            transformation = ground_transform
        
        # Step 3: Check for existing keypoints or collect new ones
        source_keypoints = self.load_keypoints(self.source_keypoints_file)
        target_keypoints = self.load_keypoints(self.target_keypoints_file)
        
        if not source_keypoints or not target_keypoints:
            logging.info("No pre-existing keypoints found, starting keypoint selection")
            
            # Collect keypoints for source
            logging.info("Select keypoints for SOURCE model")
            source_keypoint_selector = KeypointSelector(self.source_handler)
            source_keypoints = source_keypoint_selector.run()
            self.save_keypoints(source_keypoints, self.source_keypoints_file)
            
            # Collect keypoints for target
            logging.info("Select keypoints for TARGET model")
            target_keypoint_selector = KeypointSelector(self.target_handler)
            target_keypoints = target_keypoint_selector.run()
            self.save_keypoints(target_keypoints, self.target_keypoints_file)
        
        # Step 4: Keypoint-based alignment
        if source_keypoints and target_keypoints:
            logging.info("Performing keypoint-based alignment")
            keypoint_transform = self.compute_keypoint_alignment(source_keypoints, target_keypoints)
            self.source_handler.apply_transformation(keypoint_transform)
            
            # Update cumulative transformation
            transformation = np.matmul(keypoint_transform, transformation)
        
        # Step 5: Fine-tune with ICP
        logging.info("Fine-tuning alignment with ICP")
        icp_transform = self.perform_icp_alignment()
        self.source_handler.apply_transformation(icp_transform)
        
        # Update cumulative transformation
        transformation = np.matmul(icp_transform, transformation)
        
        # Step 6: Save transformed source, target, and merged result
        def create_temp_ply(suffix):
            dir_name = os.path.dirname(self.output_path)
            base_name = os.path.basename(self.output_path)
            name_without_ext = os.path.splitext(base_name)[0]
            return os.path.join(dir_name, f"{name_without_ext}_{suffix}.ply")
        
        temp_source_ply = create_temp_ply("source_aligned")
        temp_target_ply = create_temp_ply("target")
        
        logging.info("Saving and merging point clouds")
        self.save_and_merge(temp_source_ply, temp_target_ply, self.output_path)
        
        logging.info(f"Alignment completed. Final transformation matrix:\n{transformation}")
        logging.info(f"Results saved to {self.output_path}")
        
        return True

def detect_ground_plane_ransac(point_cloud, distance_threshold=0.05, ransac_n=20, num_iterations=1000):
    if not point_cloud.has_points():
        logging.error("Point cloud has no points")
        return None
        
    # Segment plane
    try:
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers) > 100:  # Ensure we have enough inliers for a valid plane
            logging.info(f"Ground plane detected with {len(inliers)} inliers")
            return plane_model
        else:
            logging.warning(f"Too few inliers for ground plane: {len(inliers)}")
            return None
    except Exception as e:
        logging.error(f"Error detecting ground plane: {str(e)}")
        return None

def compute_alignment_transformation(point_cloud, plane_model):
    # Extract plane normal
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    # Compute rotation to align normal with z-axis (0, 0, 1)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)
    
    if np.linalg.norm(rotation_axis) < 1e-6:
        # Normal is already aligned (or anti-aligned) with z-axis
        if normal[2] > 0:
            # Already aligned with z-axis, no rotation needed
            rotation_matrix = np.eye(3)
        else:
            # Anti-aligned with z-axis, rotate 180 degrees around x-axis
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(normal, z_axis)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Use scipy's Rotation to create rotation matrix
        rotation = Rotation.from_rotvec(rotation_axis * angle)
        rotation_matrix = rotation.as_matrix()
    
    # Compute centroid of point cloud
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    
    # Compute translation
    # After rotation, we want the ground plane to be at z=0
    # So we need to translate by -d (adjusted by rotation)
    rotated_normal = rotation_matrix @ normal
    translation_z = d / np.linalg.norm(normal)
    if rotated_normal[2] < 0:
        translation_z = -translation_z
    
    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    
    # First translate to center, then rotate, then translate for ground plane alignment
    translation_matrix1 = np.eye(4)
    translation_matrix1[:3, 3] = -centroid
    
    translation_matrix2 = np.eye(4)
    translation_matrix2[2, 3] = translation_z
    
    translation_matrix3 = np.eye(4)
    translation_matrix3[:3, 3] = centroid
    
    # Combined transformation: T3 * R * T2 * T1
    transformation = np.matmul(translation_matrix3, np.matmul(transformation, np.matmul(translation_matrix2, translation_matrix1)))
    
    return transformation

def compute_keypoint_alignment(keypoints_source, keypoints_target):
    if len(keypoints_source) != len(keypoints_target) or len(keypoints_source) < 3:
        logging.error("Need at least 3 matching keypoint pairs")
        return np.eye(4)
    
    # Convert to numpy arrays
    source_points = np.array(keypoints_source)
    target_points = np.array(keypoints_target)
    
    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    # Center the points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Compute the covariance matrix
    covariance_matrix = np.matmul(source_centered.T, target_centered)
    
    # Singular Value Decomposition
    try:
        U, _, Vt = np.linalg.svd(covariance_matrix)
        
        # Compute rotation matrix
        rotation_matrix = np.matmul(Vt.T, U.T)
        
        # Ensure proper rotation (determinant should be 1)
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.matmul(Vt.T, U.T)
        
        # Compute translation
        translation = target_centroid - np.matmul(rotation_matrix, source_centroid)
        
        # Assemble transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = translation
        
        return transformation
        
    except np.linalg.LinAlgError as e:
        logging.error(f"SVD computation failed: {str(e)}")
        return np.eye(4)
