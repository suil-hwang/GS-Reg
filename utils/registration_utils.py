import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.spatial.transform import Rotation
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging - ERROR level only
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_POINT_SIZE = 2.0
DEFAULT_BG_COLOR = [0.1, 0.1, 0.1]
MIN_KEYPOINTS = 3
MIN_PLANE_INLIERS = 100
EPSILON = 1e-6

class ICPMethod(Enum):
    """ICP method types"""
    POINT_TO_POINT = "point_to_point"
    POINT_TO_PLANE = "point_to_plane"

@dataclass
class ICPConfig:
    """Configuration for ICP alignment"""
    multiplier: float = 0.05
    method: ICPMethod = ICPMethod.POINT_TO_POINT
    max_iterations: int = 50
    tolerance: float = 1e-6
    threshold_decay: float = 0.8
    min_threshold: float = 0.01
    
class PointCloudProcessor:
    """Base class for point cloud operations"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._point_cloud: Optional[o3d.geometry.PointCloud] = None
        self._bounds_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    @property
    def point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Lazy loading of point cloud"""
        if self._point_cloud is None:
            self.load()
        return self._point_cloud
    
    def load(self) -> bool:
        """Load point cloud from file"""
        try:
            self._point_cloud = o3d.io.read_point_cloud(self.file_path)
            if not self._point_cloud.has_points():
                raise ValueError(f"Empty point cloud: {self.file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load {self.file_path}: {str(e)}")
            return False
    
    def save(self, output_path: str) -> bool:
        """Save point cloud to file"""
        if self._point_cloud is None:
            return False
        try:
            o3d.io.write_point_cloud(output_path, self._point_cloud)
            return True
        except Exception as e:
            logging.error(f"Failed to save to {output_path}: {str(e)}")
            return False
    
    def transform(self, transformation: np.ndarray) -> None:
        """Apply transformation to point cloud"""
        if self._point_cloud is not None:
            self._point_cloud.transform(transformation)
            self._bounds_cache = None  # Invalidate cache
    
    @lru_cache(maxsize=1)
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached bounds of point cloud"""
        if self._point_cloud is None or not self._point_cloud.has_points():
            return np.zeros(3), np.zeros(3)
        
        points = np.asarray(self._point_cloud.points)
        return np.min(points, axis=0), np.max(points, axis=0)

class InteractiveKeypointSelector:
    """Interactive keypoint selection with Open3D"""
    
    def __init__(self, point_size: float = DEFAULT_POINT_SIZE):
        self.point_size = point_size
    
    def select_keypoints(self, processor: PointCloudProcessor, 
                        window_name: str = "Select Keypoints") -> List[np.ndarray]:
        """Interactive keypoint selection"""
        if processor.point_cloud is None:
            return []
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"{window_name} (Press 'Q' to exit)", 
                         width=1024, height=768)
        
        # Configure visualization
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array(DEFAULT_BG_COLOR)
        
        vis.add_geometry(processor.point_cloud)
        vis.get_view_control().set_zoom(0.8)
        
        # Show instructions
        self._print_instructions()
        
        # Run visualization
        vis.run()
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        
        # Convert to coordinates
        if picked_indices:
            points = np.asarray(processor.point_cloud.points)
            keypoints = points[picked_indices]
            print(f"Selected {len(keypoints)} keypoints")
            return keypoints.tolist()
        return []
    
    @staticmethod
    def _print_instructions():
        print("\n" + "="*30)
        print("KEYPOINT SELECTION")
        print("="*30)
        print("• Shift+Left click: Select points")
        print("• Shift+Right click: Deselect points")
        print("• Q: Finish selection")
        print("="*30 + "\n")

class OptimizedICPAligner:
    """Optimized ICP alignment with adaptive threshold"""
    
    def __init__(self, config: ICPConfig):
        self.config = config
    
    def align(self, source: PointCloudProcessor, target: PointCloudProcessor,
             initial_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        """Perform ICP alignment with optimization"""
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # Ensure point clouds are loaded
        if source.point_cloud is None or target.point_cloud is None:
            return np.eye(4), 0.0, float('inf')
        
        # Work with copies
        source_pcd = o3d.geometry.PointCloud(source.point_cloud)
        target_pcd = o3d.geometry.PointCloud(target.point_cloud)
        
        # Apply initial transformation
        source_pcd.transform(initial_transform)
        
        # Prepare for ICP
        if self.config.method == ICPMethod.POINT_TO_PLANE:
            self._estimate_normals(source_pcd, target_pcd)
        
        # Adaptive ICP
        transformation = self._adaptive_icp(source_pcd, target_pcd)
        
        # Compose final transformation
        final_transform = transformation @ initial_transform
        
        # Evaluate result
        fitness, rmse = self._evaluate_registration(source_pcd, target_pcd, transformation)
        
        print(f"ICP completed: fitness={fitness:.4f}, RMSE={rmse:.4f}")
        return final_transform, fitness, rmse
    
    def _estimate_normals(self, source: o3d.geometry.PointCloud, 
                         target: o3d.geometry.PointCloud) -> None:
        """Estimate normals for point-to-plane ICP"""
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        if not target.has_normals():
            target.estimate_normals(search_param=search_param)
        if not source.has_normals():
            source.estimate_normals(search_param=search_param)
    
    def _adaptive_icp(self, source: o3d.geometry.PointCloud, 
                     target: o3d.geometry.PointCloud) -> np.ndarray:
        """Perform adaptive ICP with decreasing threshold"""
        # Calculate initial threshold
        _, target_max = PointCloudProcessor("").get_bounds()
        _, source_max = PointCloudProcessor("").get_bounds()
        
        # Use target bounds for threshold estimation
        target_bounds = target.get_max_bound() - target.get_min_bound()
        diagonal = np.linalg.norm(target_bounds)
        current_threshold = diagonal * self.config.multiplier
        
        best_result = {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'rmse': float('inf')
        }
        
        # Create transformation estimation object once
        if self.config.method == ICPMethod.POINT_TO_PLANE:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        
        # Convergence criteria
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=20,
            relative_fitness=self.config.tolerance,
            relative_rmse=self.config.tolerance
        )
        
        # Adaptive refinement
        iteration = 0
        while (current_threshold >= self.config.min_threshold and 
               iteration < self.config.max_iterations):
            
            result = o3d.pipelines.registration.registration_icp(
                source, target, current_threshold, np.eye(4),
                estimation, criteria
            )
            
            # Update best result
            if (result.fitness > best_result['fitness'] or 
                (result.fitness == best_result['fitness'] and 
                 result.inlier_rmse < best_result['rmse'])):
                best_result['transformation'] = result.transformation
                best_result['fitness'] = result.fitness
                best_result['rmse'] = result.inlier_rmse
                source.transform(result.transformation)
            
            current_threshold *= self.config.threshold_decay
            iteration += 1
        
        return best_result['transformation']
    
    @staticmethod
    def _evaluate_registration(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray) -> Tuple[float, float]:
        """Evaluate registration quality"""
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, 0.05, transformation
        )
        return evaluation.fitness, evaluation.inlier_rmse

class RegistrationPipeline:
    """Main registration pipeline with optimized workflow"""
    
    def __init__(self, source_path: str, target_path: str, output_path: str,
                 icp_config: Optional[ICPConfig] = None,
                 enable_ground_plane: bool = False):
        self.source = PointCloudProcessor(source_path)
        self.target = PointCloudProcessor(target_path)
        self.output_path = output_path
        self.icp_config = icp_config or ICPConfig()
        self.enable_ground_plane = enable_ground_plane
        self.keypoint_selector = InteractiveKeypointSelector()
        self.icp_aligner = OptimizedICPAligner(self.icp_config)
    
    def run(self) -> bool:
        """Execute registration pipeline"""
        print("\n🚀 Starting Registration Pipeline...")
        
        # Load point clouds
        if not self._load_point_clouds():
            return False
        
        cumulative_transform = np.eye(4)
        
        # Optional ground plane alignment
        if self.enable_ground_plane:
            print("\n📐 Aligning ground planes...")
            ground_transform = self._align_ground_planes()
            self.source.transform(ground_transform)
            cumulative_transform = ground_transform @ cumulative_transform
        
        # Interactive keypoint selection
        print("\n🎯 Interactive Keypoint Selection")
        
        print("\n[1/2] SOURCE model:")
        source_keypoints = self.keypoint_selector.select_keypoints(
            self.source, "SOURCE Model"
        )
        
        if len(source_keypoints) < MIN_KEYPOINTS:
            print(f"❌ Error: Need at least {MIN_KEYPOINTS} keypoints")
            return False
        
        print("\n[2/2] TARGET model:")
        target_keypoints = self.keypoint_selector.select_keypoints(
            self.target, "TARGET Model"
        )
        
        if len(target_keypoints) < MIN_KEYPOINTS:
            print(f"❌ Error: Need at least {MIN_KEYPOINTS} keypoints")
            return False
        
        # Keypoint alignment
        print("\n🔧 Computing keypoint alignment...")
        keypoint_transform = compute_rigid_transform(
            np.array(source_keypoints), 
            np.array(target_keypoints)
        )
        self.source.transform(keypoint_transform)
        cumulative_transform = keypoint_transform @ cumulative_transform
        
        # ICP refinement
        print("\n⚡ Refining with ICP...")
        icp_transform, _, _ = self.icp_aligner.align(self.source, self.target)
        self.source.transform(icp_transform)
        cumulative_transform = icp_transform @ cumulative_transform
        
        # Save results
        print("\n💾 Saving results...")
        return self._save_results()
    
    def _load_point_clouds(self) -> bool:
        """Load source and target point clouds"""
        if not self.source.load():
            print("❌ Failed to load source model")
            return False
        if not self.target.load():
            print("❌ Failed to load target model")
            return False
        print("✅ Models loaded successfully")
        return True
    
    def _align_ground_planes(self) -> np.ndarray:
        """Align ground planes of both point clouds"""
        source_plane = detect_ground_plane(self.source.point_cloud)
        target_plane = detect_ground_plane(self.target.point_cloud)
        
        if source_plane is None or target_plane is None:
            print("⚠️  Could not detect ground planes, skipping alignment")
            return np.eye(4)
        
        source_transform = plane_to_z_alignment(source_plane)
        target_transform = plane_to_z_alignment(target_plane)
        
        return np.linalg.inv(target_transform) @ source_transform
    
    def _save_results(self) -> bool:
        """Save aligned and merged results"""
        import os
        base_name = os.path.splitext(os.path.basename(self.output_path))[0]
        dir_name = os.path.dirname(self.output_path)
        
        # Save individual aligned clouds
        source_path = os.path.join(dir_name, f"{base_name}_aligned_source.ply")
        target_path = os.path.join(dir_name, f"{base_name}_target.ply")
        
        self.source.save(source_path)
        self.target.save(target_path)
        
        # Merge and save
        merged = merge_point_clouds(self.source.point_cloud, self.target.point_cloud)
        try:
            o3d.io.write_point_cloud(self.output_path, merged)
            print(f"✅ Registration complete! Output: {self.output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save merged result: {e}")
            return False

# Optimized helper functions
@lru_cache(maxsize=32)
def detect_ground_plane(pcd: o3d.geometry.PointCloud,
                       distance_threshold: float = 0.05,
                       ransac_n: int = 20,
                       num_iterations: int = 1000) -> Optional[np.ndarray]:
    """Detect ground plane using RANSAC with caching"""
    if not pcd.has_points():
        return None
    
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers) >= MIN_PLANE_INLIERS:
            return plane_model
        return None
    except:
        return None

def plane_to_z_alignment(plane_model: np.ndarray) -> np.ndarray:
    """Compute transformation to align plane normal with Z-axis"""
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    
    z_axis = np.array([0, 0, 1])
    
    # Check if already aligned
    if np.abs(np.dot(normal, z_axis)) > 0.999:
        return np.eye(4)
    
    # Compute rotation
    axis = np.cross(normal, z_axis)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(normal, z_axis), -1, 1))
    
    rotation = Rotation.from_rotvec(axis * angle)
    
    # Build transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[2, 3] = -d  # Translate to z=0
    
    return transform

def compute_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute optimal rigid transformation using SVD"""
    if len(source) != len(target) or len(source) < MIN_KEYPOINTS:
        return np.eye(4)
    
    # Compute centroids
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    
    # Center the points
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # SVD
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = target_centroid - R @ source_centroid
    
    # Build transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    return transform

def merge_point_clouds(pcd1: o3d.geometry.PointCloud, 
                      pcd2: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Efficiently merge two point clouds"""
    merged = o3d.geometry.PointCloud()
    
    # Merge points
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    merged.points = o3d.utility.Vector3dVector(np.vstack([points1, points2]))
    
    # Merge colors if available
    if pcd1.has_colors() and pcd2.has_colors():
        colors1 = np.asarray(pcd1.colors)
        colors2 = np.asarray(pcd2.colors)
        merged.colors = o3d.utility.Vector3dVector(np.vstack([colors1, colors2]))
    
    return merged

# Convenience function for backward compatibility
class AlignmentController:
    """Backward compatible wrapper"""
    def __init__(self, source_path, target_path, output_path, **kwargs):
        config = ICPConfig(
            multiplier=kwargs.get('multiplier', 0.05),
            method=ICPMethod(kwargs.get('icp_method', 'point_to_point')),
            max_iterations=kwargs.get('max_iterations', 50),
            tolerance=kwargs.get('tolerance', 1e-6),
            threshold_decay=kwargs.get('threshold_decay', 0.8),
            min_threshold=kwargs.get('min_threshold', 0.01)
        )
        self.pipeline = RegistrationPipeline(
            source_path, target_path, output_path,
            icp_config=config,
            enable_ground_plane=kwargs.get('align_ground_plane', False)
        )
    
    def run(self):
        return self.pipeline.run()
