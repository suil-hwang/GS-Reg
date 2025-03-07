Here's your updated README with the conda installation instructions:

# 3D Gaussian Splatting Scene Registration

This repository extends the original 3D Gaussian Splatting by adding functionality to align and merge multiple 3D Gaussian Splatting scenes. The core contribution is a robust registration pipeline that allows precise alignment between two Gaussian Splatting models using keypoint-based registration followed by ICP refinement.

## 🆕 Gaussian Scene Registration Tool

![Registration Example](assets/registration_example.png)

### Features

The `gs-registration.py` script enables the alignment and merging of two pre-trained 3D Gaussian Splatting models through these key steps:

1. **Loading pre-trained scenes**: Imports two separate Gaussian Splatting scenes (exported as PLY files)
2. **Interactive keypoint selection**: Provides an interface to select at least 3 corresponding points between scenes
3. **SVD-based initial alignment**: Computes optimal rotation and translation between point sets
4. **Transformation application**: Applies the computed transformation to all Gaussians in the source scene
5. **Adaptive ICP refinement**: Uses an iterative closest point algorithm with:
   - Automatic threshold determination based on scene scale
   - Gradual threshold reduction for precise alignment
   - Convergence monitoring via RMSE
6. **Scene merging**: Combines the aligned scenes into a unified Gaussian model

### Usage

```bash
python gs-registration.py <source_ply> <target_ply> <output_ply> [options]
```

#### Required Arguments:
- `source_ply`: Path to the first Gaussian Splatting PLY file (will be transformed)
- `target_ply`: Path to the second Gaussian Splatting PLY file (reference model)
- `output_ply`: Path where the merged result will be saved

#### Optional Arguments:
- `--align_ground_plane`: Enable automatic ground plane detection and alignment
- `--multiplier`: Coefficient for ICP threshold setting (default: 0.05)
- `--icp_method`: ICP alignment method: 'point_to_point' or 'point_to_plane' (default: 'point_to_point')
- `--max_iterations`: Maximum number of ICP iterations (default: 50)
- `--tolerance`: RMSE change tolerance for convergence (default: 1e-6)
- `--threshold_decay`: Rate at which ICP threshold decreases (default: 0.8)
- `--min_threshold`: Minimum ICP threshold value (default: 0.01)
- `--source_keypoints`: File to save/load source keypoints (default: source_keypoints.txt)
- `--target_keypoints`: File to save/load target keypoints (default: target_keypoints.txt)

### Example

```bash
# Basic usage with default parameters
python gs-registration.py ./scene_a.ply ./scene_b.ply ./merged_scene.ply

# With ground plane alignment and custom ICP settings
python gs-registration.py ./scene_a.ply ./scene_b.ply ./merged_scene.ply --align_ground_plane --multiplier 0.03 --icp_method point_to_plane
```

### Interactive Keypoint Selection

When you run the script, you'll be prompted to select corresponding keypoints in both scenes:

1. The visualization window will open showing the first (source) scene
2. Use Shift+Left click to select at least 3 keypoints
3. Press 'Q' to close the window when done
4. Repeat the process for the second (target) scene
5. Ensure you select the same corresponding points in both scenes

If keypoint files already exist, they will be loaded automatically. Delete or rename these files if you want to select new keypoints.

## Setup

### Installation

For installation, please follow the same conda environment setup as the original 3D Gaussian Splatting repository:

1. Visit [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
2. Follow their installation instructions to set up the conda environment:

```bash
# Windows only
SET DISTUTILS_USE_SDK=1 
# All platforms
conda env create --file environment.yml
conda activate gaussian_splatting
```

Once the environment is set up, you can use the registration tool provided in this repository.

### Requirements

The registration tool has the same requirements as the original 3D Gaussian Splatting project, with additional dependencies handled by the included utils/registration_utils.py file:

- OpenGL 4.5-ready GPU and drivers
- CUDA-ready GPU with Compute Capability 7.0+
- Open3D library (installed via the environment.yml)
