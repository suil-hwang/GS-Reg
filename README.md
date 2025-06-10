# KeyGS: Keypoint-Guided Registration for 3D Gaussian Splatting

An interactive registration framework that enables precise alignment and merging of multiple 3D Gaussian Splatting scenes through user-guided keypoint correspondence and ICP refinement.

## Overview

KeyGS extends [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) with a semi-automatic registration pipeline that allows users to:

- Interactively select corresponding keypoints between scenes
- Automatically compute optimal alignment using SVD and ICP
- Merge multiple 3DGS reconstructions into a unified scene

## Installation

KeyGS uses the same environment as the original 3D Gaussian Splatting:

```bash
# Windows only
SET DISTUTILS_USE_SDK=1
# Create conda environment
conda env create --file environment.yml
conda activate gaussian_splatting
```

## Quick Start

```bash
# Basic registration
python gs-registration.py source.ply target.ply output.ply

# With ground plane alignment
python gs-registration.py source.ply target.ply output.ply --align_ground_plane
```

## Method

### 1. Interactive Keypoint Selection

Select at least 3 corresponding points between scenes using our intuitive interface:

- **Shift + Left Click**: Select keypoints
- **Q**: Finish selection

#### Source Scene (House)

![House Keypoint Selection](assets/house_split.png)

#### Target Scene (Kitchen)

![Kitchen Keypoint Selection](assets/kitchen_split.png)

The colored spheres indicate selected correspondence points.

### 2. Automatic Alignment

KeyGS performs:

1. **SVD-based initial alignment** from keypoint correspondences
2. **Adaptive ICP refinement** for precise registration
3. **Optional ground plane alignment** for indoor scenes

### Results

The registration pipeline produces accurate alignments:

#### Merged Result

![Registration Result](assets/ex_result.png)

#### Ground Truth Reference

![Ground Truth](assets/gt.png)

## Advanced Usage

### Command Line Options

| Parameter              | Description                                       | Default                |
| ---------------------- | ------------------------------------------------- | ---------------------- |
| `--align_ground_plane` | Enable ground plane detection                     | False                  |
| `--icp_method`         | ICP variant: `point_to_point` or `point_to_plane` | `point_to_point`       |
| `--max_iterations`     | Maximum ICP iterations                            | 50                     |
| `--multiplier`         | ICP threshold coefficient                         | 0.05                   |
| `--source_keypoints`   | Source keypoints file                             | `source_keypoints.txt` |
| `--target_keypoints`   | Target keypoints file                             | `target_keypoints.txt` |

## Citation

If you find this work useful, please consider citing:

```bibtex
@software{keygs2024,
  title={KeyGS: Keypoint-Guided Registration for 3D Gaussian Splatting},
  author={Suil Hwang},
  year={2024},
  url={https://github.com/yourusername/keygs}
}
```

## Acknowledgments

This project builds upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for their excellent work.
