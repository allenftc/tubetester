# Gantry Vision Calibration

Tools for planar camera calibration, pixel-to-world projection, grid
reprojection, and depth calibration from horizontal camera motion.

## Scripts

- `checkerboard_mapper.py` - Create a homography from a checkerboard or manually
  selected planar grid points.
- `reproject_grid.py` - Project real-world grid coordinates into a still image
  or live camera feed.
- `depth_calibrate.py` - Fit depth or height from known-depth stereo-style
  disparity samples.

## Usage

Create a pixel-to-world homography:

```powershell
python checkerboard_mapper.py `
  --image board.jpg `
  --corners 9 6 `
  --pitch 25 `
  --origin-offset 0 0 `
  --output grid_mapping.json
```

For manually selected absolute coordinates:

```powershell
python checkerboard_mapper.py `
  --image board.jpg `
  --manual-grid `
  --absolute-coordinates `
  --output grid_mapping.json
```

Project a world grid into a still image or camera:

```powershell
python reproject_grid.py --mapping grid_mapping.json --image board.jpg --spacing 25
python reproject_grid.py --mapping grid_mapping.json --camera 0 --spacing 25
```

Calibrate depth from known-depth image pairs:

```powershell
python depth_calibrate.py `
  --sample left_300.jpg right_300.jpg 300 10 `
  --sample left_500.jpg right_500.jpg 500 10 `
  --sample left_700.jpg right_700.jpg 700 10 `
  --camera-calibration camera_calibration.npz `
  --output depth_calibration.json
```

Calibrate height from objects in one or more image pairs:

```powershell
python depth_calibrate.py `
  --reference-depth 500 `
  --multi-height-sample left.jpg right.jpg 0,50,100,150 10 `
  --camera-calibration camera_calibration.npz `
  --output height_calibration.json
```

## Coordinate transforms

The homography JSON contains `pixel_to_world_homography`. In code:

```python
import cv2
import json
import numpy as np

with open("grid_mapping.json", "r", encoding="utf-8") as file:
    calibration = json.load(file)

pixel_to_world = np.array(
    calibration["pixel_to_world_homography"],
    dtype=np.float64,
)

pixel = np.array([[[640.0, 360.0]]], dtype=np.float64)
world = cv2.perspectiveTransform(pixel, pixel_to_world)[0, 0]
print(world)
```
