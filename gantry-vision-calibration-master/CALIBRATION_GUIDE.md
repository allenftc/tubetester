# Gantry Camera Calibration Guide

This guide summarizes the calibration workflow developed for the crate/tube
vision setup. It covers both the planar pixel-to-world calibration and the
additional depth/height calibration options.

## 1. Choose the calibration target

A standard checkerboard is preferred when available. The target must be:

- Flat
- Fully visible
- In focus
- Illuminated evenly
- Coplanar with the surface being measured

For an object grid such as the blue crate, use a repeatable set of visible
structural intersections. Do not use unrelated image texture, label edges,
holes, shadows, or barcode features as calibration points.

If a chessboard is available, determine the inner-corner count, not the
number of squares. For example, a 10-by-7 square board has 9-by-6 inner
corners.

## 2. Prepare the Python environment

Install the required packages:

```powershell
pip install opencv-python numpy
```

Run commands from the repository directory.

## 3. Calibrate a checkerboard

Use a saved image:

```powershell
python .\checkerboard_mapper.py `
  --image board.jpg `
  --corners 9 6 `
  --pitch 25 `
  --origin-offset 0 0 `
  --output grid_mapping.json
```

Arguments:

- `--corners 9 6`: inner corners in the image pattern.
- `--pitch 25`: physical distance represented by one grid step.
- `--origin-offset 0 0`: real-world coordinate assigned to the chosen origin.
- `--output`: where the homography is saved.

The script prints the mean corner reprojection error and a 3x3 homography.

## 4. Calibrate an irregular or natural grid manually

The automatic checkerboard detector will not reliably detect an object crate
because its grid intersections are partially occluded and there are many
texture features that resemble corners.

Use manual mode:

```powershell
python .\checkerboard_mapper.py `
  --image crate.jpg `
  --manual-grid `
  --absolute-coordinates `
  --output grid_mapping.json
```

This opens a zoomable image window:

- Left-click: add a point.
- Left-drag an existing point: move it.
- Mouse wheel: zoom.
- Middle/right mouse drag: pan.
- `R`: clear all points and reset the view.
- `Enter`: finish selecting points.
- `Esc`: cancel.

After selection, the annotated image remains open while coordinates are
entered in the terminal. Give at least four points, preferably spread across
the full visible plane.

## 5. Understand grid coordinates versus absolute coordinates

There are two coordinate-entry modes.

### Grid-index mode

Enter integer grid indices:

```text
0 0
1 0
2 0
0 1
```

The script multiplies these values by `--pitch`:

```text
world_x = grid_x * pitch
world_y = grid_y * pitch
```

Use this mode when known regular grid spacing is easier to express as grid
indices.

### Absolute-coordinate mode

Enter actual measured coordinates:

```text
Point 0: 100 50
Point 1: 120.2 50
Point 2: 140.4 50
Point 3: 100 70.2
```

Use:

```powershell
--absolute-coordinates
```

In absolute mode:

- Entered coordinates are used directly.
- `--pitch` is not applied.
- `--origin-offset` is not applied.
- The coordinate units remain the units used when entering the points.

A common calibration mistake was entering real coordinates while also
expecting the script to use them as grid indices. That effectively applied
the pitch twice. Absolute mode prevents that.

## 6. Inspect the calibration quality

The JSON file contains:

- `pixel_to_world_homography`
- `world_to_pixel_homography`
- Mean reprojection error for checkerboard calibration
- The selected coordinate convention

A lower reprojection error is better, but a matrix should not be accepted until
several independent known points are tested. For manual calibration, select
points over the whole area and verify:

- Straight grid axes remain straight.
- The scale is correct along X and Y.
- Opposite edges agree with measured distances.
- Points outside the calibrated region are treated as extrapolations.

The homography maps the selected planar surface only. It does not map arbitrary
3D height.

## 7. Reproject a grid into an image

For a still image:

```powershell
python .\reproject_grid.py `
  --mapping grid_mapping.json `
  --image crate.jpg `
  --spacing 25 `
  --output projected.jpg `
  --labels
```

For the live camera:

```powershell
python .\reproject_grid.py `
  --mapping grid_mapping.json `
  --camera 0 `
  --spacing 25 `
  --labels
```

The live script reads a new camera frame on every loop iteration. Press `Q` or
`Esc` to close it.

If a dense grid fills the screen with red markers:

- Increase `--spacing`.
- Keep `--min-pixel-separation` at a positive value.
- Use `--min-pixel-separation 0` only when every overlapping point is wanted.
- Check the "Approximate projected spacing" console output. If that spacing is
  only a few pixels, the physical grid is denser than the image resolution.

For example:

```powershell
python .\reproject_grid.py `
  --mapping grid_mapping.json `
  --camera 0 `
  --spacing 100 `
  --min-pixel-separation 5
```

## 8. Use the homography in application code

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

def pixel_to_real(pixel_x: float, pixel_y: float) -> tuple[float, float]:
    pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
    world = cv2.perspectiveTransform(pixel, pixel_to_world)[0, 0]
    return float(world[0]), float(world[1])
```

The result uses the exact coordinate units and axis directions entered during
manual absolute calibration.

## 9. Depth and height calibration

For objects with a known depth or height, use `depth_calibrate.py`.

Depth calibration:

```powershell
python .\depth_calibrate.py `
  --sample left_300.jpg right_300.jpg 300 10 `
  --sample left_500.jpg right_500.jpg 500 10 `
  --sample left_700.jpg right_700.jpg 700 10 `
  --camera-calibration camera_calibration.npz `
  --output depth_calibration.json
```

Height calibration from several image pairs:

```powershell
python .\depth_calibrate.py `
  --reference-depth 500 `
  --height-sample left_h0.jpg right_h0.jpg 0 10 `
  --height-sample left_h50.jpg right_h50.jpg 50 10 `
  --height-sample left_h100.jpg right_h100.jpg 100 10 `
  --camera-calibration camera_calibration.npz `
  --output height_calibration.json
```

Multiple known heights in one left/right pair:

```powershell
python .\depth_calibrate.py `
  --reference-depth 500 `
  --multi-height-sample left.jpg right.jpg 0,50,100,150 10 `
  --camera-calibration camera_calibration.npz `
  --output height_calibration.json
```

The height mode converts each value to camera depth:

```text
depth = reference_depth - height
```

It fits:

```text
normalized_disparity =
    scale * camera_baseline / depth + offset
```

Runtime conversion is:

```python
depth = scale * baseline / (normalized_disparity - offset)
height = reference_depth - depth
```

Use the same physical units for depth, height, and baseline.

## 10. Camera tilt and XY limitations

Horizontal disparity calibration has extra error if camera tilt or object XY
position causes depth to vary across the frame.

The scalar depth/height model assumes:

- The camera translates horizontally without rotating.
- Distortion is corrected.
- The circle or tube is consistently located.
- The same XY region is used.
- Height is measured approximately along the camera depth axis.
- The model is not being extrapolated outside the calibration area.

For accurate height across the full tilted image plane, use:

1. Camera intrinsics and distortion.
2. The calibrated camera plane or pose.
3. Camera rays at left/right circle centers.
4. Triangulation or ray-plane intersection.
5. Transform the resulting 3D point into gantry/world coordinates.

The scalar depth result is most appropriate for a small XY region or a level
camera setup.

## 11. Circle-selection notes

Hough circles detect both inner openings and outer tube rims. To choose the
outer circle:

- Restrict the radius range to the expected tube size.
- Select a detected circle near the expected tube location.
- Among concentric candidates, choose the largest radius.
- Prefer a low-saturation or known-color ROI before Hough detection.
- Verify detections visually before using them for calibration.
- Use subpixel or contour-based center refinement for repeated measurements.

For code:

```python
outer_circle = max(circles, key=lambda circle: circle[2])
```

Do not automatically trust every Hough result. Color, position, radius range,
and contour shape should filter false detections.

## 12. Practical workflow summary

1. Undistort or calibrate the camera intrinsics.
2. Calibrate the planar pixel-to-world homography.
3. Verify the real-world scale with independent measurements.
4. Choose a consistent origin and coordinate direction.
5. Detect the tube/circle center reliably.
6. Use absolute coordinates if actual coordinates are already known.
7. Use grid indices only if they should be multiplied by the pitch.
8. Calibrate height/depth across the actual operating range.
9. Validate at depths/heights not used during fitting.
10. Recalibrate if the camera position or orientation changes.
