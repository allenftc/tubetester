# How the tube scanner works

This project is a small control system for a gantry robot that scans a rack of tubes, looks for tube centers with a camera, and then picks up a tube with vacuum.

The important idea is this:

- the rack has a known coordinate system in millimeters
- the camera sits a little offset from the gripper, so the robot must aim the camera at a different XY position than the actual tube pickup point
- the camera captures the tube circle from two X positions, triangulates its 3D location, and moves the gripper to the corrected XY
- the workflow is intentionally broken into small deliberate steps so you can debug every move

## 1. Where the settings come from

The machine configuration is loaded from the calibration files and then turned into Python dataclasses in [controller/config/settings.py](../controller/config/settings.py).

The key values live in [calibration/rack.json](../calibration/rack.json):

- origin_mm: the top-left or reference point of the rack in machine coordinates
- tube_pitch_mm: spacing between neighboring tube centers
- pickup_offset_mm: extra offset for the actual pickup point inside the tube pocket
- camera_offset_mm: signed camera position relative to the gripper in machine XY
- safe_z_mm: the height used for XY travel around the rack
- pickup_height_mm: the Z height at which the tool should descend to pick up the tube
- scan_height_mm: the height to lift a picked tube to after pickup; it must be at least safe_z_mm
- camera intrinsics and distortion: stored in `controller/vision/tubeLocator/camera_calibration.npz`

The logic is simple:

- Tube center position = origin + pickup_offset + pitch * (row, column)
- Camera position = tube center - camera_offset

That second line is the key one: the camera is not mounted exactly at the gripper, so the robot aims the camera at a different XY than the final gripper pickup target.

## 2. How the actual tube XY is calculated

The controller does this in the runtime layer in [controller/web/runtime.py](../controller/web/runtime.py):

1. choose the target rack tube and compute its nominal position
2. compute the camera's nominal XY from the configured camera offset
3. move to X-5 mm and capture a fresh circle center
4. move to X+5 mm and capture a second fresh circle center (10 mm baseline)
5. undistort both pixel centers using `camera_calibration.npz` and triangulate XYZ
6. use the stereo X/Y offset to move the gripper to the corrected XY

The estimate is expressed in millimeters relative to the midpoint camera pose:
camera X follows gantry +X, camera Y follows image-down (opposite gantry +Y),
and camera Z is the distance along the optical axis. The camera must remain
fixed in orientation during the two captures, with its image axes aligned to
the gantry axes; the intrinsics archive contains no camera-to-gantry rotation.
The computed Z is reported as a stereo estimate, but pickup continues to use
the configured `pickup_height_mm` rather than changing motion height.

## 3. Why there is a safe Z and a pickup Z

The machine uses three Z heights:

- safe_z_mm: the height used when moving around, before detection, and while aligning above a tube
- pickup_height_mm: the lower Z height used to actually contact the tube and hold it with vacuum
- scan_height_mm: the height used to lift the held tube after pickup

This matters because the robot should never move at the low pickup height unless it is already lined up. It always approaches at safe Z, then moves to the camera point, finds the tube, then adds the correction, and only then descends to pickup_height_mm.

## 4. What the UI is doing

The browser UI in [controller/web/static/app.js](../controller/web/static/app.js) and [controller/web/templates/index.html](../controller/web/templates/index.html) does not directly drive the hardware. Instead, it calls HTTP endpoints exposed by the server in [controller/web/server.py](../controller/web/server.py).

The flow is:

- select a tube cell in the rack grid
- begin a pickup session
- review the planned coordinates and list of ten steps, including both stereo views
- run one step at a time with Run Next Step
- each click calls a step endpoint and performs exactly one action

The runtime keeps a pickup session in memory so the UI can pause mid-sequence instead of doing everything in one giant request.

## 5. The pickup steps

The pickup stepper is intentionally explicit. It usually goes in this order:

1. raise to safe Z
2. move camera to the left stereo position, X-5 mm
3. capture and average fresh tube-circle centers
4. move camera to the right stereo position, X+5 mm
5. capture the second view and estimate stereo XYZ
6. move the gripper to stereo-corrected XY at safe Z
7. turn vacuum on
8. move down to pickup_height_mm
9. lift to scan_height_mm while vacuum stays on
10. return to the selected tube slot's nominal center XY, lower to pickup_height_mm, release the tube, turn vacuum off, and lift to scan_height_mm

If either view is missing or the stereo disparity is invalid, the sequence stops
before the gripper correction and pickup. The deposit still uses the nominal
slot center, not the camera-corrected pickup target.

## 6. Safety behavior

The runtime locks out other machine actions while the pickup step-through is active. That prevents:

- a second workflow from starting
- manual G-code from racing the stepper
- camera preview shutdown while the step sequence still needs it
- tool controls from interfering with a partially completed pickup

If something goes wrong or you want to abort, Cancel Steps raises the held tube back to safe Z before unlocking the controls when it is below safe Z. Vacuum stays on unless you deliberately turn it off.

## 7. What the code path looks like

If you want to trace the path through the project, start here:

- [calibration/rack.json](../calibration/rack.json): geometry and calibration values
- [controller/config/settings.py](../controller/config/settings.py): turn the JSON into dataclasses
- [controller/web/runtime.py](../controller/web/runtime.py): the runtime logic for snapshot, session-state, and step execution
- [controller/web/server.py](../controller/web/server.py): HTTP endpoints the UI calls
- [controller/web/static/app.js](../controller/web/static/app.js): browser UI state and per-step control
- [README.md](../README.md): quick project overview and startup instructions

## 8. In one sentence

The system is a calibration-driven camera-assisted pickup flow: it calculates where the camera must be, waits for a fresh tube center, turns that pixel offset into a millimeter correction, and then carefully moves the gripper to the corrected XY while using safe Z and vacuum in a controlled, debuggable sequence.
