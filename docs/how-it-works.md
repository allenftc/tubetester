# How the tube scanner works

This project is a small control system for a gantry robot that scans a rack of tubes, looks for tube centers with a camera, and then picks up a tube with vacuum.

The important idea is this:

- the rack has a known coordinate system in millimeters
- the camera sits a little offset from the gripper, so the robot must aim the camera at a different XY position than the actual tube pickup point
- the camera finds the tube center in pixels, then the controller converts that offset into millimeters and moves the gripper to the corrected XY
- the workflow is intentionally broken into small deliberate steps so you can debug every move

## 1. Where the settings come from

The machine configuration is loaded from the calibration files and then turned into Python dataclasses in [controller/config/settings.py](../controller/config/settings.py).

The key values live in [calibration/rack.json](../calibration/rack.json):

- origin_mm: the top-left or reference point of the rack in machine coordinates
- tube_pitch_mm: spacing between neighboring tube centers
- pickup_offset_mm: extra offset for the actual pickup point inside the tube pocket
- camera_offset_mm: how far the camera is from the gripper in XY
- safe_z_mm: a high safe height used between moves
- pickup_height_mm: the Z height at which the tool should descend to pick up the tube
- pixel_to_mm_multiplier: how many millimeters one pixel represents after the camera correction

The logic is simple:

- Tube center position = origin + pickup_offset + pitch * (row, column)
- Camera position = tube center - camera_offset

That second line is the key one: the camera is not mounted exactly at the gripper, so the robot aims the camera at a different XY than the final gripper pickup target.

## 2. How the actual tube XY is calculated

The controller does this in the runtime layer in [controller/web/runtime.py](../controller/web/runtime.py):

1. choose the target rack tube
2. compute the nominal tube position
3. compute where the camera must be moved to view that tube
4. wait for a fresh camera frame and a detected center
5. compare the detected center to the camera ROI center in pixels
6. convert the pixel difference into millimeters with pixel_to_mm_multiplier
7. move the gripper to the final corrected XY

The formula is:

- correction_x = (detected_x - roi_center_x) * pixel_to_mm_multiplier
- correction_y = (detected_y - roi_center_y) * pixel_to_mm_multiplier
- corrected_target_x = tube_x + correction_x
- corrected_target_y = tube_y + correction_y

This is why the robot can still land on the tube even when the camera view is slightly off-center or the camera sees the tube shifted in the frame.

## 3. Why there is a safe Z and a pickup Z

The machine uses two different Z heights:

- safe_z_mm: the height used when moving around, before detection, and after pickup
- pickup_height_mm: the lower Z height used to actually contact the tube and hold it with vacuum

This matters because the robot should never move at the low pickup height unless it is already lined up. It always approaches at safe Z, then moves to the camera point, finds the tube, then adds the correction, and only then descends to pickup_height_mm.

## 4. What the UI is doing

The browser UI in [controller/web/static/app.js](../controller/web/static/app.js) and [controller/web/templates/index.html](../controller/web/templates/index.html) does not directly drive the hardware. Instead, it calls HTTP endpoints exposed by the server in [controller/web/server.py](../controller/web/server.py).

The flow is:

- select a tube cell in the rack grid
- begin a pickup session
- review the planned coordinates and list of seven steps
- run one step at a time with Run Next Step
- each click calls a step endpoint and performs exactly one action

The runtime keeps a pickup session in memory so the UI can pause mid-sequence instead of doing everything in one giant request.

## 5. The seven pickup steps

The pickup stepper is intentionally explicit. It usually goes in this order:

1. raise to safe Z
2. move camera to the nominal XY target based on the tube cell
3. wait for a fresh detection
4. compute corrected gripper XY and move the gripper there at safe Z
5. turn vacuum on
6. move down to pickup_height_mm
7. lift back to safe Z while vacuum stays on

If step 3 finds no tube center, the sequence stops before the final gripper move. That is the safety check that prevents a bad pickup.

## 6. Safety behavior

The runtime locks out other machine actions while the pickup step-through is active. That prevents:

- a second workflow from starting
- manual G-code from racing the stepper
- camera preview shutdown while the step sequence still needs it
- tool controls from interfering with a partially completed pickup

If something goes wrong or you want to abort, Cancel Steps raises the held tube back to safe Z before unlocking the controls. Vacuum stays on unless you deliberately turn it off.

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
