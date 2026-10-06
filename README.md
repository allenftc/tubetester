# tubetester

Test tube scanner robot scaffold for a Klipper-powered gantry.

This repository is organized so the motion layer, workflow layer, vision layer, and calibration data stay separate from one another.

## Layout

- `controller/` - Python orchestration, motion adapters, and vision integration
- `controller/network/` - Moonraker networking client and request helpers
- `controller/web/` - built-in control webpage and local HTTP server
- `klipper/` - Klipper macros and machine configuration placeholders
- `calibration/` - rack geometry, yaw sweep settings, and camera settings
- `docs/` - design notes and implementation guidance
- `tests/` - layout and configuration validation

## Current status

The first implementation slice is a dry-run scaffold. It can load calibration settings and build a planned tube-scan workflow, but it does not yet drive hardware.

It now also includes a Moonraker networking configuration and a built-in local web control surface scaffold.

## Try it

Run the planned workflow in dry-run mode:

```bash
tube-tester --dry-run
```

Or run the module directly:

```bash
python -m controller.main --dry-run
```

Install and start the controller on the Linux mini PC:

```bash
sudo apt update
sudo apt install -y python3-venv v4l-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m controller.main --serve-web
```

On Linux, the camera uses V4L2 and should appear as `/dev/video0`. Check camera enumeration with `v4l2-ctl --list-devices`. If the controller user cannot open the camera, add that account to the `video` group with `sudo usermod -aG video "$USER"`, then log out and back in. Set `device_index`, resolution, and ROI in [`calibration/camera.json`](calibration/camera.json). The preview is off by default and opens the camera only when enabled; disabling it or closing the page stops capture. The preview overlays the rotated ROI in orange, its target center in yellow, and a detected center in red.

Select a rack cell to preview the camera and nominal tube coordinates, then press **Begin Pickup Steps**. Beginning does not move the machine. Review each planned position, then press **Run Next Step** to execute exactly one move or detection at a time. The seven stages are safe Z, camera XY using `camera_offset_mm` (camera position relative to the gripper), camera detection, corrected gripper XY, vacuum on, descent to `pickup_height_mm`, and lift to safe Z. `pickup_height_mm` is an absolute machine Z in [`calibration/rack.json`](calibration/rack.json). `pixel_to_mm_multiplier` is millimeters per pixel; calibrate it for the installed camera before relying on the correction. If no center is detected, the gripper does not move from the camera position and vacuum is not enabled. **Cancel Steps** stops advancement; if the gripper has descended while holding a tube, cancel first raises it to safe Z and leaves vacuum on. After successful pickup, vacuum remains on until you use Vacuum Off or release the tube. When preview was off before beginning, it starts for the step-through and stops after completion or cancellation.

The web server binds to `0.0.0.0` so another computer on the private LAN can open `http://<mini-pc-ip>:8080` (find the address with `hostname -I`). This control UI has no authentication; keep it on a trusted LAN and do not expose the port to the public internet.

For a deeper walkthrough of the calibration, camera correction math, and the step-through pickup flow, see [docs/how-it-works.md](docs/how-it-works.md).

## Klipper tooling

The scanner controls its tooling through Moonraker and Klipper. The machine's Klipper configuration must define output pins named `solenoid` and `vacuum_pump`, plus a manual stepper named `rotary`.

When Klipper first reports ready, the controller sets both `solenoid` and `vacuum_pump` to `0`. Pickup or the manual Vacuum On control turns on the pump. Release pulses the solenoid to `1` for 500 ms, returns it to `0`, and turns the pump off. Workflow shutdown also sets both outputs off. Rotary positioning uses `MANUAL_STEPPER STEPPER=rotary MOVE=<degrees>`; configure the stepper's position units and rotation distance to match degrees.

The manual controls and combined macros use these same Klipper commands. The rotary calibration macro sets the current rotary position to zero. When QR decoding is unavailable, degraded scans still perform physical pickup and release but skip QR and rotary scan steps. A failed Klipper command stops the remaining macro steps.

## Next steps

1. Wire the controller to a real Klipper transport.
2. Replace the placeholder Klipper macros with machine-specific motion primitives.
3. Add a QR camera backend under `controller/vision/`.

