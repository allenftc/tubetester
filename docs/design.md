# Test Tube Scanner Design Notes

## Boundary

- Klipper owns low-level motion primitives.
- The Python controller owns scan sequencing and future QR integration.
- The Python controller also owns Moonraker API calls and the local web control surface.
- Calibration files store rack geometry and camera settings.

## Layout

- `controller/` contains the Python runtime.
- `controller/network/` contains Moonraker API access.
- `controller/web/` contains the local browser control surface.
- `klipper/` contains machine-side macros and configuration.
- `calibration/` contains per-machine values that should change without code edits.
- `tests/` contains layout and configuration checks.

## First milestone

The first milestone is a single-tube dry run: load calibration, build the workflow, and print the planned steps before any hardware is driven.