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

Start the local web control surface:

```bash
python -m controller.main --serve-web
```

## Next steps

1. Wire the controller to a real Klipper transport.
2. Replace the placeholder Klipper macros with machine-specific motion primitives.
3. Add a QR camera backend under `controller/vision/`.

