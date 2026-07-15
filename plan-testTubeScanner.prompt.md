## Plan: Test Tube Scanner Layout

Create a small, deliberate repository layout for a single-tube proof of concept: Klipper handles motion primitives, Python handles orchestration and QR scanning, and calibration data stays separate from code. The layout should make it easy to add workflow logic later without mixing robot motion, vision, and rack geometry in the same files.

**Steps**
1. Define the top-level directories and their purpose: Klipper macros/config, Python controller, calibration data, docs, and tests. *Depends on none.*
2. Split the Python controller into workflow, motion adapter, vision adapter, and configuration loading so each responsibility stays isolated. *Depends on step 1.*
3. Split Klipper assets into reusable motion primitives and higher-level macros, while keeping machine settings and homing details separate from the scan workflow. *Depends on step 1.*
4. Define the calibration layout for rack geometry, tube offsets, yaw sweep settings, and camera scan parameters in standalone data files. *Depends on step 1.*
5. Add a minimal documentation surface in README and a short design note that explains the file layout and runtime boundary between Klipper and Python. *Depends on steps 1 through 4.*
6. Add a test/validation layout so dry-run checks, config validation, and future unit tests have an obvious home. *Depends on steps 1 through 4.*

**Proposed layout**
- `README.md` — project overview and how the pieces fit together.
- `klipper/` — all Klipper-side files.
- `klipper/macros/` — motion primitives and workflow macros.
- `klipper/config/` — printer, axis, endstop, and toolhead configuration.
- `controller/` — Python application entry point and orchestration code.
- `controller/workflow/` — scan state machine and sequencing.
- `controller/motion/` — commands or adapters that talk to Klipper.
- `controller/vision/` — QR scanning and camera integration.
- `controller/config/` — runtime settings and environment loading.
- `calibration/` — rack coordinates, offsets, yaw parameters, and scan presets.
- `docs/` — design notes, assumptions, and operator/setup guidance.
- `tests/` — unit tests and dry-run validation helpers.

**Relevant files**
- `README.md` — should describe the repo layout and first prototype scope.
- `klipper/` — where the motion-side assets should live once created.
- `controller/` — where the Python workflow and vision integration should live once created.
- `calibration/` — where rack geometry and scan parameters should live once created.

**Verification**
1. Confirm every responsibility has one clear home and no file needs to mix motion, vision, and calibration logic.
2. Confirm the Python controller can be developed and tested without editing Klipper config for each workflow change.
3. Confirm calibration values can be changed without modifying controller logic.
4. Confirm the layout supports a later expansion from single-tube proof of concept to rack-wide automation.

**Decisions**
- The first version should be organized for maintainability, not maximal abstraction.
- Python remains the orchestration layer; Klipper remains the motion layer.
- Calibration data should be file-based and easy to swap per rack or tube type.
- The layout should stay minimal until the workflow proves out.

**Further Considerations**
1. If you want, the controller can be a simple script-based package first, then grow into a service later.
2. If you already know the camera model or QR library, that can be reflected directly in the `controller/vision/` layout.
3. If you want a faster prototype, the Klipper layout can be flatter and collapsed into a single macro file initially.