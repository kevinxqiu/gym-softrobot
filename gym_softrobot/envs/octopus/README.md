# Octopus Tentacle Parameters

This note documents the main parameters that control the look and behavior of the
single-arm tentacle (`OctoArmSingle-v0`). Defaults live in `gym_softrobot/envs/octopus/arm_single_env.py`,
while material and geometry parameters are defined in `gym_softrobot/envs/octopus/build.py`.

## Actuation (motor vs action)
- `control_mode`: `"motor"` drives the base with a sinusoid; `"action"` drives curvature along the arm.
- `motor_amplitude` (radians): peak base rotation in motor mode.
- `motor_frequency` (Hz): base oscillation rate.
- `motor_axis`: rotation axis for the base (world frame).
- `fix_base`: whether the base node is fully constrained.

## Geometry
- `base_length`: total tentacle length (meters).
- `taper_ratio`: tip radius / base radius (cone shape).
- `n_elems`: discretization along the rod (higher is smoother but slower).
- `_DEFAULT_SCALE_LENGTH["base_radius"]`: base radius (set in `build.py`).

## Material + Physics
- `_OCTOPUS_PROPERTIES["youngs_modulus"]`: stiffness (higher = stiffer).
- `_OCTOPUS_PROPERTIES["density"]`: mass density (higher = heavier/slower).
- `damping_constant`: linear damping applied via `AnalyticalLinearDamper` in `build.py`.
- gravity: fixed at `-9.81` in `build.py`.
- `add_ground`: toggles contact/friction with the ground plane.
- `rod_youngs_modulus`, `rod_density`: optional env overrides for stiffness/density.

## Rendering (visual only)
- `render_view`: `"2d"` or `"3d"`.
- `render_plane`: `"xy"`, `"xz"`, `"yz"` for 2D view.
- `render_axis_limits`: fixed axis ranges to avoid auto-zoom.
- `render_axis_padding`: optional padding around auto-scaled axes.

## Optional Extensions (not enabled by default)
If you want a water-like medium, there is a drag force implementation in
`gym_softrobot/utils/actuation/forces/drag_force.py`. Typical knobs to wire in are:
- `fluid_density` (kg/m^3)
- `drag_coeff_per`, `drag_coeff_tan` (perpendicular/tangential drag coefficients)
These can be passed directly to `OctoArmSingle-v0` to enable drag.

## Where to Change Things
- `gym_softrobot/envs/octopus/arm_single_env.py`: actuation defaults, render defaults, geometry overrides.
- `gym_softrobot/envs/octopus/build.py`: material properties, base radius, damping, gravity, contact.
