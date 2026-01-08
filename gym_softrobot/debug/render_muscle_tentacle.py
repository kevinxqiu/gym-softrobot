import argparse
import os

import numpy as np

from elastica import (
    BaseSystemCollection,
    Constraints,
    Connections,
    Forcing,
    Damping,
    CallBacks,
    CosseratRod,
    OneEndFixedBC,
    PositionVerlet,
    AnalyticalLinearDamper,
)
from elastica.timestepper import extend_stepper_interface

from gym_softrobot.envs.octopus.build import create_es_muscle_layers
from gym_softrobot.envs.octopus.arm_single_env import OscillatingBaseBC
from gym_softrobot.utils.actuation.actuations.muscles.muscle import ApplyMuscle
from gym_softrobot.utils.render.matplotlib_renderer import Session


class _Simulator(BaseSystemCollection, Constraints, Connections, Forcing, Damping, CallBacks):
    pass


def _make_action(step, n_seg, action_scale, pattern):
    action = np.zeros(n_seg * 3, dtype=np.float64)
    phase = step * 0.15
    a0 = 0.5 + 0.5 * np.sin(phase)
    a1 = 0.5 + 0.5 * np.cos(phase)
    if pattern == "planar":
        action[:n_seg] = action_scale * a0
        action[n_seg : 2 * n_seg] = 0.0
        action[2 * n_seg :] = 0.0
    elif pattern == "symmetric":
        action[:n_seg] = action_scale * a0
        action[n_seg : 2 * n_seg] = action_scale * a0
        action[2 * n_seg :] = 0.0
    else:
        action[:n_seg] = action_scale * a0
        action[n_seg : 2 * n_seg] = action_scale * a1
        action[2 * n_seg :] = action_scale * 0.1
    np.clip(action, 0.0, 1.0, out=action)
    return action


def main():
    parser = argparse.ArgumentParser(
        description="Render a muscle-actuated single tentacle."
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--record-fps", type=int, default=25)
    parser.add_argument("--time-step", type=float, default=5.0e-5)
    parser.add_argument("--n-elems", type=int, default=40)
    parser.add_argument("--base-length", type=float, default=0.25)
    parser.add_argument("--base-radius", type=float, default=0.013)
    parser.add_argument("--tip-radius", type=float, default=0.0042)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--youngs-modulus", type=float, default=8.0e4)
    parser.add_argument("--poisson-ratio", type=float, default=0.5)
    parser.add_argument("--action-scale", type=float, default=0.3)
    parser.add_argument("--damping", type=float, default=2.0)
    parser.add_argument("--no-muscle", action="store_true")
    parser.add_argument("--muscle", action="store_true")
    parser.add_argument("--motor", action="store_true")
    parser.add_argument("--motor-amplitude", type=float, default=0.2)
    parser.add_argument("--motor-frequency", type=float, default=0.4)
    parser.add_argument("--motor-axis", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--motor-offset", type=float, default=0.0)
    parser.add_argument(
        "--pattern",
        type=str,
        default="planar",
        choices=["planar", "symmetric", "bilateral"],
    )
    parser.add_argument("--save", type=str, default="outputs/tentacle_muscle_render.mp4")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--log-length", action="store_true")
    parser.add_argument(
        "--plane", type=str, default="xz", choices=["xy", "xz", "yz"]
    )
    parser.add_argument("--axis-padding", type=float, default=0.05)
    parser.add_argument("--axis-limits", type=float, nargs=4, default=None)
    args = parser.parse_args()

    n_elem = args.n_elems
    shear_modulus = args.youngs_modulus / (1.0 + args.poisson_ratio)

    simulator = _Simulator()
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, -1.0])
    normal = np.array([0.0, 1.0, 0.0])
    radius = np.linspace(args.base_radius, args.tip_radius, n_elem)
    rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        args.base_length,
        radius,
        args.density,
        youngs_modulus=args.youngs_modulus,
        shear_modulus=shear_modulus,
    )
    simulator.append(rod)
    if args.motor:
        simulator.constrain(rod).using(
            OscillatingBaseBC,
            fixed_position=rod.position_collection[:, 0].copy(),
            base_directors=rod.director_collection[:, :, 0].copy(),
            axis=np.asarray(args.motor_axis, dtype=np.float64),
            amplitude=args.motor_amplitude,
            frequency=args.motor_frequency,
            offset=args.motor_offset,
        )
    else:
        simulator.constrain(rod).using(
            OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
    simulator.dampen(rod).using(
        AnalyticalLinearDamper,
        uniform_damping_constant=args.damping,
        time_step=args.time_step,
    )

    muscle_layers = None
    use_muscle = not args.no_muscle
    if args.motor and not args.muscle:
        use_muscle = False
    if use_muscle:
        muscle_layers = create_es_muscle_layers(rod.radius, args.base_radius)
        simulator.add_forcing_to(rod).using(
            ApplyMuscle,
            muscles=muscle_layers,
            step_skip=10000,
            callback_params_list=[],
        )

    simulator.finalize()

    stepper = PositionVerlet()
    do_step, stages_and_updates = extend_stepper_interface(stepper, simulator)
    step_skip = max(1, int(1.0 / (args.record_fps * args.time_step)))

    plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.plane]
    axis_limits = None
    if args.axis_limits:
        axis_limits = (
            (args.axis_limits[0], args.axis_limits[1]),
            (args.axis_limits[2], args.axis_limits[3]),
        )
    else:
        half_width = max(args.base_radius * 6.0, args.tip_radius * 6.0, args.base_length * 0.3)
        z_min = -args.base_length * 1.2
        z_max = args.base_length * 0.2
        axis_limits = ((-half_width, half_width), (z_min, z_max))
    renderer = Session(
        width=800,
        height=600,
        projection="2d",
        plane_axes=plane_axes,
        padding_ratio=args.axis_padding,
        axis_limits=axis_limits,
    )
    renderer.add_rod(rod)

    writer = None
    if not args.no_save:
        import imageio.v2 as imageio

        video_dir = os.path.dirname(args.save)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(args.save, fps=args.record_fps)

    viewer = None
    if args.view:
        from gym_softrobot.utils.render import pyglet_rendering

        viewer = pyglet_rendering.SimpleImageViewer(maxwidth=800)

    time = np.float64(0.0)
    n_seg = n_elem - 1
    for step in range(args.steps):
        if muscle_layers is not None:
            action = _make_action(step, n_seg, args.action_scale, args.pattern)
            for j, muscle in enumerate(muscle_layers):
                muscle.set_activation(action[n_seg * j : n_seg * (j + 1)])
        for _ in range(step_skip):
            time = do_step(stepper, stages_and_updates, simulator, time, args.time_step)
        if args.log_length:
            length = float(np.sum(rod.rest_lengths * rod.dilatation))
            print(f"step={step:04d} length={length:.6f}")
        frame = renderer.render()
        if writer is not None:
            writer.append_data(frame)
        if viewer is not None:
            viewer.imshow(frame)

    if writer is not None:
        writer.close()
    if viewer is not None:
        viewer.close()
    renderer.close()


if __name__ == "__main__":
    main()
