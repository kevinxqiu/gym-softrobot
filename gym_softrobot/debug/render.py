import argparse
import os

import gymnasium

import gym_softrobot
from gym_softrobot.config import RendererType

# Use Matplotlib renderer to avoid POV-Ray dependency.
gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB

USE_REALISTIC_PRESET = True
REALISTIC_PRESET = {
    "control_mode": "motor",
    "motor_amplitude": 3,
    "motor_frequency": 0.35,
    "motor_axis": (0.0, 1.0, 0.0),
    "n_elems": 80,
    "base_length": 0.3,
    "taper_ratio": 0.2,
    "rod_youngs_modulus": 8.0e4,
    "rod_density": 1050.0,
    "damping_constant": 8.0e-2,
    "fluid_density": 1022.0,
    "render_view": "2d",
    "render_plane": "yz",
    "render_axis_padding": 0.0,
    "render_axis_limits": ((-0.12, 0.12), (-0.4, 0.05)),
}
SAVE_VIDEO = True
VIDEO_PATH = "save/tentacle_render.mp4"
VIDEO_FPS = 30


def main():
    parser = argparse.ArgumentParser(
        description="Make registered environment and test run."
    )
    parser.add_argument("--env", type=str, default="OctoArmSingle-v0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-scale", type=float, default=0.1)
    parser.add_argument("--render-view", type=str, default="2d", choices=["2d", "3d"])
    parser.add_argument(
        "--render-plane", type=str, default="yz", choices=["xy", "xz", "yz"]
    )
    parser.add_argument("--base-length", type=float, default=None)
    parser.add_argument("--axis-padding", type=float, default=0.05)
    parser.add_argument("--axis-limits", type=float, nargs=4, default=None)
    parser.add_argument(
        "--control-mode", type=str, default="motor", choices=["motor", "action"]
    )
    args = parser.parse_args()

    env_kwargs = {"recording_fps": 30}
    if args.env == "OctoArmSingle-v0":
        env_kwargs["render_view"] = args.render_view
        env_kwargs["render_plane"] = args.render_plane
        env_kwargs["base_length"] = args.base_length
        env_kwargs["render_axis_padding"] = args.axis_padding
        env_kwargs["control_mode"] = args.control_mode
        if args.axis_limits and args.render_view == "2d":
            x_min, x_max, y_min, y_max = args.axis_limits
            env_kwargs["render_axis_limits"] = ((x_min, x_max), (y_min, y_max))
        if USE_REALISTIC_PRESET:
            env_kwargs.update(REALISTIC_PRESET)
    env = gymnasium.make(args.env, **env_kwargs)

    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, info = reset_result
    else:
        observation = reset_result
        info = {}
    writer = None
    fig = None
    im = None
    if SAVE_VIDEO:
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib import animation

        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required to save videos (writer=ffmpeg)")
        video_dir = os.path.dirname(VIDEO_PATH)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.axis("off")
        writer = animation.FFMpegWriter(
            fps=VIDEO_FPS,
            metadata={"title": "Tentacle Render", "artist": "gym-softrobot"},
        )
        writer.setup(fig, VIDEO_PATH, dpi=100)

    for step in range(args.steps):
        action = args.action_scale * env.action_space.sample()
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            observation, reward, done, info = step_result
        frame = env.render()  # rendering
        if writer is not None:
            if im is None:
                im = fig.axes[0].imshow(frame)
            else:
                im.set_data(frame)
            writer.grab_frame()
        print(f"{step=:2}| {reward=}, {done=}")
        if done:
            break
    if writer is not None:
        writer.finish()


if __name__ == "__main__":
    main()
