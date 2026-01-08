import argparse
import os

import gymnasium
import numpy as np

import gym_softrobot
from gym_softrobot.config import RendererType


class _NoOpViewer:
    def imshow(self, _frame):
        return None

    def close(self):
        return None


def _make_action(env, step, action_scale):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    n_arm = getattr(env.unwrapped, "n_arm", 1)
    n_action = getattr(env.unwrapped, "n_action", action.size // n_arm)
    n_muscle = getattr(env.unwrapped, "n_muscle", 3)
    n_seg = n_action // n_muscle
    phase = step * 0.15
    a0 = 0.5 + 0.5 * np.sin(phase)
    a1 = 0.5 + 0.5 * np.cos(phase)
    for arm in range(n_arm):
        base = arm * n_action
        for seg in range(n_seg):
            idx = base + seg * n_muscle
            action[idx + 0] = action_scale * a0
            action[idx + 1] = action_scale * a1
            action[idx + 2] = action_scale * 0.1
    return action


def main():
    parser = argparse.ArgumentParser(
        description="Render muscle-based octopus model (Reach/Crawl)."
    )
    parser.add_argument("--env", type=str, default="OctoReach-v0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-scale", type=float, default=0.8)
    parser.add_argument("--record-fps", type=int, default=25)
    parser.add_argument("--save", type=str, default="outputs/octopus_muscle_render.mp4")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    # Use Matplotlib renderer to avoid POV-Ray dependency.
    gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB

    env = gymnasium.make(
        args.env,
        recording_fps=args.record_fps,
    )

    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, info = reset_result
    else:
        observation = reset_result
        info = {}

    if not args.view:
        env.unwrapped.disable_viewer = True
        env.unwrapped.viewer = _NoOpViewer()

    writer = None
    fig = None
    im = None
    if not args.no_save:
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib import animation

        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required to save videos (writer=ffmpeg)")
        video_dir = os.path.dirname(args.save)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.axis("off")
        writer = animation.FFMpegWriter(
            fps=args.record_fps,
            metadata={"title": "Octopus Muscle Render", "artist": "gym-softrobot"},
        )
        writer.setup(fig, args.save, dpi=100)

    for step in range(args.steps):
        action = _make_action(env, step, args.action_scale)
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            observation, reward, done, info = step_result
        frame = env.render()
        if writer is not None:
            if im is None:
                im = fig.axes[0].imshow(frame)
            else:
                im.set_data(frame)
            writer.grab_frame()
        if done:
            break

    if writer is not None:
        writer.finish()
    env.close()


if __name__ == "__main__":
    main()
