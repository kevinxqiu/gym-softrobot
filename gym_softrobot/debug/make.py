import gymnasium as gym

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Make registered environment and test run."
    )
    parser.add_argument("--env", type=str, default="OctoFlat-v0")
    args = parser.parse_args()

    # env is created, now we can use it:
    env = gym.make(args.env)

    for episode in range(10):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
        else:
            observation = reset_result
            info = {}
        for step in range(50):
            action = env.action_space.sample()
            step_result = env.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                observation, reward, done, info = step_result
            print(f"{episode=:2} |{step=:2}, {reward=}, {done=}")
            if done:
                break


if __name__ == "__main__":
    main()
