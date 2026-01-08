from gym_softrobot.envs.octopus.flat_env import FlatEnv
from gym_softrobot.envs.octopus.arm_single_env import ArmSingleEnv

# Import each optional env separately so one failure doesn't mask others.
try:
    from gym_softrobot.envs.octopus.arm_two_env import ArmTwoEnv
except Exception:
    ArmTwoEnv = None
try:
    from gym_softrobot.envs.octopus.arm_push_env import ArmPushEnv
    from gym_softrobot.envs.octopus.arm_push_env import ArmPullWeightEnv
except Exception:
    ArmPushEnv = None
    ArmPullWeightEnv = None
try:
    from gym_softrobot.envs.octopus.reach_env import ReachEnv
except Exception:
    ReachEnv = None
try:
    from gym_softrobot.envs.octopus.crawl_env import CrawlEnv
except Exception:
    CrawlEnv = None
# from gym_softrobot.envs.octopus.crawlCurvature_env import CrawlCurvatureEnv
