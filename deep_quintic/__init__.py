from gym.envs.registration import register

from deep_quintic.env import WolfgangWalkEnv, WolfgangDynupEnv
from deep_quintic.ros_runner import ExecuteEnv

register(
    id='WolfgangWalkEnv-v1',
    entry_point='deep_quintic:WolfgangWalkEnv',
    max_episode_steps=10000,
    # tags={"pg_complexity": 200*1000000},
)

register(
    id='WolfgangDynupEnv-v1',
    entry_point='deep_quintic:WolfgangDynupEnv',
    max_episode_steps=10000,
    # tags={"pg_complexity": 200*1000000},
)

register(
    id='ExecuteEnv-v1',
    entry_point='deep_quintic:ExecuteEnv',
    max_episode_steps=10000,
    # tags={"pg_complexity": 200*1000000},
)