from gym.envs.registration import register

from deep_quintic.env import WolfgangBulletEnv
from deep_quintic.ros_runner import ExecuteEnv

register(
    id='WolfgangBulletEnv-v1',
    entry_point='deep_quintic:WolfgangBulletEnv',
    max_episode_steps=10000,
    # tags={"pg_complexity": 200*1000000},
)

register(
    id='ExecuteEnv-v1',
    entry_point='deep_quintic:ExecuteEnv',
    max_episode_steps=10000,
    # tags={"pg_complexity": 200*1000000},
)