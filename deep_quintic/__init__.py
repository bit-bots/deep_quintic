from gym.envs.registration import register

from deep_quintic.env import WolfgangWalkEnv, CartesianEulerEnv, CartesianFusedEnv, CartesianQuaternionEnv, CartesianSixdEnv, JointEnv, CartesianEulerStateEnv, CartesianEulerNoncyclicEnv
from deep_quintic.ros_runner import ExecuteEnv
from deep_quintic.walker2d_bias import Walker2DBulletBiasEnv

register(id='Walker2DBulletBiasEnv-v0',
         entry_point='deep_quintic:Walker2DBulletBiasEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(
    id='WolfgangWalkEnv-v1',
    entry_point='deep_quintic:WolfgangWalkEnv',
    max_episode_steps=10000,
)

register(
    id='ExecuteEnv-v1',
    entry_point='deep_quintic:ExecuteEnv',
    max_episode_steps=10000,
)

register(
    id='CartesianEulerEnv-v1',
    entry_point='deep_quintic:CartesianEulerEnv',
    max_episode_steps=10000,
)

register(
    id='CartesianFusedEnv-v1',
    entry_point='deep_quintic:CartesianFusedEnv',
    max_episode_steps=10000,
)

register(
    id='CartesianQuaternionEnv-v1',
    entry_point='deep_quintic:CartesianQuaternionEnv',
    max_episode_steps=10000,
)

register(
    id='CartesianSixdEnv-v1',
    entry_point='deep_quintic:CartesianSixdEnv',
    max_episode_steps=10000,
)
register(
    id='JointEnv-v1',
    entry_point='deep_quintic:JointEnv',
    max_episode_steps=10000,
)

register(
    id='JointNobiasEnv-v1',
    entry_point='deep_quintic:JointEnv',
    max_episode_steps=10000,
)

register(
    id='CartesianEulerStateEnv-v1',
    entry_point='deep_quintic:CartesianEulerStateEnv',
    max_episode_steps=10000,
)

register(
    id='CartesianEulerNoncyclicEnv-v1',
    entry_point='deep_quintic:CartesianEulerNoncyclicEnv',
    max_episode_steps=10000,
)