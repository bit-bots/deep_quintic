import math
from stable_baselines3.common.callbacks import BaseCallback


class RewardLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.highest_episode_reward = -math.inf

    def _on_step(self) -> bool:
        # iterate through the infos of the envs
        reward_name = None
        for info in self.locals["infos"]:
            # see if episode was finished
            if "rewards" in info.keys():
                # log all rewards
                for key in info["rewards"].keys():
                    if "Reward" in key:
                        reward_name = key
                        self.logger.record(f"Rewards/{reward_name}", info["rewards"][reward_name])
                        self.logger.record(f"PerStepRewards/{reward_name}",
                                           info["rewards"][reward_name] / info["episode"]["l"], exclude="stdout")
            if "ik_error" in info.keys():
                self.logger.record(f"IK/error", info["ik_error"] / info["episode"]["l"])
        # check if we have written some data
        if reward_name is not None:
            # we need to call dump explicitly otherwise it will only be written on end of epoch
            self.logger.dump(self.num_timesteps)
        return True


class CmdVelIncreaseCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cmd_vel_start_bounds = self.training_env.env_method("get_cmd_vel_bounds")[0]
        self.cmd_vel_max_bounds = self.training_env.env_method("get_cmd_vel_bounds")[1]
        self.end_steps = 10000000
        self.cmd_vel_increase = []
        for i in range(3):
            self.cmd_vel_increase.append(
                ((self.cmd_vel_max_bounds[i][0] - self.cmd_vel_start_bounds[i][0]) / self.end_steps,
                 (self.cmd_vel_max_bounds[i][1] - self.cmd_vel_start_bounds[i][1]) / self.end_steps))

    def _on_step(self) -> bool:
        pass

    def _on_rollout_start(self) -> None:
        current_cmd_vel_bounds = []
        for i in range(3):
            current_cmd_vel_bounds.append(
                (max(self.cmd_vel_max_bounds[i][0],
                     self.cmd_vel_start_bounds[i][0] + self.cmd_vel_increase[i][0] * self.num_timesteps),
                 min(self.cmd_vel_max_bounds[i][1],
                     self.cmd_vel_start_bounds[i][1] + self.cmd_vel_increase[i][1] * self.num_timesteps)))
            self.logger.record(f"CmdVelBoundsLow/{i}", current_cmd_vel_bounds[i][0], exclude="stdout")
            self.logger.record(f"CmdVelBoundsHigh/{i}", current_cmd_vel_bounds[i][1], exclude="stdout")
        self.logger.dump(self.num_timesteps)
        self.training_env.env_method("set_cmd_vel_bounds", current_cmd_vel_bounds)


class RewardWeightCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _init_callback(self) -> None:
        self.end_steps = 10000000
        self.reward_index = 0
        self.start_weights = self.training_env.env_method("get_reward_weights")[0]
        self.weights_change = []
        for i in range(len(self.start_weights)):
            if i == self.reward_index:
                end_weight = 0
            else:
                # change so that we reach 1 at end_steps with all other rewards combined
                end_weight = self.start_weights[i] * (1 / (1-self.start_weights[self.reward_index]))
            self.weights_change.append((end_weight - self.start_weights[i]) / self.end_steps)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_start(self) -> None:
        current_weights = []
        for i in range(len(self.start_weights)):
            current_weights.append(
                min(1, (max(0, self.start_weights[i] + self.weights_change[i] * self.num_timesteps))))
            self.logger.record(f"RewardFactors/{i}", current_weights[i], exclude="stdout")
        self.logger.dump(self.num_timesteps)
        self.training_env.env_method("set_reward_weights", current_weights)
