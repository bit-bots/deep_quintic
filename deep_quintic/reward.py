from abc import ABC
import math
import numpy as np
import rospy
from moveit_msgs.srv import GetPositionFKRequest
from pyquaternion import Quaternion
from std_msgs.msg import Float32
from transforms3d.euler import quat2euler

from bitbots_moveit_bindings import get_position_fk
from deep_quintic.utils import compute_imu_orientation_from_world, Rot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deep_quintic import DeepQuinticEnv


class AbstractReward(ABC):
    def __init__(self, env: "DeepQuinticEnv"):
        self.name = self.__class__.__name__
        self.env = env
        self.episode_reward = 0
        self.current_reward = 0
        self.publisher = rospy.Publisher("Reward_" + self.name, Float32, queue_size=1)
        self.episode_publisher = rospy.Publisher("Reward_" + self.name + "_episode", Float32, queue_size=1)

    def reset_episode_reward(self):
        self.episode_reward = 0

    def get_episode_reward(self):
        return self.episode_reward

    def get_name(self):
        return self.name

    def publish_reward(self):
        if self.publisher.get_num_connections() > 0:
            self.publisher.publish(Float32(self.current_reward))
        if self.episode_publisher.get_num_connections() > 0:
            self.episode_publisher.publish(Float32(self.episode_reward))

    def compute_reward(self):
        print("not implemented, this is abstract")

    def compute_current_reward(self):
        self.current_reward = self.compute_reward()
        self.episode_reward += self.current_reward
        return self.current_reward


class CombinedReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = None

    def compute_current_reward(self):
        self.current_reward = 0
        for reward_type in self.reward_classes:
            self.current_reward += reward_type.compute_current_reward()
        self.episode_reward += self.current_reward
        return self.current_reward

    def get_episode_reward(self):
        reward_sum = 0
        for reward_type in self.reward_classes:
            reward_sum += reward_type.get_episode_reward()
        return reward_sum

    def reset_episode_reward(self):
        self.episode_reward = 0
        for reward in self.reward_classes:
            reward.reset_episode_reward()

    def get_info_dict(self):
        info = dict()
        for reward in self.reward_classes:
            info[reward.get_name()] = reward.get_episode_reward()
        return info

    def publish_reward(self):
        if self.publisher.get_num_connections() > 0:
            self.publisher.publish(Float32(self.current_reward))
        if self.episode_publisher.get_num_connections() > 0:
            self.episode_publisher.publish(Float32(self.episode_reward))
        for reward in self.reward_classes:
            reward.publish_reward()


class WeightedCombinedReward(CombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.weights = None

    def compute_current_reward(self):
        # weight the rewards
        self.current_reward = 0
        for i in range(0, len(self.reward_classes)):
            self.current_reward += self.weights[i] * self.reward_classes[i].compute_current_reward()
        self.episode_reward += self.current_reward
        return self.current_reward

    def get_episode_reward(self):
        reward_sum = 0
        for i in range(0, len(self.reward_classes)):
            reward_sum += self.weights[i] * self.reward_classes[i].get_episode_reward()
        return reward_sum


class EmptyTest(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = []
        self.weights = []


class DeepMimicReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [JointPositionReward(env),
                               JointVelocityReward(env),
                               EndEffectorReward(env),
                               CenterOfMassReward(env)]
        self.weights = [0.65, 0.1, 0.15, 0.1]


class DeepMimicActionReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [JointPositionActionReward(env),
                               JointVelocityReward(env),
                               EndEffectorReward(env),
                               CenterOfMassReward(env)]
        self.weights = [0.65, 0.1, 0.15, 0.1]


class DeepMimicActionCartesianReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               JointVelocityReward(env),
                               EndEffectorReward(env),
                               CenterOfMassReward(env)]
        self.weights = [0.65, 0.1, 0.15, 0.1]


class CartesianReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FeetPosReward(env),
                               FeetOriReward(env),
                               TrajectoryPositionReward(env),
                               TrajectoryOrientationReward(env)]
        self.weights = [0.3, 0.3, 0.2, 0.2]


class CartesianRelativeReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [TrajectoryPositionReward(env),
                               TrajectoryOrientationReward(env)]
        self.weights = [0.5, 0.5]


class CartesianActionReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               TrajectoryPositionReward(env),
                               TrajectoryOrientationReward(env)]
        self.weights = [0.6, 0.2, 0.2]


class CartesianStateVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FeetPosReward(env),
                               FeetOriReward(env),
                               CommandVelReward(env)]
        self.weights = [0.25, 0.25, 0.5]


class CartesianActionVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               CommandVelReward(env),
                               IKErrorReward(env)]
        self.weights = [0.5, 0.5, 0]


class CartesianDoubleActionVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootPositionActionReward(env),
                               FootOrientationActionReward(env),
                               CommandVelReward(env)]
        self.weights = [0.3, 0.3, 0.4]


class CartesianActionMovementReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               MovementReward(env)]
        self.weights = [0.6, 0.4]


class SmoothCartesianActionVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               SmoothCommandVelReward(env)]
        self.weights = [0.6, 0.4]


class CartesianStableActionVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [StableReward(env),
                               FootActionReward(env),
                               CommandVelReward(env)]
        self.weights = [0.3, 0.4, 0.3]


class CartesianActionOnlyReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env)]
        self.weights = [1.0]


class JointActionVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [JointPositionActionReward(env),
                               CommandVelReward(env)]
        self.weights = [0.5, 0.5]


class JointStateVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [JointPositionReward(env),
                               CommandVelReward(env)]
        self.weights = [0.5, 0.5]


class DeepQuinticReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               CommandVelReward(env),
                               UprightReward(env, True),
                               StableReward(env)
                               ]
        self.weights = [0.5, 0.3, 0.1, 0.1]


class CassieReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        # todo this is similar but not exactly the same
        self.reward_classes = [JointPositionReward(env),
                               TrajectoryPositionReward(env),
                               CommandVelReward(env),
                               TrajectoryOrientationReward(env),
                               StableReward(env),
                               AppliedTorqueReward(env),
                               ContactForcesReward(env)]
        self.weights = [0.3, 0.24, 0.15, 0.13, 0.06, 0.06, 0.06]


class CassieActionReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [JointPositionActionReward(env),
                               TrajectoryPositionReward(env),
                               CommandVelReward(env),
                               TrajectoryOrientationReward(env),
                               StableReward(env),
                               AppliedTorqueReward(env),
                               ContactForcesReward(env)]
        self.weights = [0.3, 0.24, 0.15, 0.13, 0.06, 0.06, 0.06]


class CassieCartesianReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FeetPosReward(env),
                               FeetOriReward(env),
                               TrajectoryPositionReward(env),
                               CommandVelReward(env),
                               TrajectoryOrientationReward(env),
                               StableReward(env),
                               AppliedTorqueReward(env),
                               ContactForcesReward(env)]
        self.weights = [0.15, 0.15, 0.24, 0.15, 0.13, 0.06, 0.06, 0.06]


class CassieCartesianActionReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               TrajectoryPositionReward(env),
                               CommandVelReward(env),
                               TrajectoryOrientationReward(env),
                               StableReward(env),
                               AppliedTorqueReward(env),
                               ContactForcesReward(env)]
        self.weights = [0.3, 0.24, 0.15, 0.13, 0.06, 0.06, 0.06]


class CassieCartesianActionVelReward(WeightedCombinedReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)
        self.reward_classes = [FootActionReward(env),
                               CommandVelReward(env),
                               StableReward(env),
                               AppliedTorqueReward(env),
                               ContactForcesReward(env)]
        self.weights = [0.4, 0.3, 0.1, 0.1, 0.1]


class ActionNotPossibleReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)

    def compute_reward(self):
        # give -1 if not possible. this will lead to negative rewards
        if self.env.action_possible:
            return 0
        else:
            return -1


class IKErrorReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=1):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        # gives reward based on the size of the IK error for the current action.
        # this can be helpful to avoid wrong actions and as debug
        request = GetPositionFKRequest()
        for i in range(len(self.env.robot.leg_joints)):
            request.robot_state.joint_state.name.append(self.env.robot.leg_joints[i])
            request.robot_state.joint_state.position.append(self.env.robot.last_ik_result[i])
        request.fk_link_names = ['l_sole', 'r_sole']
        result = get_position_fk(request)  # type: GetPositionFKResponse
        l_sole = result.pose_stamped[result.fk_link_names.index('l_sole')].pose
        fk_left_foot_pos = np.array([l_sole.position.x, l_sole.position.y, l_sole.position.z])
        l_sole_quat = l_sole.orientation
        fk_left_foot_rpy = quat2euler([l_sole_quat.w, l_sole_quat.x, l_sole_quat.y, l_sole_quat.z])
        r_sole = result.pose_stamped[result.fk_link_names.index('r_sole')].pose
        fk_right_foot_pos = np.array([r_sole.position.x, r_sole.position.y, r_sole.position.z])
        r_sole_quat = r_sole.orientation
        fk_right_foot_rpy = quat2euler([r_sole_quat.w, r_sole_quat.x, r_sole_quat.y, r_sole_quat.z])

        # compare to the given goals for the IK
        fk_action = self.env.robot.scale_pose_to_action(fk_left_foot_pos, fk_left_foot_rpy, fk_right_foot_pos,
                                                        fk_right_foot_rpy, self.env.rot_type)

        action_diff = np.linalg.norm(np.array(self.env.last_action) - fk_action)
        return math.exp(-self.factor * (action_diff ** 2))


class CommandVelReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=5):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        # give reward for being close to the commanded velocities in 2D space
        # other velocities are only leading to falls anyway and dont need to be handled here
        command_vel = self.env.current_command_speed
        diff_sum = 0
        diff_sum += (command_vel[0] - self.env.robot.walk_vel[0]) ** 2
        diff_sum += (command_vel[1] - self.env.robot.walk_vel[1]) ** 2
        diff_sum += (command_vel[2] - self.env.robot.walk_vel[2]) ** 2

        reward = math.exp(-self.factor * diff_sum)
        return reward


class SmoothCommandVelReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=20):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        # give reward for being close to the commanded velocities in 2D space
        # other velocities are only leading to falls anyway and dont need to be handled here
        command_vel = self.env.current_command_speed
        diff_sum = 0
        # extra factors to keep influence of different direction identical in reward.
        # turning is faster then sidewards walk
        diff_sum += (command_vel[0] - self.env.robot.smooth_vel[0]) ** 2
        diff_sum += (command_vel[1] - self.env.robot.smooth_vel[1]) ** 2
        diff_sum += (command_vel[2] - self.env.robot.smooth_vel[2]) ** 2

        reward = math.exp(-self.factor * diff_sum)
        return reward


class MovementReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=100):
        super().__init__(env)
        # the size of the difference is proportional to the control loop rate, so take it into account
        self.factor = factor * env.step_freq

    def compute_reward(self):
        # compute movement from last pose of the robot to this pose and compare it with the reference trajectory
        robot_pos_diff = self.env.robot.pos_in_world - self.env.robot.previous_pos_in_world
        robot_yaw_diff = quat2euler(self.env.robot.quat_in_world, axes='szxy')[0] - quat2euler(
            self.env.robot.previous_quat_in_world, axes='szxy')[0]
        ref_pos_diff = self.env.refbot.pos_in_world - self.env.refbot.previous_pos_in_world
        ref_yaw_diff = quat2euler(self.env.refbot.quat_in_world, axes='szxy')[0] - quat2euler(
            self.env.refbot.previous_quat_in_world, axes='szxy')[0]

        # only take movement in x,y and yaw
        diff_sum = 0
        diff_sum += (robot_pos_diff[0] - ref_pos_diff[0]) ** 2
        diff_sum += (robot_pos_diff[1] - ref_pos_diff[1]) ** 2
        diff_sum += (robot_yaw_diff - ref_yaw_diff) ** 2

        reward = math.exp(-self.factor * diff_sum)
        return reward


class FeetPosReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=100):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        left_pos_diff = self.env.robot.left_foot_pos - self.env.refbot.previous_left_foot_pos
        right_pos_diff = self.env.robot.right_foot_pos - self.env.refbot.previous_right_foot_pos
        return math.exp(-self.factor * (np.linalg.norm(left_pos_diff) ** 2 +
                                        np.linalg.norm(right_pos_diff) ** 2))


class FeetOriReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=100):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        geo_diff_left = Quaternion.distance(Quaternion(*self.env.robot.left_foot_quat),
                                            Quaternion(*self.env.refbot.left_foot_quat))
        geo_diff_right = Quaternion.distance(Quaternion(*self.env.robot.right_foot_quat),
                                             Quaternion(*self.env.refbot.right_foot_quat))
        return math.exp(-self.factor * (np.linalg.norm(geo_diff_left) ** 2 +
                                        np.linalg.norm(geo_diff_right) ** 2))


class FootActionReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=1):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        # we interpret the reference as an action.
        # the reference was already stepped when computing the reward therefore we use the previous values
        if self.env.relative:
            ref_action = self.env.robot.scale_relative_pose_to_action(self.env.refbot.previous_left_foot_pos,
                                                                      self.env.refbot.previous_left_foot_quat,
                                                                      self.env.refbot.previous_right_foot_pos,
                                                                      self.env.refbot.previous_right_foot_quat,
                                                                      self.env.rot_type)
        else:
            ref_action = self.env.robot.scale_pose_to_action(self.env.refbot.previous_left_foot_pos,
                                                             quat2euler(self.env.refbot.previous_left_foot_quat),
                                                             self.env.refbot.previous_right_foot_pos,
                                                             quat2euler(self.env.refbot.previous_right_foot_quat),
                                                             self.env.rot_type)

        action_diff = np.linalg.norm(np.array(self.env.last_action) - ref_action)
        return math.exp(-self.factor * (action_diff ** 2))


class FootPositionActionReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=10):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        if self.env.rot_type != Rot.RPY or self.env.relative or not self.env.cartesian_action:
            raise NotImplementedError()
        ref_action = self.env.robot.scale_pose_to_action(self.env.refbot.previous_left_foot_pos,
                                                         quat2euler(self.env.refbot.previous_left_foot_quat),
                                                         self.env.refbot.previous_right_foot_pos,
                                                         quat2euler(self.env.refbot.previous_right_foot_quat),
                                                         self.env.rot_type)
        # only take translation part
        ref_action = np.concatenate([ref_action[:3], ref_action[6:9]])
        action = np.concatenate([self.env.last_action[:3], self.env.last_action[6:9]])
        action_diff = np.linalg.norm(action - ref_action)
        return math.exp(-self.factor * (action_diff ** 2))


class FootOrientationActionReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=40):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        if self.env.rot_type != Rot.RPY or self.env.relative or not self.env.cartesian_action:
            raise NotImplementedError()
        ref_action = self.env.robot.scale_pose_to_action(self.env.refbot.previous_left_foot_pos,
                                                         quat2euler(self.env.refbot.previous_left_foot_quat),
                                                         self.env.refbot.previous_right_foot_pos,
                                                         quat2euler(self.env.refbot.previous_right_foot_quat),
                                                         self.env.rot_type)
        # only take translation part
        ref_action = np.concatenate([ref_action[3:6], ref_action[9:12]])
        action = np.concatenate([self.env.last_action[3:6], self.env.last_action[9:12]])
        action_diff = np.linalg.norm(action - ref_action)
        return math.exp(-self.factor * (action_diff ** 2))


class FootVelReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=5):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        dt = self.env.refbot.previous_time - self.env.refbot.time
        reference_left_lin_vel = (self.env.refbot.left_foot_pos - self.env.refbot.previous_left_foot_pos) / dt
        reference_left_ang_vel = (quat2euler(self.env.refbot.left_foot_quat) - quat2euler(
            self.env.refbot.previous_left_foot_quat)) / dt
        reference_right_lin_vel = (self.env.refbot.right_foot_pos - self.env.refbot.previous_right_foot_pos) / dt
        reference_right_ang_vel = (quat2euler(self.env.refbot.right_foot_quat) - quat2euler(
            self.env.refbot.previous_right_foot_quat)) / dt

        lin_diff_left = np.linalg.norm(reference_left_lin_vel - self.env.robot.left_foot_lin_vel)
        rpy_diff_left = np.linalg.norm(reference_left_ang_vel - self.env.robot.left_foot_ang_vel)
        lin_diff_right = np.linalg.norm(reference_right_lin_vel - self.env.robot.right_foot_lin_vel)
        rpy_diff_right = np.linalg.norm(reference_right_ang_vel - self.env.robot.right_foot_ang_vel)

        reward = math.exp(-self.factor * (
                lin_diff_left ** 2 + rpy_diff_left ** 2 + lin_diff_right ** 2 + rpy_diff_right ** 2))
        return reward


class JointPositionReward(AbstractReward):
    """This reward function rewards joint positions similar to those in a reference action.
       Similar to DeepMimic"""

    def __init__(self, env: "DeepQuinticEnv", factor=2):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        # solve ik if necessary
        self.env.refbot.solve_ik_exactly()
        # take difference between joint in simulation and joint in reference trajectory
        position_diff = np.linalg.norm(self.env.refbot.joint_positions - self.env.robot.joint_positions)
        # add squared diff to sum
        reward = math.exp(-self.factor * position_diff ** 2)
        return reward


class JointPositionActionReward(AbstractReward):

    def __init__(self, env: "DeepQuinticEnv", factor=2):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        # the reference was already stepped when computing the reward therefore we use the current values not the next
        if self.env.cartesian_action:
            print("DeepMimicActionReward does not work with cartesian action")
            exit(0)
        elif self.env.last_action is None:
            # necessary to survive env_check()
            print("env check")
            return 0
        else:
            scaled_ref_positions = self.env.robot.joints_radiant_to_scaled(self.env.refbot.previous_joint_positions)
            if self.env.relative:
                # if we do relative actions, we need to add the action to the positions of the joints at that time
                action = self.env.robot.joints_radiant_to_scaled(
                    self.env.robot.joint_positions + self.env.relative_scaling_joint_action * self.env.last_action)
            else:
                action = self.env.last_action
            action_diff = np.linalg.norm(scaled_ref_positions - action)
            reward = math.exp(-self.factor * action_diff ** 2)
            return reward


class JointVelocityReward(AbstractReward):
    """This reward function rewards joint velocities similar to the joint velocities of a reference action.
       Similar to DeepMimic"""

    def __init__(self, env: "DeepQuinticEnv", factor=0.1):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        # check if we need to compute the velocities
        ref_joint_vels = self.env.refbot.joint_velocities
        if ref_joint_vels is None:
            self.env.refbot.solve_ik_exactly()
            dt = self.env.refbot.previous_time - self.env.refbot.time
            joint_diff = self.env.refbot.joint_positions - self.env.refbot.previous_joint_positions
            ref_joint_vels = joint_diff / dt

        # take difference between joint vel in reference trajectory and current joint vel
        velocity_diff = np.linalg.norm(ref_joint_vels - self.env.robot.joint_velocities)
        reward = math.exp(-self.factor * velocity_diff ** 2)
        return reward


class EndEffectorReward(AbstractReward):
    """This reward function rewards a correct position of end effectors in the world compared to a reference state.
       Similar to DeepMimic"""

    def __init__(self, env: "DeepQuinticEnv", factor=40):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        left_leg, right_leg = self.env.robot.get_legs_in_world()
        ref_left_leg, ref_right_leg = self.env.refbot.get_legs_in_world()
        left_distance = np.linalg.norm(left_leg - ref_left_leg)
        right_distance = np.linalg.norm(right_leg - ref_right_leg)
        return math.exp(-self.factor * (left_distance ** 2 + right_distance ** 2))


class CenterOfMassReward(AbstractReward):
    """This reward function rewards a center of mass close to that of a reference action.
       Similar to DeepMimic"""

    def __init__(self, env: "DeepQuinticEnv", factor=10):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        # we assume center of mass = base link which is roughly true
        # todo could be improved by computing actual CoM
        # distance between position and ref_position
        distance = np.linalg.norm(self.env.refbot.pos_in_world - self.env.robot.pos_in_world)
        return math.exp(-self.factor * distance ** 2)


class TrajectoryPositionReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=10):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        distance = np.linalg.norm(self.env.refbot.pos_in_world - self.env.robot.pos_in_world)
        return math.exp(-self.factor * distance ** 2)


class TrajectoryHeightReward(AbstractReward):
    # similar to center of mass reward in DeepMimic. We assume that the CoM is close to base_link
    # but only using the height in z. this makes looping reference trajectories more easy

    def __init__(self, env: "DeepQuinticEnv", factor=200):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        height_diff = self.env.refbot.pos_in_world[2] - self.env.robot.pos_in_world[2]
        return math.exp(-self.factor * height_diff ** 2)


class TrajectoryOrientationReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=2):
        self.factor = factor
        super().__init__(env)

    def compute_reward(self):
        geo_diff = Quaternion.distance(Quaternion(*self.env.robot.quat_in_world),
                                       Quaternion(*self.env.refbot.quat_in_world))
        return math.exp(-self.factor * np.linalg.norm(geo_diff) ** 2)


class ForwardDistanceReward(AbstractReward):

    def __init__(self, env: "DeepQuinticEnv", factor=2):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        x_distance = self.env.robot.pos_in_world[0] - self.env.robot.pos_on_episode_start[0]
        return math.exp(-self.factor * x_distance ** 2)


class UprightReward(AbstractReward):

    def __init__(self, env: "DeepQuinticEnv", reference, factor=10):
        super().__init__(env)
        self.reference = reference
        self.factor = factor

    def compute_reward(self):
        # only take roll and pitch
        robot_imu_rpy = quat2euler(self.env.robot.quat_in_world)[:2]
        if self.reference:
            reference_rpy = np.array(compute_imu_orientation_from_world(self.env.refbot.quat_in_world)[0][:2])
        else:
            reference_rpy = np.array([0, 0])
        diff = np.linalg.norm(reference_rpy - robot_imu_rpy)
        return math.exp(-self.factor * diff ** 2)


class StableReward(AbstractReward):

    def __init__(self, env: "DeepQuinticEnv", factor=5):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        ang_velocity = self.env.robot.ang_vel
        return math.exp(-self.factor * (ang_velocity[0] ** 2 + ang_velocity[1] ** 2))


class AppliedTorqueReward(AbstractReward):

    def __init__(self, env: "DeepQuinticEnv", factor=0.01):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        # take sum of absolute torque values
        torque_sum = np.sum(np.absolute(self.env.robot.joint_torques))
        reward = math.exp(-self.factor * torque_sum ** 2)
        return reward


class ContactForcesReward(AbstractReward):
    def __init__(self, env: "DeepQuinticEnv", factor=10):
        super().__init__(env)
        self.factor = factor

    def compute_reward(self):
        contact_forces = []
        for name in self.env.robot.pressure_sensors.keys():
            contact_forces.append(self.env.robot.pressure_sensors[name].get_force()[0])

        contact_sum = np.sum(np.absolute(contact_forces))
        return math.exp(-self.factor * contact_sum ** 2)
