#!/usr/bin/env python3
# necessary to register envs
import math
import time

from ament_index_python import get_package_share_directory
from bitbots_utils.utils import get_parameters_from_ros_yaml
from humanoid_league_msgs.msg import RobotControlState
from parallel_parameter_search.utils import load_yaml_to_param
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray
from bitbots_msgs.msg import Float32MultiArrayStamped

import deep_quintic
import argparse
import glob
import os
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.action import ActionClient
import stable_baselines3
import yaml
from bitbots_moveit_bindings import get_position_fk
from bitbots_msgs.msg import JointCommand, FootPressure
from biped_interfaces.msg import Phase
from geometry_msgs.msg import PoseStamped, Twist
from sb3_contrib import QRDQN, TQC
from sensor_msgs.msg import Imu, JointState
from stable_baselines3 import PPO, DQN, DDPG, A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, VecFrameStack
from transforms3d.affines import compose, decompose
from transforms3d.euler import quat2euler, mat2euler, euler2mat, euler2quat
from transforms3d.quaternions import quat2mat
from urdf_parser_py.urdf import URDF

from deep_quintic import WolfgangWalkEnv
from deep_quintic.robot import Robot
from deep_quintic.state import CartesianState, BaseState
from deep_quintic.utils import Rot, compute_ik
from deep_quintic.reward import CartesianActionReward

from rcl_interfaces.msg import SetParametersResult

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "qrdqn": QRDQN,
    "tqc": TQC,
}


class DummyPressureSensor:
    pressure = 0

    def get_force(self):
        return None, self.pressure

    def get_value(self, type):
        if type == "raw":
            return self.pressure
        elif type == "filtered":
            return self.pressure
        elif type == "binary":
            print("not implemented")
            exit()


class ExecuteEnv(WolfgangWalkEnv):
    def __init__(self, simulator_type="pybullet", reward_function="CartesianActionVelReward", step_freq=30,
                 ros_debug=False,
                 gui=False, trajectory_file=None, ep_length_in_s=10, use_engine=False,
                 cartesian_state=True, cartesian_action=True, relative=False, use_state_buffer=False,
                 state_type="full", cyclic_phase=True, rot_type="rpy", filter_actions=False, terrain_height=0,
                 phase_in_state=True, foot_sensors_type="", leg_vel_in_state=False, use_rt_in_state=False,
                 randomize=False, use_complementary_filter=True, random_head_movement=True, adaptive_phase=False,
                 use_gyro=True, use_imu_orientation=True, node: Node = None, robot_type="wolfgang", 
                 roll_offset=0.0, pitch_offset=0.0, ang_vel_x_offset=0.0, ang_vel_y_offset=0.0, 
                 roll_scale=1.0, pitch_scale=1.0, ang_vel_x_scale=1.0, ang_vel_y_scale=1.0, use_previous_state=False):
        super().__init__(simulator_type=simulator_type + "_off", reward_function=reward_function, step_freq=step_freq,
                         ros_debug=True, gui=gui,
                         trajectory_file=trajectory_file, state_type=state_type, ep_length_in_s=ep_length_in_s,
                         use_engine=False, cartesian_state=cartesian_state,
                         cartesian_action=cartesian_action, relative=relative,
                         use_state_buffer=use_state_buffer, cyclic_phase=cyclic_phase, rot_type=rot_type,
                         use_rt_in_state=use_rt_in_state, filter_actions=filter_actions,
                         terrain_height=terrain_height, foot_sensors_type=foot_sensors_type,
                         phase_in_state=phase_in_state, randomize=randomize, leg_vel_in_state=leg_vel_in_state,
                         use_complementary_filter=False, random_head_movement=False, adaptive_phase=adaptive_phase,
                         use_gyro=use_gyro, use_imu_orientation=use_imu_orientation, node=node, robot_type=robot_type, ros=True)
        # use dummy pressure sensors since we are not connected to a simulation
        self.robot.pressure_sensors = defaultdict(lambda: DummyPressureSensor())
        self.node = node
        self.roll_offset = roll_offset
        self.pitch_offset = pitch_offset
        self.ang_vel_x_offset = ang_vel_x_offset
        self.ang_vel_y_offset = ang_vel_y_offset
        self.roll_scale = roll_scale
        self.pitch_scale = pitch_scale
        self.ang_vel_x_scale = ang_vel_x_scale
        self.ang_vel_y_scale = ang_vel_y_scale
        self.last_state = None
        # needs to be set manually
        self.venv = None
        self.model = None

        # additional class variables we need since we are not running the walking at the same time
        self.refbot.phase = 0
        self.last_time = None
        walk_parameters = get_parameters_from_ros_yaml("walking",
                                                       f"{get_package_share_directory('bitbots_quintic_walk')}"
                                                       f"/config/deep_quintic_{simulator_type}_wolfgang.yaml",
                                                       use_wildcard=True)
        for parameter in walk_parameters:
            if parameter.name == "engine.freq":
                self.freq = parameter.value.double_value
        # todo we dont initialize action filter correctly. history is empty

        # initialize ROS stuff
        self.got_foot_pressure = False
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.imu_rpy = None
        self.ang_vel = None
        self.last_joint_state_time = 0
        self.current_command_speed = [0, 0, 0]
        self.got_stop_command = True
        self.performing_stop_step = False
        self.is_stopped = True
        self.robot_state = RobotControlState.CONTROLLABLE

        self.joint_publisher = self.node.create_publisher(JointCommand, 'walking_motor_goals', 1)
        self.support_publisher = self.node.create_publisher(Phase, 'walk_support_state', 1)
        self.current_command_speed_sub = self.node.create_subscription(Twist, 'cmd_vel', self.current_command_speed_cb,
                                                                       1, callback_group=MutuallyExclusiveCallbackGroup())
        #debug
        self.action_publisher_not_normalized = self.node.create_publisher( Float32MultiArrayStamped, "action_cartesian_pose_not_normalized", 1)
        self.state_array_normalized_pub = self.node.create_publisher( Float32MultiArrayStamped, "state_array_normalized", 1)
        self.current_joint_pub= self.node.create_publisher( Float32MultiArrayStamped, "current_joint", 1)

        self.imu_sub = self.node.create_subscription(Imu, '/imu/data', self.imu_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.joint_state_sub = self.node.create_subscription(JointState, '/joint_states', self.joint_state_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.robot_state_sub = self.node.create_subscription(RobotControlState, 'robot_state', self.robot_state_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        if foot_sensors_type == "filtered":
            self.left_pressure_sub = self.node.create_subscription(FootPressure, 'foot_pressure_left/filtered',
                                                                   lambda msg: self.foot_pressure_cb(msg, True), 1, callback_group=MutuallyExclusiveCallbackGroup())
            self.right_pressure_sub = self.node.create_subscription(FootPressure, 'foot_pressure_right/filtered',
                                                                    lambda msg: self.foot_pressure_cb(msg, False), 1, callback_group=MutuallyExclusiveCallbackGroup())
        elif foot_sensors_type in ["raw", "binary"]:
            self.left_pressure_sub = self.node.create_subscription(FootPressure, 'foot_pressure_left/raw',
                                                                   lambda msg: self.foot_pressure_cb(msg, True), 1, callback_group=MutuallyExclusiveCallbackGroup())
            self.right_pressure_sub = self.node.create_subscription(FootPressure, 'foot_pressure_right/raw',
                                                                    lambda msg: self.foot_pressure_cb(msg, False), 1, callback_group=MutuallyExclusiveCallbackGroup())
        elif foot_sensors_type == "":
            self.got_foot_pressure = True
        else:
            print(f"Problem: use foot sensors is {foot_sensors_type}")
            exit()

        # urdf = URDF.from_xml_string(self.node.get_parameter('robot_description'))
        # self.joint_limits = {
        #    joint.name: (joint.limit.lower, joint.limit.upper) for joint in urdf.joints if joint.limit
        # }

        self.node.add_on_set_parameters_callback(self.parameters_callback)

        # see that we got all data from the various subscribers
        for i in range(5):
            rclpy.spin_once(self.node)
        while self.current_joint_positions is None or self.ang_vel is None or not self.got_foot_pressure:
            self.node.get_logger().info("Waiting for data from subscribers", throttle_duration_sec=10)
            time.sleep(0.01)
            rclpy.spin_once(self.node)
        
    def start_timer(self, model, venv):
        self.model = model
        self.venv = venv
        self.node.create_timer(1.0/ self.step_freq, callback=self.run_node, 
                callback_group=MutuallyExclusiveCallbackGroup())

    def parameters_callback(self, params):
        for param in params:
            if param.name == "roll_offset":
                self.roll_offset = math.radians(param.value)
            elif param.name == "pitch_offset":
                self.pitch_offset = math.radians(param.value)
            elif param.name == "ang_vel_x_offset":
                self.ang_vel_x_offset = math.radians(param.value)
            elif param.name == "ang_vel_y_offset":
                self.ang_vel_y_offset = math.radians(param.value)
            elif param.name == "roll_scale":
                self.roll_scale = math.radians(param.value)
            elif param.name == "pitch_scale":
                self.pitch_scale = math.radians(param.value)
            elif param.name == "ang_vel_x_scale":
                self.ang_vel_x_scale = math.radians(param.value)
            elif param.name == "ang_vel_y_scale":
                self.ang_vel_y_scale = math.radians(param.value)
        return SetParametersResult(successful=True)

    def apply_action(self, action):
        # only do something if we are not stopped
        if not self.is_stopped:
            if self.filter_actions:
                action = self.action_filter.filter(action)
            msg = JointCommand()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.joint_names = self.robot.used_joint_names
            msg.velocities = [-1.0] * len(self.robot.used_joint_names)
            msg.accelerations = [-1.0] * len(self.robot.used_joint_names)
            msg.max_currents = [-1.0] * len(self.robot.used_joint_names)
            if self.cartesian_action:
                left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = \
                    self.robot.scale_action_to_pose(action, self.rot_type) #todo other rotation types are not handled correctly
                pose_action_msg = Float32MultiArrayStamped()
                pose_action_msg.array.data = np.concatenate([left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy]).tolist()
                pose_action_msg.header.stamp = self.node.get_clock().now().to_msg()
                self.action_publisher_not_normalized.publish(pose_action_msg)
                left_foot_quat = euler2quat(*left_foot_rpy)
                right_foot_quat = euler2quat(*right_foot_rpy)
                ik_result, success = compute_ik(left_foot_pos, left_foot_quat, right_foot_pos, right_foot_quat,
                                                self.robot.used_joint_names, self.robot.joint_indexes,
                                                collision=False, approximate=True)
                msg.positions = list(ik_result)
            else:
                msg.positions = self.robot.joints_scaled_to_radiant(action, use_sim=False)
            self.joint_publisher.publish(msg)

            # support state is necessary for odometry
            msg = Phase()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            if self.refbot.phase > 0.5:
                msg.phase = Phase.LEFT_STANCE
            else:
                msg.phase = Phase.RIGHT_STANCE
            # todo we are not detecting double support state
            # msg.state = SupportState.DOUBLE
            self.support_publisher.publish(msg)

    def compute_observation(self):
        # save current state as previous state
        self.robot.step()

        # compute current state
        self.robot.imu_ang_vel = self.ang_vel
        self.robot.imu_rpy = self.imu_rpy
        self.robot.joint_positions = self.current_joint_positions
        if not self.cartesian_state:
            self.robot.joint_positions = self.current_joint_positions
            self.robot.joint_velocities = self.current_joint_velocities
        if self.cartesian_state:
            self.robot.solve_fk(force=True)
            if self.robot.previous_left_foot_pos is None:
                # need to have two states updates first so that the previous positions are set for velocity computation
                return None
        # calc velocities
        time = self.node.get_clock().now().nanoseconds / 1e9
        if self.last_time is None:
            self.last_time = time - 0.001
        time_diff = time - self.last_time
        if time_diff == 0:
            time_diff = 1 / self.step_freq
        if self.cartesian_state:
            self.robot.compute_velocities(time_diff)

        observation = self.state.get_state_array(scaled=True)
        if len(self.state_buffer) > 0:
            self.state_buffer.pop(0)
        self.state_buffer.append(observation)

        # debug
        state_msg = Float32MultiArrayStamped()
        state_msg.array.data = observation.tolist()
        state_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.state_array_normalized_pub.publish(state_msg)
        joint_msg = Float32MultiArrayStamped()
        joint_msg.array.data = self.current_joint_positions.tolist()
        joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.current_joint_pub.publish(joint_msg)

        return observation

    def progress_phase(self, time, action):
        if self.adaptive_phase:
            # progress phase based on the policies action
            time_of_single_step = 0.5 / self.freq
            dt = (action[-1] + 1) / 2 * time_of_single_step
        else:
            if self.last_time is None:
                # handle first iteration
                dt = 0.001
            else:
                dt = time - self.last_time
                if dt <= 0:
                    # handle edge case or sim time being resettled
                    dt = 0.001
            self.last_time = time

        # Check for too long dt
        if dt > 0.5 / self.freq:
            self.node.get_logger().error(f"DeepQuintic error too long dt phase={self.refbot.phase} dt={dt}")
            return

        # Update the phase
        last_phase = self.refbot.phase
        self.refbot.phase += dt * self.freq

        if (last_phase < 0.5 and self.refbot.phase >= 0.5) or (last_phase >= 0.5 and self.refbot.phase < 0.5):
            # if we need to stop, this is our chance. we need to do one step with zero vel.
            # this is basically a tiny implicit state machine
            if self.got_stop_command:
                if self.performing_stop_step:
                    self.is_stopped = True
                else:
                    self.performing_stop_step = True
            else:
                self.is_stopped = False
                self.performing_stop_step = False

        # reset if step complete
        if self.refbot.phase > 1.0:
            self.refbot.phase -= 1

    def current_command_speed_cb(self, msg: Twist):
        self.current_command_speed = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.got_stop_command = msg.angular.x < 0

    def imu_cb(self, msg: Imu):
        rpy = [*quat2euler((msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z))]
        # add the offset to compensate for unprecicely mounted IMU
        self.imu_rpy = [rpy[0] * self.roll_scale + self.roll_offset, 
                        rpy[1] * self.pitch_scale + self.pitch_offset, 
                        rpy[2]]
        self.ang_vel = [msg.angular_velocity.x * self.ang_vel_x_scale + self.ang_vel_x_offset, 
                        msg.angular_velocity.y * self.ang_vel_y_scale + self.ang_vel_y_offset, 
                        msg.angular_velocity.z]

    def foot_pressure_cb(self, msg: FootPressure, is_left: bool):
        print("foot_pressure_cb")
        self.got_foot_pressure = True
        # LLB, LLF, LRF, LRB, RLB, RLF, RRF, RRB
        pressures = [msg.left_back, msg.left_front, msg.right_front, msg.right_back]
        prefix = 'L' if is_left else 'R'
        self.robot.pressure_sensors[prefix + 'LB'].pressure = pressures[0]
        self.robot.pressure_sensors[prefix + 'LF'].pressure = pressures[1]
        self.robot.pressure_sensors[prefix + 'RF'].pressure = pressures[2]
        self.robot.pressure_sensors[prefix + 'RB'].pressure = pressures[3]

    def joint_state_cb(self, msg: JointState):
        # sort joint positions in correct order
        self.current_joint_positions = np.empty(self.robot.num_used_joints)
        self.current_joint_velocities = np.empty(self.robot.num_used_joints)
        for i in range(len(msg.name)):
            joint_name = msg.name[i]
            if joint_name in self.robot.used_joint_names:
                index = self.robot.joint_indexes[joint_name]
                self.current_joint_positions[index] = msg.position[i]
                self.current_joint_velocities[index] = msg.velocity[i]

    def robot_state_cb(self, msg: RobotControlState):
        self.robot_state = msg.state

    def run_node(self):                                
        if self.robot_state == RobotControlState.WALKING or (
                self.robot_state == RobotControlState.CONTROLLABLE and (
                not self.current_command_speed == (0, 0, 0) or not self.is_stopped)):
            # only run if we are in the correct state
            obs = self.compute_observation()
            if obs is None:
                return
            # update normalization statistics                
            self.venv.obs_rms.update(obs)
            # we need to normalize the observation
            norm_obs = self.venv.normalize_obs(obs)
            action, self.last_state = self.model.predict(norm_obs, state=self.last_state, deterministic=True)
            self.last_action = action
            self.apply_action(action)
            # progress phase and compute current state
            self.time = self.node.get_clock().now().nanoseconds / 1.0e9
            self.progress_phase(self.time, action)
            self.ros_interface.publish()
        else:
            # hard stop walking when falling, getting up, etc.
            self.current_command_speed = (0, 0, 0)
            self.is_stopped = True


class StoreDict(argparse.Action):
    """
    From https://github.com/DLR-RM/rl-baselines3-zoo
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def get_saved_hyperparams(
        stats_path: str,
        norm_reward: bool = False,
        test_mode: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    From https://github.com/DLR-RM/rl-baselines3-zoo
    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


def create_test_env(
        env_id: str,
        n_envs: int = 1,
        stats_path: Optional[str] = None,
        seed: int = 0,
        log_dir: Optional[str] = None,
        should_render: bool = True,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """
    hyperparams = {} if hyperparams is None else hyperparams

    vec_env_kwargs = {}
    vec_env_cls = DummyVecEnv

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                print(f"VecNormalize stats {path_} not found")
                exit(1)
        else:
            print("normalize param is False. There is probably something wrong")
            exit(1)

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--exp-id", help="Experiment ID", required=True, type=int)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
             "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict,
        help="Optional keyword argument to pass to the env constructor"
    )
    args = parser.parse_args()

    # load model, like it is done in the enjoy.py from rl-baselines zoo
    env_id = "WolfgangBulletEnv-v1"
    algo = args.algo
    folder = args.folder
    log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)
    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)
    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")


        def step_count(s):
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(s.split("_")[-2])


        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")
    print(f"Loading {model_path}")

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    print(env_kwargs)
    venv = create_test_env(
        "ExecuteEnv-v1",
        n_envs=1,
        stats_path=stats_path,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    # direct reference to wolfgnag env object
    env = venv.venv.envs[0].env.env

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = ALGOS[algo].load(model_path, env=venv, custom_objects=custom_objects)

    env.run_node()
