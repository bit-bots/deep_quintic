import rcl_interfaces.msg
import rclpy
from ament_index_python import get_package_share_directory
from bitbots_moveit_bindings import set_moveit_parameters
from bitbots_moveit_bindings.libbitbots_moveit_bindings import initRos
from rclpy.node import Node

import numpy as np
from bitbots_msgs.msg import FootPressure
from geometry_msgs.msg import Twist
from numpy import random
import gym
from sensor_msgs.msg import Imu

from transforms3d.euler import quat2euler

from deep_quintic.engine import WalkEngine
from stable_baselines3.common.env_checker import check_env

from deep_quintic.butter_filter import ButterFilter
from deep_quintic.ros_debug_interface import ROSDebugInterface
from deep_quintic.robot import Robot
from deep_quintic.reward import DeepMimicReward, DeepMimicActionReward, CartesianReward, CartesianRelativeReward, \
    CassieReward, DeepMimicActionCartesianReward, CassieActionReward, CassieCartesianReward, \
    CassieCartesianActionReward, CartesianActionReward, EmptyTest, CartesianActionVelReward, CartesianActionOnlyReward, \
    CassieCartesianActionVelReward, JointActionVelReward, SmoothCartesianActionVelReward, \
    CartesianStableActionVelReward, CartesianDoubleActionVelReward, CartesianActionMovementReward, DeepQuinticReward, \
    CartesianStateVelReward, JointStateVelReward
from deep_quintic.simulation import WebotsSim, PybulletSim
from deep_quintic.state import CartesianState, JointSpaceState, PhaseState, BaseState
from deep_quintic.trajectory import Trajectory
from deep_quintic.utils import Rot
from bitbots_utils.utils import load_moveit_parameter, get_parameters_from_ros_yaml


class DeepQuinticEnv(gym.Env):
    """This is an OpenAi environment for RL. It extends the simulation to provide the necessary methods for
    compatibility with openai RL algorithms, e.g. PPO.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, simulator_type="pybullet", reward_function="CartesianActionVelReward", used_joints="Legs",
                 step_freq=30, ros_debug=False, gui=False, trajectory_file=None, ep_length_in_s=10, use_engine=True,
                 cartesian_state=True, cartesian_action=True, relative=False, use_state_buffer=False,
                 state_type="full", cyclic_phase=True, rot_type='rpy', filter_actions=False, terrain_height=0,
                 phase_in_state=True, foot_sensors_type="", leg_vel_in_state=False, use_rt_in_state=False,
                 randomize=False, use_complementary_filter=True, random_head_movement=True,
                 adaptive_phase=False, random_force=False, use_gyro=True, use_imu_orientation=True,
                 node: Node = None) -> None:
        """
        @param reward_function: a reward object that specifies the reward function
        @param used_joints: which joints should be enabled
        @param step_freq: how many steps are done per second
        @param ros_debug: enables ROS debug messages (needs roscore)
        @param gui: enables pybullet debug GUI
        @param trajectory_file: file containing reference trajectory. without the environment will not use it
        @param early_termination: if episode should be terminated early when robot falls
        """
        self.node = node
        self.gui = gui
        self.ros_debug = ros_debug
        self.cartesian_state = cartesian_state
        self.cartesian_action = cartesian_action
        self.relative = relative
        self.use_state_buffer = use_state_buffer
        self.state_type = state_type
        self.cyclic_phase = cyclic_phase
        self.use_rt_in_state = use_rt_in_state
        self.foot_sensors_type = foot_sensors_type
        self.filter_actions = filter_actions
        self.terrain_height = terrain_height
        self.phase_in_state = phase_in_state
        self.randomize = randomize
        self.use_complementary_filter = use_complementary_filter
        self.random_head_movement = random_head_movement
        self.adaptive_phase = adaptive_phase
        self.random_force = random_force
        self.leg_vel_in_state = leg_vel_in_state
        self.use_gyro = use_gyro
        self.use_imu_orientation = use_imu_orientation

        self.reward_function = eval(reward_function)(self)
        self.rot_type = {'rpy': Rot.RPY,
                         'fused': Rot.FUSED,
                         'sixd': Rot.SIXD,
                         'quat': Rot.QUAT}[rot_type]

        self.cmd_vel_current_bounds = [(-0.35, 0.35), (-0.2, 0.2), (-1, 1)]
        self.cmd_vel_max_bounds = [(-0.35, 0.35), (-0.2, 0.2), (-1, 1), 0.35]

        self.domain_rand_bounds = {
            # percentages
            "mass": (0.8, 1.2),
            "inertia": (0.5, 1.5),
            "motor_torque": (0.8, 1.2),
            "motor_vel": (0.8, 1.2),
            # absolute values
            "lateral_friction": [0.5, 1.25],  # Lateral friction coefficient (dimensionless)
            "spinning_friction": [0.01, 0.2],  # Spinning friction coefficient (dimensionless)
            "rolling_friction": [0.01, 0.2],  # Rolling friction coefficient (dimensionless)
            "restitution": [0.0, 0.95],  # Bounciness of contacts (dimensionless)
            "max_force": 10,  # N
            "max_torque": 20  # Nm
        }

        self.last_action = None
        self.last_leg_action = None
        self.step_count = 0
        self.start_frame = 0
        self.episode_start_time = 0
        self.pose_on_episode_start = None
        self.last_step_time = 0
        self.action_possible = True

        self.camera_distance = 1.0
        self.camera_yaw = 0
        self.camera_pitch = -30
        self.render_width = 800
        self.render_height = 600

        # Instantiating Simulation
        if "_off" in simulator_type:
            simulator_type = simulator_type[:-4]
            self.sim = None
        else:
            if simulator_type == "webots":
                self.sim = WebotsSim(self.node, self.gui, start_webots=True)
            elif simulator_type == "webots_extern":
                self.sim = WebotsSim(self.node, self.gui)
            elif simulator_type == "webots_fast":
                self.sim = WebotsSim(self.node, self.gui, start_webots=True, fast_physics=True)
            elif simulator_type == "pybullet":
                self.sim = PybulletSim(self.node, self.gui, self.terrain_height)

        self.time = 0
        # How many simulation steps have to be done per policy step
        self.sim_steps = int((1 / self.sim.time_step) / step_freq)
        # since we can only do full sim steps, actual step freq might be different than requested one
        self.step_freq = 1 / (self.sim_steps * self.sim.time_step)
        # length of one step in env
        self.env_timestep = self.sim_steps * self.sim.time_step
        # ep_length_in_s is in seconds, compute how many steps this is
        self.max_episode_steps = ep_length_in_s * step_freq
        DeepQuinticEnv.metadata['video.frames_per_second'] = step_freq

        print(f"sim timestep {self.sim.time_step}")
        print(f"sim_steps {self.sim_steps}")
        print(f"requests env_timestep {1 / step_freq}")
        print(f"actual env_timestep {self.env_timestep}")
        print(f"requests freq {step_freq}")
        print(f"actual freq {self.step_freq}")

        # create real robot + reference robot which is only to display ref trajectory
        compute_feet = (isinstance(self.reward_function, CartesianReward) or (
            isinstance(self.reward_function, CartesianStateVelReward)) or (
                                self.cartesian_state and not self.state_type == "base"))
        # load moveit parameters for IK calls later
        moveit_parameters = load_moveit_parameter("wolfgang")
        initRos()  # need to be initialized for the c++ ros2 node
        set_moveit_parameters(moveit_parameters)
        self.robot = Robot(node, simulation=self.sim, compute_joints=True, compute_feet=compute_feet,
                           used_joints=used_joints, physics=True,
                           compute_smooth_vel=isinstance(self.reward_function, SmoothCartesianActionVelReward),
                           use_complementary_filter=use_complementary_filter)
        if self.gui:
            # the reference bot only needs to be connect to pybullet if we want to display it
            self.refbot = Robot(node, simulation=self.sim, used_joints=used_joints)
            self.refbot.set_alpha(0.5)
        else:
            self.refbot = Robot(node, used_joints=used_joints)

        # load trajectory if provided
        self.trajectory = None
        self.engine = None
        self.current_command_speed = [0.0, 0.0, 0.0]
        self.namespace = ""
        if trajectory_file is not None:
            self.trajectory = Trajectory()
            self.trajectory.load_from_json(trajectory_file)
        elif use_engine:
            # load walking params
            sim_name = simulator_type
            if sim_name in ["webots_extern", "webots_fast"]:
                sim_name = "webots"
            walk_parameters = get_parameters_from_ros_yaml("walking",
                                                           f"{get_package_share_directory('bitbots_quintic_walk')}/config/deep_quintic_{sim_name}.yaml",
                                                           use_wildcard=True)
            self.engine = WalkEngine(self.namespace, walk_parameters + moveit_parameters)
            self.engine_freq = self.engine.get_freq()
        else:
            print("Warning: Neither trajectory nor engine provided")

        if self.cartesian_action:
            # 12 dimensions for left and right foot pose (x,y,z,roll,pitch,yaw)
            if self.rot_type in (Rot.RPY, Rot.FUSED):
                self.num_actions = 12
            elif self.rot_type == Rot.QUAT:
                self.num_actions = 14
            elif self.rot_type == Rot.SIXD:
                self.num_actions = 18
        else:
            # actions are bound between their joint position limits and represented between -1 and 1
            self.num_actions = self.robot.num_used_joints
        if self.adaptive_phase:
            # additional action for timestep
            self.num_actions += 1

        if self.state_type == "phase":
            self.state = PhaseState(self)
        elif self.state_type == "base":
            self.state = BaseState(self, self.foot_sensors_type, self.randomize)
        elif self.state_type == "full":
            if self.cartesian_state:
                self.state = CartesianState(self, self.foot_sensors_type, self.leg_vel_in_state, self.randomize)
            else:
                self.state = JointSpaceState(self, self.foot_sensors_type, self.leg_vel_in_state, self.randomize)
        else:
            print("state type not known")
            exit()
        self.state_buffer = []
        self.action_buffer = []
        if self.use_state_buffer:
            self.state_buffer_size = 5
            self.action_buffer_size = 4
        else:
            self.state_buffer_size = 1
            self.action_buffer_size = 0

        self.action_filter = ButterFilter(action_size=self.num_actions, sampling_rate=1 / self.env_timestep)

        # set some values to allow computing number of observations
        self.robot.left_foot_pos = (0, 0, 0)
        self.robot.right_foot_pos = (0, 0, 0)
        self.robot.left_foot_quat = (1, 0, 0, 0)
        self.robot.right_foot_quat = (1, 0, 0, 0)

        # All actions and observations are bound between to -1 and 1
        self.num_observations = self.state.get_num_observations() * self.state_buffer_size + \
                                self.num_actions * self.action_buffer_size
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_observations,), dtype=np.float32)

        # add publisher if ROS debug is active
        if self.ros_debug:
            self.ros_interface = ROSDebugInterface(self)

        # run check after everything is initialized
        if False and self.ros_debug:
            check_env(self)

    def get_cmd_vel_bounds(self):
        return self.cmd_vel_current_bounds

    def set_cmd_vel_bounds(self, bounds):
        self.cmd_vel_current_bounds = bounds

    def get_reward_weights(self):
        return self.reward_function.weights

    def set_reward_weights(self, weights):
        self.reward_function.weights = weights

    def randomize_domain(self):
        # todo
        # latency -> needs a lot of changes
        # joint friction -> would need to reload URDF
        # joint damping -> would need to reload URDF. or changeDynamics()?
        # contact stiffness
        # contact damping

        self.sim.randomize_links(self.domain_rand_bounds["mass"], self.domain_rand_bounds["inertia"],
                                 self.robot.robot_index)
        self.sim.randomize_joints(self.domain_rand_bounds["motor_torque"], self.domain_rand_bounds["motor_vel"],
                                  self.robot.robot_index)
        self.sim.randomize_foot_friction(self.domain_rand_bounds["restitution"],
                                         self.domain_rand_bounds["lateral_friction"],
                                         self.domain_rand_bounds["spinning_friction"],
                                         self.domain_rand_bounds["rolling_friction"], self.robot.robot_index)

        self.sim.randomize_floor(self.domain_rand_bounds["restitution"],
                                 self.domain_rand_bounds["lateral_friction"],
                                 self.domain_rand_bounds["spinning_friction"],
                                 self.domain_rand_bounds["rolling_friction"])

    def reset(self):
        if self.gui:
            # reset refbot
            self.sim.reset_joints_to_init_pos(self.refbot.robot_index)
        if self.randomize:
            self.randomize_domain()
        if self.terrain_height > 0:
            self.sim.randomize_terrain(self.terrain_height)

        # if we have a reference trajectory we set the simulation to a random start in it
        if self.trajectory:
            # choose randomly a start index from all available frames
            self.trajectory.reset()
            self.current_command_speed = self.trajectory.get_current_command_speed()
            # set robot body accordingly
            self.robot.reset_to_reference(self.refbot, self.randomize)
        elif self.engine is not None:
            # choose random initial state of the engine
            self.current_command_speed = [random.uniform(*self.cmd_vel_current_bounds[0]),
                                          random.uniform(*self.cmd_vel_current_bounds[1]),
                                          random.uniform(*self.cmd_vel_current_bounds[2])]
            # make sure that the combination of x and y speed is not too low or too high
            if abs(self.current_command_speed[0]) + abs(self.current_command_speed[1]) > self.cmd_vel_max_bounds[3]:
                # decrease one of the two
                direction = 1  # random.randint(0, 2)
                sign = 1 if self.current_command_speed[direction] > 0 else -1
                self.current_command_speed[direction] = sign * (abs(self.cmd_vel_max_bounds[3]) - abs(
                    self.current_command_speed[(direction + 1) % 2]))

            # set command vel based on GUI input if appropriate
            if self.gui:
                gui_cmd_vel = self.sim.read_command_vel_from_gui()
                if gui_cmd_vel is not None:
                    self.current_command_speed = gui_cmd_vel
            engine_state = "WALKING"  # IDLE, WALKING, START_STEP, STOP_STEP, START_MOVEMENT, STOP_MOVEMENT, PAUSED, KICK
            phase = random.uniform()
            # reset the engine to specific start values
            self.engine.special_reset(engine_state, phase, cmd_vel_to_twist(self.current_command_speed), True)
            # compute 3 times to set previous, current and next frame
            # previous
            self.refbot_compute_next_step(reset=True)
            self.refbot.step()
            self.refbot.solve_ik_exactly()
            # correctly set time between frames
            self.time += self.env_timestep

            # current
            self.refbot_compute_next_step(reset=True)
            self.refbot.step()
            self.refbot.solve_ik_exactly()
            # next
            self.refbot_compute_next_step(reset=True)
            # hacky since terrain >1 represents different type of terrain
            if self.terrain_height > 1:
                reset_terrain_height = self.terrain_height - 1
            else:
                reset_terrain_height = self.terrain_height
            # set robot to initial pose
            self.robot.reset_to_reference(self.refbot, self.randomize, reset_terrain_height + 0.02)
        else:
            # without trajectory we just go to init
            self.robot.reset()
        self.step_count = 0
        self.episode_start_time = self.time
        self.reward_function.reset_episode_reward()
        # return robot state and current phase value
        # since we did a reset, we will fill the buffer with the same exact states and 0 actions
        current_state = self.state.get_state_array(scaled=True)
        self.state_buffer = [current_state] * self.state_buffer_size
        self.action_buffer = [[0] * self.num_actions] * self.action_buffer_size
        if self.filter_actions:
            self.action_filter.reset()
            if self.cartesian_action and not self.relative:
                self.action_filter.init_history(self.robot.scale_pose_to_action(self.refbot.left_foot_pos,
                                                                                quat2euler(self.refbot.left_foot_quat),
                                                                                self.refbot.right_foot_pos,
                                                                                quat2euler(self.refbot.right_foot_quat),
                                                                                self.rot_type))
            else:
                print("not implemented")
                exit()
        if self.gui:
            self.refbot.update_ref_in_sim()
        return self.create_state_from_buffer()

    def create_state_from_buffer(self):
        if self.state_buffer_size == 1 and self.action_buffer_size == 0:
            return self.state_buffer[0]
        else:
            return [*self.state_buffer[0], *self.state_buffer[1], *self.state_buffer[2], *self.state_buffer[3],
                    *self.state_buffer[4], *self.action_buffer[0], *self.action_buffer[1], *self.action_buffer[2],
                    *self.action_buffer[3]]

    def step_simulation(self):
        self.time += self.sim.time_step
        self.sim.step()
        # filters need to be done each step to have a high frequency
        if self.foot_sensors_type == "filtered":
            self.robot.step_pressure_filters()
        if self.use_complementary_filter:
            self.robot.update_complementary_filter(self.sim.time_step)

    def step_trajectory(self):
        # step the trajectory further, based on the time
        # To get correctly matching frame, as just stepping through it would not conserve time
        # function will automatically loop the trajectory
        self.trajectory.step_to_time(self.time - self.episode_start_time)

    def refbot_compute_next_step(self, timestep=None, reset=False):
        if not timestep:
            timestep = self.env_timestep
        # step the reference robot based on the engine
        if self.cartesian_state:
            result = self.engine.step_open_loop(timestep, cmd_vel_to_twist(self.current_command_speed))
        else:
            if reset:
                imu_msg = Imu()
                imu_msg.orientation.w = 1
                left_pressure = FootPressure()
                right_pressure = FootPressure()
            else:
                imu_msg = self.robot.get_imu_msg()
                left_pressure = self.robot.get_pressure(left=True, filtered=True, time=False)
                right_pressure = self.robot.get_pressure(left=False, filtered=True, time=False)
            result = self.engine.step(timestep, cmd_vel_to_twist(self.current_command_speed), imu_msg,
                                      self.robot.get_joint_state_msg(), left_pressure, right_pressure)
        phase = self.engine.get_phase()
        odom_msg = self.engine.get_odom()
        position = np.array(
            [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        orientation = np.array(
            [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
             odom_msg.pose.pose.orientation.z,
             ])
        if self.cartesian_state:
            left_foot_pos = np.array(
                [result.poses[0].position.x, result.poses[0].position.y, result.poses[0].position.z])
            left_foot_quat = np.array(
                [result.poses[0].orientation.w, result.poses[0].orientation.x, result.poses[0].orientation.y,
                 result.poses[0].orientation.z])
            right_foot_pos = np.array(
                [result.poses[1].position.x, result.poses[1].position.y, result.poses[1].position.z])
            right_foot_quat = np.array(
                [result.poses[1].orientation.w, result.poses[1].orientation.x, result.poses[1].orientation.y,
                 result.poses[1].orientation.z])
            self.refbot.set_next_step(time=self.time, phase=phase, position=position, orientation=orientation,
                                      robot_lin_vel=self.current_command_speed[:2] + [0],
                                      robot_ang_vel=[0, 0] + self.current_command_speed[2:2],
                                      left_foot_pos=left_foot_pos, left_foot_quat=left_foot_quat,
                                      right_foot_pos=right_foot_pos,
                                      right_foot_quat=right_foot_quat)
            if isinstance(self.reward_function, DeepMimicActionReward):
                # solve IK for reward
                self.refbot.solve_ik_exactly()
        else:
            # sort joint positions in correct order
            joint_positions = np.empty(self.robot.num_used_joints)
            for i in range(self.robot.num_used_joints):
                index = self.robot.joint_indexes[result.joint_names[i]]
                joint_positions[index] = result.positions[i]
            self.refbot.set_next_step(time=self.time, phase=phase, position=position, orientation=orientation,
                                      robot_lin_vel=self.current_command_speed[:2] + [0],
                                      robot_ang_vel=[0, 0] + self.current_command_speed[2:2],
                                      joint_positions=joint_positions)

    def handle_gui(self):
        if self.gui:
            # render ref trajectory
            self.refbot.update_ref_in_sim()
            self.refbot.set_alpha(self.sim.get_alpha())
            if self.sim.is_fixed_position():
                self.refbot.reset_base_to_pose(self.robot.pos_in_world, self.robot.quat_in_world)
            self.sim.handle_gui()

    def step(self, action):
        if False:
            # test reference as action
            if self.cartesian_action:
                action = self.robot.scale_pose_to_action(self.refbot.left_foot_pos,
                                                         quat2euler(self.refbot.left_foot_quat),
                                                         self.refbot.right_foot_pos,
                                                         quat2euler(self.refbot.right_foot_quat),
                                                         self.rot_type)
            else:
                action = []
                i = 0
                for joint_name in self.robot.used_joint_names:
                    joint = self.robot.joints[joint_name]
                    ref_position = self.refbot.joint_positions[i]
                    scaled_position = joint.convert_radiant_to_scaled(ref_position)
                    action.append(scaled_position)
                    i += 1
            action = np.array(action)
        if False:
            action = self.action_space.sample()
        if False:
            action = np.array([-1] * self.num_actions)

        # handle Infs and NaNs
        action_finit = np.isfinite(action).all()

        # save action as class variable since we may need it to compute reward
        self.last_action = action
        if self.adaptive_phase:
            self.last_leg_action = action[:-1]
        else:
            self.last_leg_action = action

        if action_finit:
            # filter action
            if self.filter_actions:
                action = self.action_filter.filter(action)
            # apply action and let environment perform a step (which are maybe multiple simulation steps)
            self.action_possible = self.robot.apply_action(action, self.cartesian_action, self.relative, self.refbot,
                                                           self.rot_type)
        if self.random_force:
            # apply some random force
            self.robot.apply_random_force(self.domain_rand_bounds["max_force"], self.domain_rand_bounds["max_torque"])
        if self.random_head_movement:
            self.robot.set_random_head_goals()

        for i in range(self.sim_steps):
            self.step_simulation()
        # update the robot model just once after simulation to save performance
        self.robot.update()
        if self.trajectory:
            self.step_trajectory()
        if self.engine is not None:
            # step the refbot and compute the next goals
            self.refbot.step()
            if self.adaptive_phase:
                # compute timestep from networks action. scaled to [0, single_step_time]
                time_of_single_step = 0.5 / self.engine_freq
                timestep = (action[-1] + 1) / 2 * time_of_single_step
                if timestep == 0:
                    # walk engine will not accept exactly zero. hacky way to avoid this
                    timestep = 0.000000001
            else:
                # use fixed timestep
                timestep = None
            self.refbot_compute_next_step(timestep=timestep)
        self.step_count += 1
        # compute reward
        if action_finit:
            reward = self.reward_function.compute_current_reward()
        else:
            reward = 0

        if self.ros_debug:
            self.ros_interface.publish()
        self.handle_gui()

        # create state, including previous states and actions
        current_state = self.state.get_state_array(scaled=True)
        self.state_buffer.pop(0)
        self.state_buffer.append(current_state)
        if self.action_buffer_size > 0:
            self.action_buffer.pop(0)
            self.action_buffer.append(self.last_action)

        obs = self.create_state_from_buffer()
        # if not np.isfinite(obs).all():
        #    raise AssertionError
        # if not np.isfinite(reward).all():
        #    raise AssertionError

        dead = not self.action_possible or not self.robot.is_alive()
        done = dead or self.step_count >= self.max_episode_steps - 1
        info = dict()
        if done:
            # write custom episode info with key that does not get overwritten by monitor
            info["rewards"] = self.reward_function.get_info_dict()
            info["is_success"] = not dead
            if not dead:
                # write some information that shows stable-baselines that this was ended due to a timelimit
                # so that the bootstrapping is correctly performed
                info["terminal_observation"] = obs
                info["TimeLimit.truncated"] = True
        return obs, reward, bool(done), info

    def get_action_biases(self):
        if self.relative:
            return self.num_actions * [0]
        else:
            # reset the engine to start of step and compute refbot accordingly
            self.engine.special_reset("START_MOVEMENT", 0.0, cmd_vel_to_twist((0.0, 0.0, 0.0), True), True)
            self.refbot_compute_next_step(0.0001)
            self.refbot.step()
            self.refbot.solve_ik_exactly()
            action_mu = self.robot.get_init_mu(self.cartesian_action, self.rot_type, self.refbot)
            if self.adaptive_phase:
                # just use the normal fixed timestep as initial value for the timestep but scale it to [-1, 1]
                time_of_single_step = 0.5 / self.engine_freq
                action_mu.append((self.env_timestep / time_of_single_step) * 2 - 1)
            return action_mu

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            return
            # raise ValueError('Unsupported render mode:{}'.format(mode))

        return self.sim.get_render(self.render_width, self.render_height, self.camera_distance, self.camera_pitch,
                                   self.camera_yaw, self.robot.pos_in_world)

    def close(self):
        pass  # self.pybullet_client.disconnect()


def cmd_vel_to_twist(cmd_vel, stop=False):
    cmd_vel_msg = Twist()
    cmd_vel_msg.linear.x = float(cmd_vel[0])
    cmd_vel_msg.linear.y = float(cmd_vel[1])
    cmd_vel_msg.linear.z = 0.0
    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = float(cmd_vel[2])
    if stop:
        cmd_vel_msg.angular.x = -1.0
    return cmd_vel_msg


class WolfgangWalkEnv(DeepQuinticEnv):

    def __init__(self, simulator_type="pybullet", reward_function="CartesianActionVelReward", step_freq=30,
                 ros_debug=False,
                 gui=False, trajectory_file=None, ep_length_in_s=10, use_engine=True,
                 cartesian_state=True, cartesian_action=True, relative=False, use_state_buffer=False,
                 state_type="full", cyclic_phase=True, rot_type="rpy", filter_actions=False, terrain_height=0,
                 phase_in_state=True, foot_sensors_type="", leg_vel_in_state=False, use_rt_in_state=False,
                 randomize=False, use_complementary_filter=True, random_head_movement=True, adaptive_phase=False,
                 random_force=False, use_gyro=True, use_imu_orientation=True, node=None):
        if node is None:
            rclpy.init()
            node_name = 'walking_env'
            node = Node(node_name)

        DeepQuinticEnv.__init__(self, simulator_type=simulator_type, reward_function=reward_function,
                                used_joints="Legs", step_freq=step_freq, ros_debug=ros_debug, gui=gui,
                                trajectory_file=trajectory_file, state_type=state_type, ep_length_in_s=ep_length_in_s,
                                use_engine=use_engine, cartesian_state=cartesian_state,
                                cartesian_action=cartesian_action, relative=relative,
                                use_state_buffer=use_state_buffer, cyclic_phase=cyclic_phase, rot_type=rot_type,
                                use_rt_in_state=use_rt_in_state, filter_actions=filter_actions,
                                terrain_height=terrain_height, foot_sensors_type=foot_sensors_type,
                                phase_in_state=phase_in_state, randomize=randomize, leg_vel_in_state=leg_vel_in_state,
                                use_complementary_filter=use_complementary_filter,
                                random_head_movement=random_head_movement,
                                adaptive_phase=adaptive_phase, random_force=random_force, use_gyro=use_gyro,
                                use_imu_orientation=use_imu_orientation, node=node)
