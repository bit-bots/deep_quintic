from bitbots_msgs.msg import FootPressure
from geometry_msgs.msg import Twist, PoseStamped
from numpy import random
import time
import gym
import pybullet_data
import rospkg
from rosgraph_msgs.msg import Clock

from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32MultiArray
from transforms3d.euler import quat2euler

from deep_quintic.butter_filter import ButterFilter
from deep_quintic.ros_interface import ROSInterface
from deep_quintic.robot import Robot

from deep_quintic.reward import DeepMimicReward, DeepMimicActionReward, CartesianReward, CartesianRelativeReward, \
    CassieReward, DeepMimicActionCartesianReward, CassieActionReward, CassieCartesianReward, \
    CassieCartesianActionReward, CartesianActionReward, EmptyTest, CartesianActionVelReward, CartesianActionOnlyReward, \
    CassieCartesianActionVelReward, JointActionVelReward, SmoothCartesianActionVelReward, \
    CartesianStableActionVelReward, CartesianDoubleActionVelReward, CartesianActionMovementReward, DeepQuinticReward, \
    CartesianStateVelReward, JointStateVelReward
import pybullet as p
import numpy as np
import rospy

from deep_quintic.state import CartesianState, JointSpaceState, PhaseState, BaseState
from deep_quintic.terrain import Terrain
from deep_quintic.trajectory import Trajectory
from stable_baselines3.common.env_checker import check_env

from parallel_parameter_search.utils import load_robot_param, load_yaml_to_param

from deep_quintic.utils import BulletClient, Rot

from bitbots_quintic_walk import PyWalk


class DeepQuinticEnv(gym.Env):
    """This is an OpenAi environment for RL. It extends the simulation to provide the necessary methods for
    compatibility with openai RL algorithms, e.g. PPO.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, pybullet_active=True, reward_function=CartesianActionReward, used_joints="Legs", step_freq=30,
                 ros_debug=False,
                 gui=False, trajectory_file=None, early_termination=True, ep_length_in_s=100, gravity=True,
                 use_engine=True, cartesian_state=False, cartesian_action=False, relative=False,
                 use_state_buffer=False, only_phase_state=False, only_base_state=False, use_foot_sensors=True,
                 cyclic_phase=False, rot_type=Rot.RPY, use_rt_in_state=False, filter_actions=False, vel_in_state=False,
                 terrain_height=0, phase_in_state=True, randomize=False, leg_vel_in_state=True) -> None:
        """
        @param reward_function: a reward object that specifies the reward function
        @param used_joints: which joints should be enabled
        @param step_freq: how many steps are done per second
        @param ros_debug: enables ROS debug messages (needs roscore)
        @param gui: enables pybullet debug GUI
        @param trajectory_file: file containing reference trajectory. without the environment will not use it
        @param early_termination: if episode should be terminated early when robot falls
        """
        self.pybullet_active = pybullet_active
        self.gui = gui
        self.paused = False
        self.realtime = False
        self.time_multiplier = 0
        self.gravity = gravity
        self.ros_debug = ros_debug
        self.early_termination = early_termination
        self.cartesian_state = cartesian_state
        self.cartesian_action = cartesian_action
        self.relative = relative
        self.use_state_buffer = use_state_buffer
        self.only_phase_state = only_phase_state
        self.only_base_state = only_base_state
        self.use_foot_sensors = use_foot_sensors
        self.cyclic_phase = cyclic_phase
        self.rot_type = rot_type
        self.use_rt_in_state = use_rt_in_state
        self.filter_actions = filter_actions
        self.vel_in_state = vel_in_state
        self.terrain_height = terrain_height
        self.phase_in_state = phase_in_state
        self.step_freq = step_freq
        self.randomize = randomize
        self.leg_vel_in_state = leg_vel_in_state
        self.reward_function = reward_function(self)

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

        self.time = 0
        # time step should be at 240Hz (due to pyBullet documentation)
        self.sim_timestep = (1 / 240)
        # How many simulation steps have to be done per policy step
        self.sim_steps = int((1 / self.sim_timestep) / step_freq)
        # length of one step in env
        self.env_timestep = 1 / step_freq
        # ep_length_in_s is in seconds, compute how many steps this is
        self.max_episode_steps = ep_length_in_s * step_freq
        WolfgangBulletEnv.metadata['video.frames_per_second'] = step_freq

        # Instantiating Bullet
        self.pybullet_client = None
        if self.pybullet_active:
            if self.gui:
                self.pybullet_client = BulletClient(p.GUI)
                self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_GUI, True)
                self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                                                              0)
                self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
                self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
                self.debug_alpha_index = self.pybullet_client.addUserDebugParameter("display reference", 0, 1, 0.5)
                self.debug_refbot_fix_position = self.pybullet_client.addUserDebugParameter("fixed refbot position", 0,
                                                                                            1, 0)
                self.debug_random_vel = self.pybullet_client.addUserDebugParameter("random vel", 0, 1, 0)
                self.debug_cmd_vel = [self.pybullet_client.addUserDebugParameter("cmd vel x", -1, 1, 0.1),
                                      self.pybullet_client.addUserDebugParameter("cmd vel y", -1, 1, 0.0),
                                      self.pybullet_client.addUserDebugParameter("cmd vel yaw", -2, 2, 0.0)]
            else:
                self.pybullet_client = BulletClient(p.DIRECT)

            if self.gravity:
                if self.terrain_height > 0:
                    self.terrain = Terrain(self.terrain_height, clear_center=False)
                    self.terrain_index = self.terrain.id
                    self.pybullet_client.changeDynamics(self.terrain_index, -1, lateralFriction=1,
                                                        spinningFriction=0.1, rollingFriction=0.1, restitution=0.9)
                else:
                    # Loading floor
                    self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
                    self.plane_index = self.pybullet_client.loadURDF('plane.urdf')
                    self.pybullet_client.changeDynamics(self.plane_index, -1, lateralFriction=1, spinningFriction=0.1,
                                                        rollingFriction=0.1, restitution=0.9)
                self.pybullet_client.setGravity(0, 0, -9.81)
            else:
                self.pybullet_client.setGravity(0, 0, 0)
            # no real time, as we will publish own clock
            self.pybullet_client.setRealTimeSimulation(0)

        # create real robot + reference robot which is only to display ref trajectory
        compute_feet = (isinstance(self.reward_function, CartesianReward) or (
            isinstance(self.reward_function, CartesianStateVelReward)) or (
                                self.cartesian_state and not self.only_base_state)) and not self.pybullet_client is None
        self.robot = Robot(pybullet_client=self.pybullet_client, compute_joints=True,
                           compute_feet=compute_feet, used_joints=used_joints, physics=True,
                           compute_smooth_vel=isinstance(self.reward_function, SmoothCartesianActionVelReward))
        if self.gui:
            # the reference bot only needs to be connectcompute_joined to pybullet if we want to display it
            self.refbot = Robot(pybullet_client=self.pybullet_client, used_joints=used_joints)
            self.refbot.set_alpha(0.5)
        else:
            self.refbot = Robot(used_joints=used_joints)

        # load trajectory if provided
        self.trajectory = None
        self.engine = None
        self.current_command_speed = [0, 0, 0]
        if trajectory_file is not None:
            self.trajectory = Trajectory()
            self.trajectory.load_from_json(trajectory_file)
        elif use_engine:
            self.namespace = ""
            # load robot model to ROS
            try:
                rospack = rospkg.RosPack()
                load_robot_param(self.namespace, rospack, "wolfgang")
            except ConnectionRefusedError:
                print("### Start roscore! ###")
                exit(1)
            # load walking params
            load_yaml_to_param(self.namespace, "bitbots_quintic_walk", "/config/deep_quintic.yaml", rospack)
            self.engine = PyWalk(self.namespace)
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
        if self.only_phase_state:
            self.state = PhaseState(self)
        elif self.only_base_state:
            self.state = BaseState(self, self.use_foot_sensors, self.randomize)
        else:
            if self.cartesian_state:
                self.state = CartesianState(self, self.use_foot_sensors, self.leg_vel_in_state, self.randomize)
            else:
                self.state = JointSpaceState(self, self.use_foot_sensors, self.leg_vel_in_state, self.randomize)
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
            self.action_publisher = rospy.Publisher("action_debug", Float32MultiArray, queue_size=1)
            self.action_publisher_not_normalized = rospy.Publisher("action_debug_not_normalized", Float32MultiArray,
                                                                   queue_size=1)
            self.state_publisher = rospy.Publisher("state", Float32MultiArray, queue_size=1)
            self.refbot_joint_publisher = rospy.Publisher("ref_joint_states", JointState, queue_size=1)
            self.refbot_left_foot_publisher = rospy.Publisher("ref_left_foot", PoseStamped, queue_size=1)
            self.refbot_right_foot_publisher = rospy.Publisher("ref_right_foot", PoseStamped, queue_size=1)
            self.refbot_pose_publisher = rospy.Publisher("ref_pose", PoseStamped, queue_size=1)
            self.lin_vel_publisher = rospy.Publisher("lin_vel", Float32MultiArray, queue_size=1)
            self.ang_vel_publisher = rospy.Publisher("ang_vel", Float32MultiArray, queue_size=1)
            self.ros_interface = ROSInterface(self.robot, init_node=True)

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
        # joint damping -> would need to reload URDF. aber changeDynamics hat das aus?
        # contact stiffness
        # contact damping

        self.robot.randomize_links(self.domain_rand_bounds["mass"], self.domain_rand_bounds["inertia"])
        self.robot.randomize_joints(self.domain_rand_bounds["motor_torque"], self.domain_rand_bounds["motor_vel"])
        self.robot.randomize_foot_friction(self.domain_rand_bounds["restitution"],
                                           self.domain_rand_bounds["lateral_friction"],
                                           self.domain_rand_bounds["spinning_friction"],
                                           self.domain_rand_bounds["rolling_friction"])

        # set dynamic values for the ground
        rand_restitution = random.uniform(self.domain_rand_bounds["restitution"][0],
                                          self.domain_rand_bounds["restitution"][1])
        rand_lateral_friction = random.uniform(self.domain_rand_bounds["lateral_friction"][0],
                                               self.domain_rand_bounds["lateral_friction"][1])
        rand_spinning_friction = random.uniform(self.domain_rand_bounds["spinning_friction"][0],
                                                self.domain_rand_bounds["spinning_friction"][1])
        rand_rolling_friction = random.uniform(self.domain_rand_bounds["rolling_friction"][0],
                                               self.domain_rand_bounds["rolling_friction"][1])
        if self.terrain_height > 0:
            p.changeDynamics(self.terrain_index, -1,
                             lateralFriction=rand_lateral_friction,
                             spinningFriction=rand_spinning_friction,
                             rollingFriction=rand_rolling_friction,
                             restitution=rand_restitution)
        else:
            p.changeDynamics(self.plane_index, -1,
                             lateralFriction=rand_lateral_friction,
                             spinningFriction=rand_spinning_friction,
                             rollingFriction=rand_rolling_friction,
                             restitution=rand_restitution)

    def reset(self):
        if self.randomize:
            self.randomize_domain()
        if self.terrain_height > 0:
            self.terrain.randomize(self.terrain_height)

        # if we have a reference trajectory we set the simulation to a random start in it
        if self.trajectory:
            # choose randomly a start index from all available frames
            self.trajectory.reset()
            self.current_command_speed = self.trajectory.get_current_command_speed()
            # set robot body accordingly
            self.robot.reset_to_reference(self.refbot)
        elif self.engine:
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

            if self.gui and self.pybullet_client.readUserDebugParameter(self.debug_random_vel) < 0.5:
                # manual setting parameters via gui for testing
                self.current_command_speed[0] = self.pybullet_client.readUserDebugParameter(self.debug_cmd_vel[0])
                self.current_command_speed[1] = self.pybullet_client.readUserDebugParameter(self.debug_cmd_vel[1])
                self.current_command_speed[2] = self.pybullet_client.readUserDebugParameter(self.debug_cmd_vel[2])
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
            # set robot to initial pose
            self.robot.reset_to_reference(self.refbot, self.randomize)
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
        self.time += self.sim_timestep
        self.pybullet_client.stepSimulation()
        if self.use_foot_sensors == "filtered":
            for name, ps in self.robot.pressure_sensors.items():
                ps.filter_step()

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
                left_pressure = self.robot.get_pressure_filtered_left()
                right_pressure = self.robot.get_pressure_filtered_right()
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
        # get keyboard events if gui is active
        if self.gui:
            # check if simulation should continue currently
            while True:
                # rest if R-key was pressed
                rKey = ord('r')
                nKey = ord('n')
                sKey = ord('s')
                tKey = ord('t')
                zKey = ord('z')
                spaceKey = self.pybullet_client.B3G_SPACE
                keys = self.pybullet_client.getKeyboardEvents()
                if rKey in keys and keys[rKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                    self.reset()
                if spaceKey in keys and keys[spaceKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                    self.paused = not self.paused
                if sKey in keys and keys[sKey] & self.pybullet_client.KEY_IS_DOWN:
                    time.sleep(0.1)
                    break
                if nKey in keys and keys[nKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                    if self.gravity:
                        self.pybullet_client.setGravity(0, 0, 0)
                    else:
                        self.pybullet_client.setGravity(0, 0, -9.81)
                    self.gravity = not self.gravity
                if tKey in keys and keys[tKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                    self.realtime = not self.realtime
                    print("Realtime is " + str(self.realtime))
                if zKey in keys and keys[zKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                    self.time_multiplier = (self.time_multiplier + 1) % 3
                    print(self.time_multiplier)

                if not self.paused:
                    break

            # render ref trajectory
            self.refbot.update_ref_in_sim()
            self.refbot.set_alpha(self.pybullet_client.readUserDebugParameter(self.debug_alpha_index))
            if self.pybullet_client.readUserDebugParameter(self.debug_refbot_fix_position) >= 0.5:
                self.refbot.reset_base_to_pose(self.robot.pos_in_world, self.robot.quat_in_world)
            if self.realtime:
                # sleep long enough to run the simulation in real time and not in accelerated speed
                step_computation_time = time.time() - self.last_step_time
                self.last_step_time = time.time()
                time_to_sleep = self.env_timestep - step_computation_time
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep * (self.time_multiplier + 1))

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
            action = np.array([0] * self.num_actions)

        # handle Infs and NaNs
        action_finit = np.isfinite(action).all()

        # save action as class variable since we may need it to compute reward
        self.last_action = action
        if action_finit:
            # filter action
            if self.filter_actions:
                action = self.action_filter.filter(action)
            # apply action and let environment perform a step (which are maybe multiple simulation steps)
            self.action_possible = self.robot.apply_action(action, self.cartesian_action, self.relative, self.refbot,
                                                           self.rot_type)
        if self.randomize:
            # apply some random force
            self.robot.apply_random_force(self.domain_rand_bounds["max_force"], self.domain_rand_bounds["max_torque"])

        for i in range(self.sim_steps):
            self.step_simulation()
        # update the robot model just once after simulation to save performance
        self.robot.update()
        if self.trajectory:
            self.step_trajectory()
        if self.engine:
            # step the refbot and compute the next goals
            self.refbot.step()
            self.refbot_compute_next_step()
        self.step_count += 1
        # compute reward
        if action_finit:
            reward = self.reward_function.compute_current_reward()
        else:
            reward = 0

        dead = self.early_termination and (not self.action_possible or not self.robot.is_alive())
        done = dead or self.step_count >= self.max_episode_steps - 1
        info = dict()
        if done:
            # write custom episode info with key that does not get overwritten by monitor
            info["rewards"] = self.reward_function.get_info_dict()
            info["is_success"] = not dead

        if self.ros_debug:
            self.publish(action)
        self.handle_gui()

        # create state, including previous states and actions
        current_state = self.state.get_state_array(scaled=True)
        self.state_buffer.pop(0)
        self.state_buffer.append(current_state)
        if self.action_buffer_size > 0:
            self.action_buffer.pop(0)
            self.action_buffer.append(self.last_action)

        clock_msg = Clock()
        clock_msg.clock = rospy.Time.from_sec(self.time)
        self.ros_interface.clock_publisher.publish(clock_msg)

        obs = self.create_state_from_buffer()
        # if not np.isfinite(obs).all():
        #    raise AssertionError
        # if not np.isfinite(reward).all():
        #    raise AssertionError
        return obs, reward, bool(done), info

    def get_action_biases(self):
        if self.relative:
            return self.num_actions * [0]
        else:
            # reset the engine to start of step and compute refbot accordingly
            self.engine.special_reset("START_MOVEMENT", 0, cmd_vel_to_twist((0, 0, 0), True), True)
            self.refbot_compute_next_step(0.0001)
            self.refbot.step()
            self.refbot.solve_ik_exactly()
            return self.robot.get_init_mu(self.cartesian_action, self.rot_type, self.refbot)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def publish(self, action):
        self.state.publish_debug()

        if self.action_publisher.get_num_connections() > 0:
            action_msg = Float32MultiArray()
            action_msg.data = action
            self.action_publisher.publish(action_msg)

        if self.action_publisher_not_normalized.get_num_connections() > 0:
            action_msg = Float32MultiArray()
            left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = \
                self.robot.scale_action_to_pose(action, self.rot_type)
            action_msg.data = np.concatenate([left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy])
            self.action_publisher_not_normalized.publish(action_msg)

        if self.state_publisher.get_num_connections() > 0:
            state_msg = Float32MultiArray()
            state_msg.data = self.create_state_from_buffer()
            self.state_publisher.publish(state_msg)

        if self.lin_vel_publisher.get_num_connections() > 0:
            lin_msg = Float32MultiArray()
            lin_msg.data = self.robot.lin_vel
            self.lin_vel_publisher.publish(lin_msg)

        if self.ang_vel_publisher.get_num_connections() > 0:
            ang_msg = Float32MultiArray()
            ang_msg.data = self.robot.ang_vel
            self.ang_vel_publisher.publish(ang_msg)

        self.reward_function.publish_reward()
        if self.refbot_joint_publisher.get_num_connections() > 0:
            self.refbot.solve_ik_exactly()
            self.refbot_joint_publisher.publish(self.refbot.get_joint_position_as_msg())
        if self.refbot_left_foot_publisher.get_num_connections() > 0:
            self.refbot_left_foot_publisher.publish(self.refbot.get_left_foot_msg())
        if self.refbot_right_foot_publisher.get_num_connections() > 0:
            self.refbot_right_foot_publisher.publish(self.refbot.get_right_foot_msg())
        if self.refbot_pose_publisher.get_num_connections() > 0:
            self.refbot_pose_publisher.publish(self.refbot.get_pose_msg())

        self.ros_interface.publish_true_odom()
        if hasattr(self.robot, "joints"):
            self.ros_interface.publish_joints()
        self.ros_interface.publish_foot_pressure()
        self.ros_interface.publish_imu()

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            return
            # raise ValueError('Unsupported render mode:{}'.format(mode))
        view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.robot.pos_in_world,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch, roll=0,
            upAxisIndex=2)
        proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(fov=60, aspect=float(
            self.render_width) / self.render_height, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.pybullet_client.getCameraImage(width=self.render_width, height=self.render_height,
                                                               renderer=self.pybullet_client.ER_BULLET_HARDWARE_OPENGL,
                                                               viewMatrix=view_matrix, projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        pass  # self.pybullet_client.disconnect()


def cmd_vel_to_twist(cmd_vel, stop=False):
    cmd_vel_msg = Twist()
    cmd_vel_msg.linear.x = cmd_vel[0]
    cmd_vel_msg.linear.y = cmd_vel[1]
    cmd_vel_msg.linear.z = 0
    cmd_vel_msg.angular.x = 0
    cmd_vel_msg.angular.y = 0
    cmd_vel_msg.angular.z = cmd_vel[2]
    if stop:
        cmd_vel_msg.angular.x = -1
    return cmd_vel_msg


class WolfgangBulletEnv(DeepQuinticEnv):

    def __init__(self, pybullet_active=True, gui=False, debug=False, trajectory_file=None, early_termination=True,
                 gravity=True,
                 step_freq=30, reward_function=None, ep_length_in_s=20, use_engine=True, cartesian_state=False,
                 cartesian_action=False, relative=False, use_state_buffer=False, only_phase_state=False,
                 only_base_state=False, use_foot_sensors="", cyclic_phase=True, rot_type='rpy',
                 use_rt_in_state=False, filter_actions=False, vel_in_state=False, terrain_height=0,
                 phase_in_state=True, randomize=False, leg_vel_in_state=True):
        reward = {'DeepMimic': DeepMimicReward,
                  'DeepMimicAction': DeepMimicActionReward,
                  'Cartesian': CartesianReward,
                  'CartesianRelative': CartesianRelativeReward,
                  'CartesianAction': CartesianActionReward,
                  'Cassie': CassieReward,
                  'DeepMimicActionCartesian': DeepMimicActionCartesianReward,
                  'CassieAction': CassieActionReward,
                  'CassieCartesian': CassieCartesianReward,
                  'CassieCartesianAction': CassieCartesianActionReward,
                  'CartesianActionVel': CartesianActionVelReward,
                  'EmptyTest': EmptyTest,
                  'CartesianActionOnly': CartesianActionOnlyReward,
                  'CassieCartesianActionVel': CassieCartesianActionVelReward,
                  'JointActionVel': JointActionVelReward,
                  'SmoothCartesianActionVel': SmoothCartesianActionVelReward,
                  'CartesianStableActionVel': CartesianStableActionVelReward,
                  'CartesianDoubleActionVel': CartesianDoubleActionVelReward,
                  'CartesianActionMovement': CartesianActionMovementReward,
                  'DeepQuintic': DeepQuinticReward,
                  'CartesianStateVel': CartesianStateVelReward,
                  'JointStateVelReward': JointStateVelReward,
                  None: EmptyTest,
                  '': EmptyTest}[reward_function]
        rot = {'rpy': Rot.RPY,
               'fused': Rot.FUSED,
               'sixd': Rot.SIXD,
               'quat': Rot.QUAT}[rot_type]

        DeepQuinticEnv.__init__(self, pybullet_active=pybullet_active, reward_function=reward, used_joints="Legs",
                                step_freq=step_freq,
                                ros_debug=debug, gui=gui, trajectory_file=trajectory_file,
                                early_termination=early_termination, ep_length_in_s=ep_length_in_s, gravity=gravity,
                                use_engine=use_engine, cartesian_state=cartesian_state,
                                cartesian_action=cartesian_action, relative=relative, use_state_buffer=use_state_buffer,
                                only_phase_state=only_phase_state, only_base_state=only_base_state,
                                use_foot_sensors=use_foot_sensors, cyclic_phase=cyclic_phase, rot_type=rot,
                                use_rt_in_state=use_rt_in_state, filter_actions=filter_actions,
                                vel_in_state=vel_in_state, terrain_height=terrain_height, phase_in_state=phase_in_state,
                                randomize=randomize, leg_vel_in_state=leg_vel_in_state)
