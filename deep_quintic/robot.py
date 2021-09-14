import random

import rospkg
import math
import pybullet as p
import rospy
from bitbots_msgs.msg import FootPressure
from moveit_msgs.srv import GetPositionIKRequest, GetPositionFKRequest
from optuna import TrialPruned
from scipy import signal
import numpy as np
from sensor_msgs.msg import JointState, Imu
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3
from transforms3d.affines import compose, decompose
from transforms3d.euler import mat2euler, quat2euler, euler2mat, euler2quat

from deep_quintic.utils import Rot, fused2quat, sixd2quat, quat2fused, quat2sixd, compute_ik
from deep_quintic.state import CartesianState, State, JointSpaceState
from deep_quintic.utils import compute_imu_orientation_from_world, wxyz2xyzw, xyzw2wxyz
from transforms3d.quaternions import quat2mat, rotate_vector, qinverse, qmult, mat2quat
from bitbots_moveit_bindings import get_position_ik, get_position_fk


class Robot:
    def __init__(self, pybullet_client=None, compute_joints=False, compute_feet=False, used_joints="Legs",
                 physics=False, env_time_step=None, compute_smooth_vel=False):
        self.physics_active = physics
        self.pybullet_client = pybullet_client
        self.compute_joints = compute_joints
        self.compute_feet = compute_feet
        self.compute_smooth_vel = compute_smooth_vel

        self.pos_on_episode_start = [0, 0, 0.43]
        self.quat_on_episode_start = euler2quat(0, 0.25, 0)
        # we always keep 3 steps of the reference, the next the current and the previous
        self.next_time = None
        self.time = None
        self.previous_time = None
        self.next_phase = None
        self.phase = 0
        self.next_pos_in_world = None
        self.pos_in_world = None
        self.previous_pos_in_world = None
        self.next_quat_in_world = None
        self.quat_in_world = None
        self.previous_quat_in_world = None
        self.imu_rpy = [0, 0]
        self.walk_vel = None
        self.last_walk_vels = []
        self.smooth_vel = None
        self.next_lin_vel = None
        self.lin_vel = None
        self.last_lin_vel = None
        self.next_ang_vel = None
        self.ang_vel = [0, 0, 0]
        self.lin_acc = [0, 0, 0]
        self.next_joint_positions = None
        self.joint_positions = None
        self.previous_joint_positions = None
        self.joint_velocities = None
        self.joint_torques = None
        self.next_left_foot_pos = None
        self.left_foot_pos = None
        self.previous_left_foot_pos = None
        self.next_left_foot_quat = None
        self.left_foot_quat = None
        self.previous_left_foot_quat = None
        self.next_right_foot_pos = None
        self.right_foot_pos = None
        self.previous_right_foot_pos = None
        self.next_right_foot_quat = None
        self.right_foot_quat = None
        self.previous_right_foot_quat = None
        self.left_foot_lin_vel = [0, 0, 0]
        self.left_foot_ang_vel = [0, 0, 0]
        self.right_foot_lin_vel = [0, 0, 0]
        self.right_foot_ang_vel = [0, 0, 0]
        self.alpha = 1

        # config values
        self.initial_joints_positions = {"LAnklePitch": -30, "LAnkleRoll": 0, "LHipPitch": 30, "LHipRoll": 0,
                                         "LHipYaw": 0, "LKnee": 60, "RAnklePitch": 30, "RAnkleRoll": 0,
                                         "RHipPitch": -30, "RHipRoll": 0, "RHipYaw": 0, "RKnee": -60,
                                         "LShoulderPitch": 75, "LShoulderRoll": 0, "LElbow": 36, "RShoulderPitch": -75,
                                         "RShoulderRoll": 0, "RElbow": -36, "HeadPan": 0, "HeadTilt": 0}
        # how the foot pos and rpy are scaled. from [-1:1] action to meaningful m or rad
        # foot goal is relative to the start of the leg, so that 0 would be the center of possible poses
        # x, y, z, roll, pitch, yaw
        self.cartesian_limits_left = [(-0.15, 0.27), (0, 0.25), (-0.44, -0.24), (-math.tau / 12, math.tau / 12),
                                      (-math.tau / 12, math.tau / 12), (-math.tau / 12, math.tau / 12)]
        # right is same just y pos is inverted
        self.cartesian_limits_right = self.cartesian_limits_left.copy()
        self.cartesian_limits_right[1] = (-self.cartesian_limits_left[1][1], -self.cartesian_limits_left[1][0])
        # compute mid_positions of limits just once now, to use them later
        self.cartesian_mid_positions_left = []
        self.cartesian_mid_positions_right = []
        for i in range(6):
            self.cartesian_mid_positions_left.append(
                0.5 * (self.cartesian_limits_left[i][0] + self.cartesian_limits_left[i][1]))
            self.cartesian_mid_positions_right.append(
                0.5 * (self.cartesian_limits_right[i][0] + self.cartesian_limits_right[i][1]))

        self.relative_scaling_joint_action = 0.1
        self.relative_scaling_cartesian_action_pos = 0.05
        self.relative_scaling_cartesian_action_ori = math.tau / 24

        self.leg_joints = ["LAnklePitch", "LAnkleRoll", "LHipPitch", "LHipRoll", "LHipYaw", "LKnee",
                           "RAnklePitch", "RAnkleRoll", "RHipPitch", "RHipRoll", "RHipYaw", "RKnee"]
        if used_joints == "Legs":
            self.used_joint_names = self.leg_joints
        elif used_joints == "LegsAndArms":
            self.used_joint_names = self.leg_joints + ["LShoulderPitch", "LShoulderRoll", "LElbow", "RShoulderPitch",
                                                       "RShoulderRoll", "RElbow"]
        elif used_joints == "All":
            self.used_joint_names = self.leg_joints + ["LShoulderPitch", "LShoulderRoll", "LElbow", "RShoulderPitch",
                                                       "RShoulderRoll", "RElbow", "HeadPan", "HeadTilt"]
        else:
            print("Used joint group \"{}\" not known".format(used_joints))
        self.num_used_joints = len(self.used_joint_names)
        # precompute mapping of name to index for faster usage later
        self.joint_indexes = {}
        i = 0
        for joint_name in self.used_joint_names:
            self.joint_indexes[joint_name] = i
            i += 1

        # messages for use with python engine interface
        self.joint_state_msg = JointState()
        self.joint_state_msg.header.frame_id = "base_link"
        self.joint_state_msg.name = self.initial_joints_positions.keys()
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "imu_frame"
        self.pressure_msg_left = FootPressure()
        self.pressure_msg_left.header.frame_id = 'l_sole'
        self.pressure_msg_right = FootPressure()
        self.pressure_msg_right.header.frame_id = 'r_sole'
        self.left_foot_msg = PoseStamped()
        self.left_foot_msg.header.frame_id = 'base_link'
        self.right_foot_msg = PoseStamped()
        self.right_foot_msg.header.frame_id = 'base_link'

        self.pressure_sensors = {}
        if self.pybullet_client is not None:
            self.robot_index = None
            self.joints = {}
            self.links = {}
            self.torso_id = {}
            self.link_masses = []
            self.link_inertias = []
            self.joint_stall_torques = []
            self.joint_max_vels = []
            self.init_robot_in_sim(physics)

    def init_robot_in_sim(self, physics):
        # Loading robot in simulation
        if physics:
            flags = self.pybullet_client.URDF_USE_SELF_COLLISION + \
                    self.pybullet_client.URDF_USE_INERTIA_FROM_FILE + \
                    self.pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        else:
            flags = self.pybullet_client.URDF_USE_INERTIA_FROM_FILE

        rospack = rospkg.RosPack()
        urdf_file = rospack.get_path("wolfgang_description") + "/urdf/robot.urdf"
        self.robot_index = self.pybullet_client.loadURDF(urdf_file,
                                                         self.pos_on_episode_start,
                                                         self.quat_on_episode_start,
                                                         flags=flags, useFixedBase=not physics)
        if not physics:
            self._disable_physics()

        # Retrieving joints and foot pressure sensors
        for i in range(self.pybullet_client.getNumJoints(self.robot_index)):
            joint_info = self.pybullet_client.getJointInfo(self.robot_index, i)
            name = joint_info[1].decode('utf-8')
            # we can get the links by seeing where the joint is attached. we only get parent link so do +1
            self.links[joint_info[12].decode('utf-8')] = joint_info[16] + 1
            if name in self.initial_joints_positions.keys():
                # remember joint
                self.joints[name] = Joint(i, self.robot_index, self.pybullet_client)
                self.joint_stall_torques.append(joint_info[10])
                self.joint_max_vels.append(joint_info[11])
            elif name in ["LLB", "LLF", "LRF", "LRB", "RLB", "RLF", "RRF", "RRB"]:
                self.pybullet_client.enableJointForceTorqueSensor(self.robot_index, i)
                self.pressure_sensors[name] = PressureSensor(name, i, self.robot_index, 10, 5, self.pybullet_client)
        self.torso_id = self.links["torso"]

        for link_id in self.links.values():
            dynamics_info = self.pybullet_client.getDynamicsInfo(self.robot_index, link_id)
            self.link_masses.append(dynamics_info[0])
            self.link_inertias.append(dynamics_info[2])

        # set dynamics for feet, maybe overwritten later by domain randomization
        for link_name in self.links.keys():
            if link_name in ["llb", "llf", "lrf", "lrb", "rlb", "rlf", "rrf", "rrb"]:
                # print(p.getLinkState(self.robot_index, self.links[link_name]))
                p.changeDynamics(self.robot_index, self.links[link_name], lateralFriction=1, spinningFriction=0.1,
                                 rollingFriction=0.1, restitution=0.9)

        # robot to initial position
        self.reset()

    def randomize_links(self, mass_bounds, inertia_bounds):
        i = 0
        for link_id in self.links.values():
            randomized_mass = random.uniform(mass_bounds[0], mass_bounds[1]) * self.link_masses[i]
            randomized_inertia = random.uniform(inertia_bounds[0], inertia_bounds[1]) * np.array(self.link_inertias[i])
            self.pybullet_client.changeDynamics(self.robot_index, link_id, mass=randomized_mass,
                                                localInertiaDiagonal=randomized_inertia)
            i += 1

    def randomize_joints(self, torque_bounds, vel_bounds):
        i = 0
        for joint_name in self.joints.keys():
            randomized_stall_torque = random.uniform(torque_bounds[0], torque_bounds[1]) * self.joint_stall_torques[i]
            self.joints[joint_name].max_torque = randomized_stall_torque
            randomized_max_vel = random.uniform(vel_bounds[0], vel_bounds[1]) * self.joint_max_vels[i]
            self.joints[joint_name].max_vel = randomized_max_vel

    def randomize_foot_friction(self, restitution_bounds, lateral_friction_bounds, spinning_friction_bounds,
                                rolling_friction_bounds):
        # set dynamic values for all foot links
        rand_restitution = random.uniform(restitution_bounds[0], restitution_bounds[1])
        rand_lateral_friction = random.uniform(lateral_friction_bounds[0], lateral_friction_bounds[1])
        rand_spinning_friction = random.uniform(spinning_friction_bounds[0], spinning_friction_bounds[1])
        rand_rolling_friction = random.uniform(rolling_friction_bounds[0], rolling_friction_bounds[1])

        for link_name in self.links.keys():
            if link_name in ["llb", "llf", "lrf", "lrb", "rlb", "rlf", "rrf", "rrb"]:
                p.changeDynamics(self.robot_index, self.links[link_name],
                                 lateralFriction=rand_lateral_friction,
                                 spinningFriction=rand_spinning_friction,
                                 rollingFriction=rand_rolling_friction,
                                 restitution=rand_restitution)

    def transform_world_to_robot(self, pos_in_world, quat_in_world, lin_vel_in_world,
                                 ang_vel_in_world):
        # transform from world to relative to base link
        robot_mat_in_world = quat2mat(self.quat_in_world)
        # normally we would just do this, but this takes a lot of CPU. so we do it manually
        # robot_trans_in_world = compose(self.pos_in_world, robot_mat_in_world, [1, 1, 1])
        # world_trans_in_robot = np.linalg.inv(robot_trans_in_world)
        inv_rot = robot_mat_in_world.T
        inv_trans = np.matmul(-inv_rot, self.pos_in_world)
        world_trans_in_robot = compose(inv_trans, inv_rot, [1, 1, 1])

        mat_in_world = quat2mat(quat_in_world)
        trans_in_world = compose(pos_in_world, mat_in_world, [1, 1, 1])
        trans_in_robot = np.matmul(world_trans_in_robot, trans_in_world)

        # normally we would just do the following, but np.linalg.inv takes a lot of cpu
        # pos_in_robot, mat_in_robot, _, _ = decompose(trans_in_robot)
        pos_in_robot = trans_in_robot[:-1, -1]
        RZS = trans_in_robot[:-1, :-1]
        # we know that we dont have any sheer or zoom
        ZS = np.diag([1, 1, 1])
        # since ZS is a unit matrix, we dont need to build the inverse
        mat_in_robot = np.dot(RZS, ZS)

        quat_in_robot = mat2quat(mat_in_robot)

        # rotate velocities so that they are in robot frame
        lin_vel_in_robot = rotate_vector(lin_vel_in_world, qinverse(self.quat_in_world))
        # subtract linear velocity of robot, since it should be relative to it
        rel_lin_vel = lin_vel_in_robot - self.lin_vel
        # same for angular vels
        ang_vel_in_robot = rotate_vector(ang_vel_in_world, qinverse(self.quat_in_world))
        rel_ang_vel = ang_vel_in_robot - self.ang_vel

        return pos_in_robot, quat_in_robot, rel_lin_vel, rel_ang_vel

    def update(self):
        """
        Updates the state of the robot from pybullet. This is only done once per step to improve performance.
        """
        self.previous_pos_in_world = self.pos_in_world
        self.previous_quat_in_world = self.quat_in_world
        (x, y, z), (qx, qy, qz, qw) = self.pybullet_client.getBasePositionAndOrientation(self.robot_index)
        self.pos_in_world = np.array([x, y, z])
        self.quat_in_world = xyzw2wxyz([qx, qy, qz, qw])

        # imu orientation has roll and pitch relative to gravity vector. yaw in world frame
        self.imu_rpy, yaw_quat = compute_imu_orientation_from_world(self.quat_in_world)
        # rotate velocities from world to robot frame
        self.last_lin_vel = self.lin_vel
        lin_vel_in_world, ang_vel_in_world = self.pybullet_client.getBaseVelocity(self.robot_index)
        walk_lin_vel = rotate_vector(lin_vel_in_world, qinverse(yaw_quat))
        walk_ang_vel = rotate_vector(ang_vel_in_world, qinverse(yaw_quat))
        self.walk_vel = np.concatenate([walk_lin_vel[:2], walk_ang_vel[2:3]])
        if self.compute_smooth_vel:
            if len(self.last_walk_vels) > 60:
                self.last_walk_vels.pop(0)
            self.last_walk_vels.append(self.walk_vel)
            self.smooth_vel = np.array(self.last_walk_vels).mean(axis=0)
        self.lin_vel = rotate_vector(lin_vel_in_world, qinverse(self.quat_in_world))
        self.ang_vel = rotate_vector(ang_vel_in_world, qinverse(self.quat_in_world))
        if self.last_lin_vel is None:
            # handle issues on start of episode
            self.last_lin_vel = self.lin_vel

        # simple acceleration computation by using diff of velocities
        linear_acc = np.array(list(map(lambda i, j: i - j, self.last_lin_vel, self.lin_vel)))
        # adding gravity to the acceleration, following REP-145
        gravity_world_frame = np.array([0, 0, 9.81])
        gravity_robot_frame = rotate_vector(gravity_world_frame, qinverse(self.quat_in_world))
        self.lin_acc = linear_acc + gravity_robot_frame

        # compute joints
        if self.compute_joints:
            self.previous_joint_positions = self.joint_positions
            self.joint_positions = []
            self.joint_velocities = []
            self.joint_torques = []
            for joint_name in self.used_joint_names:
                joint = self.joints[joint_name]
                pos, vel, tor = joint.update()
                self.joint_positions.append(pos)
                self.joint_velocities.append(vel)
                self.joint_torques.append(tor)

        # foot poses
        if self.compute_feet:
            # left foot
            _, _, _, _, foot_pos_in_world, foot_xyzw_in_world, lin_vel_in_world, ang_vel_in_world = p.getLinkState(
                self.robot_index, self.links["l_sole"], 1, 0)
            foot_quat_in_world = xyzw2wxyz(foot_xyzw_in_world)
            self.left_foot_pos, self.left_foot_quat, self.left_foot_lin_vel, self.left_foot_ang_vel = self.transform_world_to_robot(
                foot_pos_in_world, foot_quat_in_world, lin_vel_in_world, ang_vel_in_world)

            # right foot
            _, _, _, _, foot_pos_in_world, foot_xyzw_in_world, lin_vel_in_world, ang_vel_in_world = p.getLinkState(
                self.robot_index, self.links["r_sole"], 1, 0)
            # PyBullet does weird stuff and rotates our feet
            foot_quat_in_world = xyzw2wxyz(foot_xyzw_in_world)
            self.right_foot_pos, self.right_foot_quat, self.right_foot_lin_vel, self.right_foot_ang_vel = self.transform_world_to_robot(
                foot_pos_in_world, foot_quat_in_world, lin_vel_in_world, ang_vel_in_world)

    def apply_action(self, action, cartesian, relative, refbot, rot_type):
        if not cartesian:
            i = 0
            # iterate through all joints, always in same order to keep the same matching between NN and simulation
            for joint_name in self.used_joint_names:
                # scaling needed since action space is -1 to 1, but joints have lower and upper limits
                joint = self.joints[joint_name]
                if relative:
                    goal_position = joint.get_position() + action[
                        i] * self.relative_scaling_joint_action  # todo we can get out of bounds by this
                    joint.set_position(goal_position)
                else:
                    joint.set_scaled_position(action[i])
                i += 1
            return True
        else:
            # split action values into corresponding position and rpy
            if relative:
                left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = self.scale_action_to_relative_pose(
                    action)
            else:
                left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = self.scale_action_to_pose(action,
                                                                                                         rot_type)
            left_foot_quat = euler2quat(*left_foot_rpy)
            right_foot_quat = euler2quat(*right_foot_rpy)
            ik_result, success = compute_ik(left_foot_pos, left_foot_quat, right_foot_pos, right_foot_quat,
                                            self.used_joint_names, self.joint_indexes, collision=False,
                                            approximate=True)
            if success:
                for i in range(len(self.leg_joints)):
                    joint = self.joints[self.leg_joints[i]]
                    joint.set_position(ik_result[i])
                return True
            else:
                return False

    def set_next_step(self, time, phase, position, orientation, robot_lin_vel, robot_ang_vel, left_foot_pos=None,
                      left_foot_quat=None, right_foot_pos=None, right_foot_quat=None, joint_positions=None):
        self.next_time = time
        self.next_phase = phase
        self.next_pos_in_world = position
        self.next_quat_in_world = orientation
        self.next_lin_vel = robot_lin_vel
        self.next_ang_vel = robot_ang_vel
        self.next_left_foot_pos = left_foot_pos
        self.next_left_foot_quat = left_foot_quat
        self.next_right_foot_pos = right_foot_pos
        self.next_right_foot_quat = right_foot_quat
        self.next_joint_positions = joint_positions

    def step(self):
        # step from given reference trajectory
        self.previous_time = self.time
        self.time = self.next_time
        self.next_time = None
        self.phase = self.next_phase
        self.previous_pos_in_world = self.pos_in_world
        self.pos_in_world = self.next_pos_in_world
        self.next_pos_in_world = None
        self.previous_quat_in_world = self.quat_in_world
        self.quat_in_world = self.next_quat_in_world
        self.next_quat_in_world = None
        self.lin_vel = self.next_lin_vel
        self.next_lin_vel = None
        self.ang_vel = self.next_ang_vel
        self.next_ang_vel = None
        self.previous_left_foot_pos = self.left_foot_pos
        self.left_foot_pos = self.next_left_foot_pos
        self.next_left_foot_pos = None
        self.previous_left_foot_quat = self.left_foot_quat
        self.left_foot_quat = self.next_left_foot_quat
        self.next_left_foot_quat = None
        self.previous_right_foot_pos = self.right_foot_pos
        self.right_foot_pos = self.next_right_foot_pos
        self.next_right_foot_pos = None
        self.previous_right_foot_quat = self.right_foot_quat
        self.right_foot_quat = self.next_right_foot_quat
        self.next_right_foot_quat = None
        self.previous_joint_positions = self.joint_positions
        self.joint_positions = self.next_joint_positions
        self.next_joint_positions = None

    def is_alive(self):
        alive = True
        (x, y, z), (a, b, c, d), _, _, _, _ = self.get_head_pose()

        # head higher than starting position of body
        alive = alive and z > self.get_start_height()
        # angle of the robot in roll and pitch not to far from zero
        alive = alive and abs(self.imu_rpy[0] < math.tau / 4) and abs(self.imu_rpy[1] < math.tau / 4)
        return alive

    def reset_joints_to_init_pos(self):
        # set joints to initial position
        for name in self.joints:
            joint = self.joints[name]
            pos_in_rad = math.radians(self.initial_joints_positions[name])
            joint.reset_position(pos_in_rad, 0)
            joint.set_position(pos_in_rad)

    def reset(self):
        # todo this does not follow wxyz convention
        self.pose_on_episode_start = ([0, 0, 0.43], p.getQuaternionFromEuler((0, 0.25, 0)))
        self.reset_joints_to_init_pos()

        for sensor in self.pressure_sensors.values():
            sensor.reset()

        # reset body pose and velocity
        self.pybullet_client.resetBasePositionAndOrientation(self.robot_index, self.pose_on_episode_start[0],
                                                             self.pose_on_episode_start[1])
        self.pybullet_client.resetBaseVelocity(self.robot_index, [0, 0, 0], [0, 0, 0])
        self.update()
        # make sure we dont get artefacts in accelerometer
        self.last_lin_vel = self.lin_vel

    def reset_to_reference(self, refbot: "Robot", randomize):
        self.pose_on_episode_start = (refbot.pos_in_world, refbot.quat_in_world)
        self.pose_on_episode_start[0][2] += 0.07
        self.pybullet_client.resetBasePositionAndOrientation(self.robot_index, self.pose_on_episode_start[0],
                                                             wxyz2xyzw(self.pose_on_episode_start[1]))
        self.pybullet_client.resetBaseVelocity(self.robot_index, refbot.lin_vel, refbot.ang_vel)
        # set all joints to initial position since they can be modified from last fall
        self.reset_joints_to_init_pos()

        # first compute the joints via IK
        refbot.solve_ik_exactly()
        i = 0
        for joint_name in self.used_joint_names:
            # simple approximation of the velocity by taking difference to last positions
            vel = (refbot.joint_positions[i] - refbot.previous_joint_positions[i]) / (
                    refbot.time - refbot.previous_time)
            joint_pos = refbot.joint_positions[i]
            if randomize:
                joint_pos = random.uniform(-0.1, 0.1) + joint_pos
            self.joints[joint_name].reset_position(joint_pos, vel)
            i += 1
        self.update()
        self.last_lin_vel = self.lin_vel

    def reset_base_to_pose(self, pos, quat):
        self.pybullet_client.resetBasePositionAndOrientation(self.robot_index, pos, wxyz2xyzw(quat))
        self.pybullet_client.resetBaseVelocity(self.robot_index, [0, 0, 0], [0, 0, 0])

    def update_ref_in_sim(self):
        self.pybullet_client.resetBasePositionAndOrientation(self.robot_index, self.pos_in_world,
                                                             wxyz2xyzw(self.quat_in_world))
        self.solve_ik_exactly()
        i = 0
        for joint_name in self.used_joint_names:
            self.joints[joint_name].reset_position(self.joint_positions[i], 0)
            i += 1

    def solve_ik_exactly(self):
        # only compute if not already done to prevent unnecessary cpu load
        if self.joint_positions is None:
            # the trajectories are solvable. if computer load is high sometimes IK does not return meaningful result
            success = False
            for i in range(10):
                joint_results, success = compute_ik(self.left_foot_pos, self.left_foot_quat, self.right_foot_pos,
                                                    self.right_foot_quat, self.used_joint_names, self.joint_indexes,
                                                    collision=False)
                if success:
                    break
            if not success:
                # as fallback solve approximately
                print("needed to solve RT approximatly!!!")
                joint_results, success = compute_ik(self.left_foot_pos, self.left_foot_quat, self.right_foot_pos,
                                                    self.right_foot_quat, self.used_joint_names, self.joint_indexes,
                                                    collision=False, approximate=True)
            self.joint_positions = joint_results

    def solve_fk(self, force=False):
        # only solve if necessary
        if self.left_foot_pos is None or self.right_foot_pos is None or force:
            request = GetPositionFKRequest()
            for i in range(len(self.leg_joints)):
                # todo this only works when we just use only the legs
                request.robot_state.joint_state.name.append(self.leg_joints[i])
                request.robot_state.joint_state.position.append(self.joint_positions[i])
            request.fk_link_names = ['l_sole', 'r_sole']
            result = get_position_fk(request)  # type: GetPositionFKResponse
            l_sole = result.pose_stamped[result.fk_link_names.index('l_sole')].pose
            self.left_foot_pos = np.array([l_sole.position.x, l_sole.position.y, l_sole.position.z])
            l_sole_quat = l_sole.orientation
            self.left_foot_quat = np.array([l_sole_quat.w, l_sole_quat.x, l_sole_quat.y, l_sole_quat.z])
            r_sole = result.pose_stamped[result.fk_link_names.index('r_sole')].pose
            self.right_foot_pos = np.array([r_sole.position.x, r_sole.position.y, r_sole.position.z])
            r_sole_quat = r_sole.orientation
            self.right_foot_quat = np.array([r_sole_quat.w, r_sole_quat.x, r_sole_quat.y, r_sole_quat.z])

    def compute_velocities(self, time_diff):
        self.left_foot_lin_vel = (self.left_foot_pos - self.previous_left_foot_pos) / time_diff
        self.left_foot_ang_vel = (np.array(quat2euler(self.left_foot_quat)) - np.array(
            quat2euler(self.previous_left_foot_quat))) / time_diff

        self.right_foot_lin_vel = (self.right_foot_pos - self.previous_right_foot_pos) / time_diff
        self.right_foot_ang_vel = (np.array(quat2euler(self.right_foot_quat)) - np.array(
            quat2euler(self.previous_right_foot_quat))) / time_diff

    def apply_random_force(self, max_force, max_torque):
        force = [random.uniform(-max_force, max_force), random.uniform(-max_force, max_force),
                 random.uniform(-max_force, max_force)]
        torque = [random.uniform(-max_torque, max_torque), random.uniform(-max_torque, max_torque),
                  random.uniform(-max_torque, max_torque)]
        self.pybullet_client.applyExternalForce(self.robot_index, self.torso_id, force, [0, 0, 0], flags=p.WORLD_FRAME)
        self.pybullet_client.applyExternalTorque(self.robot_index, self.torso_id, torque, flags=p.WORLD_FRAME)

    def get_init_mu(self, cartesian_action, rot_type, refbot):
        """Get the mu values that can be used on learning start to search from a standing pose."""
        if cartesian_action:
            if False:
                # compute FK from initial joint positions
                request = GetPositionFKRequest()
                for name in self.initial_joints_positions.keys():
                    request.robot_state.joint_state.name.append(name)
                    request.robot_state.joint_state.position.append(math.radians(self.initial_joints_positions[name]))
                request.fk_link_names = ['l_sole', 'r_sole']
                result = get_position_fk(request)
                l_sole = result.pose_stamped[result.fk_link_names.index('l_sole')].pose
                l_rpy = quat2euler(
                    (l_sole.orientation.w, l_sole.orientation.x, l_sole.orientation.y, l_sole.orientation.z))
                r_sole = result.pose_stamped[result.fk_link_names.index('r_sole')].pose
                r_rpy = quat2euler(
                    (r_sole.orientation.w, r_sole.orientation.x, r_sole.orientation.y, r_sole.orientation.z))
                action = self.scale_pose_to_action([l_sole.position.x, l_sole.position.y, l_sole.position.z], l_rpy,
                                                   [r_sole.position.x, r_sole.position.y, r_sole.position.z], r_rpy,
                                                   rot_type)
            else:
                # compute from reference
                action = self.scale_pose_to_action(refbot.left_foot_pos, quat2euler(refbot.left_foot_quat),
                                                   refbot.right_foot_pos, quat2euler(refbot.right_foot_quat), rot_type)
            return action
        else:
            mu_values = []
            if False:
                # compute from initial joint position
                for joint_name in self.used_joint_names:
                    joint = self.joints[joint_name]
                    mu_values.append(
                        joint.convert_radiant_to_scaled(math.radians(self.initial_joints_positions[joint_name])))
            else:
                # compute from refbot
                i = 0
                for joint_name in self.used_joint_names:
                    joint = self.joints[joint_name]
                    mu_values.append(joint.convert_radiant_to_scaled(refbot.joint_positions[i]))
                    i += 1
            return mu_values

    def get_head_pose(self):
        return self.pybullet_client.getLinkState(self.robot_index, self.links["head"])

    def get_start_height(self):
        return self.pose_on_episode_start[0][2]

    def get_legs_in_world(self):
        self.solve_fk()
        mat_in_world = quat2mat(self.quat_in_world)
        world_to_robot_trans = compose(self.pos_in_world, mat_in_world, [1, 1, 1])
        # We have to append 1 for the matrix multiplication, we remove it afterwards
        left_leg_vector = [*self.left_foot_pos, 1]
        left_leg_in_world = np.matmul(world_to_robot_trans, left_leg_vector)[:-1]
        right_leg_vector = [*self.right_foot_pos, 1]
        right_leg_in_world = np.matmul(world_to_robot_trans, right_leg_vector)[:-1]
        return left_leg_in_world, right_leg_in_world

    def scale_joint_positions(self, positions):
        scaled = []
        i = 0
        for joint_name in self.used_joint_names:
            joint = self.joints[joint_name]
            scaled.append(joint.convert_radiant_to_scaled(positions[i]))
            i += 1
        return scaled

    def scale_action_to_motor_goal(self, action):
        motor_goal = []
        i = 0
        for joint_name in self.used_joint_names:
            joint = self.joints[joint_name]
            motor_goal.append(joint.convert_scaled_to_radiant(action[i]))
            i += 1
        return motor_goal

    def scale_action_to_pose(self, action, rot_type):
        # todo here we convert first to euler since our scaling is in this, maybe change this directly to quat somehow
        # action is structured as left_pos, left_rot, right_pos, right_rot
        # based on rot type we have different number of values. first transform to RPY then scale
        if rot_type == Rot.RPY:
            # nothing to to
            action_rpy = action
        elif rot_type == Rot.FUSED:
            left_fused = action[3:6]
            right_fused = action[9:13]
            # todo check if hemi is correct
            left_rpy = np.array(quat2euler(fused2quat(*left_fused, True)))
            right_rpy = np.array(quat2euler(fused2quat(*right_fused, True)))
            # stitch it back together
            action_rpy = np.concatenate([action[:3], left_rpy, action[6:9], right_rpy])
        elif rot_type == Rot.QUAT:
            left_quat = action[3:7]
            right_quat = action[10:14]
            left_rpy = np.array(quat2euler(left_quat))
            right_rpy = np.array(quat2euler(right_quat))
            action_rpy = np.concatenate([action[:3], left_rpy, action[7:10], right_rpy])
        elif rot_type == Rot.SIXD:
            left_sixd = action[3:9]
            right_sixd = action[12:18]
            left_rpy = np.array(quat2euler(sixd2quat(left_sixd)))
            right_rpy = np.array(quat2euler(sixd2quat(right_sixd)))
            action_rpy = np.concatenate([action[:3], left_rpy, action[9:12], right_rpy])
        scaled_left = []
        for i in range(6):
            scaled_left.append(
                action_rpy[i] * (self.cartesian_limits_left[i][1] - self.cartesian_limits_left[i][0]) / 2 +
                self.cartesian_mid_positions_left[i])
        scaled_right = []
        for i in range(6):
            scaled_right.append(
                action_rpy[i + 6] * (self.cartesian_limits_right[i][1] - self.cartesian_limits_right[i][0]) / 2 +
                self.cartesian_mid_positions_right[i])
        return np.array(scaled_left[:3]), np.array(scaled_left[3:6]), np.array(scaled_right[:3]), np.array(
            scaled_right[3:6])

    def scale_pose_to_action(self, left_pos, left_rpy, right_pos, right_rpy, rot_type):
        action = []
        for i in range(3):
            action.append(2 * (left_pos[i] - self.cartesian_mid_positions_left[i]) / (
                    self.cartesian_limits_left[i][1] - self.cartesian_limits_left[i][0]))
        rpy = []
        for i in range(3, 6):
            rpy.append(2 * (left_rpy[i - 3] - self.cartesian_mid_positions_left[i]) / (
                    self.cartesian_limits_left[i][1] - self.cartesian_limits_left[i][0]))
        if rot_type == Rot.RPY:
            action.extend(rpy)
        elif rot_type == Rot.FUSED:
            action.extend(quat2fused(euler2quat(*rpy))[:-1])
        elif rot_type == Rot.QUAT:
            action.extend(euler2quat(*rpy))
        elif rot_type == Rot.SIXD:
            action.extend(quat2sixd(euler2quat(*rpy)))
        for i in range(3):
            action.append(2 * (right_pos[i] - self.cartesian_mid_positions_right[i]) / (
                    self.cartesian_limits_right[i][1] - self.cartesian_limits_right[i][0]))
        rpy = []
        for i in range(3, 6):
            rpy.append(2 * (right_rpy[i - 3] - self.cartesian_mid_positions_right[i]) / (
                    self.cartesian_limits_right[i][1] - self.cartesian_limits_right[i][0]))
        if rot_type == Rot.RPY:
            action.extend(rpy)
        elif rot_type == Rot.FUSED:
            action.extend(quat2fused(euler2quat(*rpy))[:-1])
        elif rot_type == Rot.QUAT:
            action.extend(euler2quat(*rpy))
        elif rot_type == Rot.SIXD:
            action.extend(quat2sixd(euler2quat(*rpy)))
        return action

    def scale_action_to_relative_pose(self, action):
        # since we just shift pos and rpy relatively we can scale easily
        pos_left = self.left_foot_pos + np.array(action[0:3]) * self.relative_scaling_cartesian_action_pos
        rpy_left = np.array(quat2euler(self.left_foot_quat)) + np.array(
            action[3:6]) * self.relative_scaling_cartesian_action_ori
        pos_right = self.right_foot_pos + np.array(action[6:9]) * self.relative_scaling_cartesian_action_pos
        rpy_right = np.array(quat2euler(self.right_foot_quat)) + np.array(
            action[9:12]) * self.relative_scaling_cartesian_action_ori
        return pos_left, rpy_left, pos_right, rpy_right

    def scale_relative_pose_to_action(self, pos_left, rpy_left, pos_right, rpy_right):
        action_left_pos = (pos_left - self.previous_left_foot_pos) / self.relative_scaling_cartesian_action_pos
        action_left_rpy = (rpy_left - np.array(
            quat2euler(self.previous_left_foot_quat))) / self.relative_scaling_cartesian_action_ori
        action_right_pos = (pos_right - self.previous_right_foot_pos) / self.relative_scaling_cartesian_action_pos
        action_right_rpy = (rpy_right - np.array(
            quat2euler(self.previous_right_foot_quat))) / self.relative_scaling_cartesian_action_ori
        return np.concatenate([action_left_pos, action_left_rpy, action_right_pos, action_right_rpy])

    def set_joint_pos_vel(self, positions, velocities):
        # hacky method to let env run on actual robot
        for name, position, velocity in zip(self.used_joint_names, positions, velocities):
            self.joints[name].state = [position, velocity, 0, 0]

    def get_imu_msg(self):
        imu_quat = euler2quat(*self.imu_rpy, axes='sxyz')
        self.imu_msg.orientation = Quaternion(*wxyz2xyzw(imu_quat))
        self.imu_msg.angular_velocity = Vector3(*self.ang_vel)
        self.imu_msg.linear_acceleration = Vector3(*self.lin_acc)
        return self.imu_msg

    def get_pose_msg(self):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = Point(*self.pos_in_world)
        msg.pose.orientation = Quaternion(*wxyz2xyzw(self.quat_in_world))
        return msg

    def get_joint_state_msg(self):
        positions = []
        velocities = []
        efforts = []
        for name in self.joint_state_msg.name:
            joint = self.joints[name]
            positions.append(joint.get_position())
            velocities.append(joint.get_velocity())
            efforts.append(joint.get_torque())
        self.joint_state_msg.position = positions
        self.joint_state_msg.velocity = velocities
        self.joint_state_msg.effort = efforts
        return self.joint_state_msg

    def get_joint_position_as_msg(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.used_joint_names
        msg.position = self.joint_positions
        return msg

    def get_left_foot_msg(self):
        self.left_foot_msg.pose = Pose(Point(*self.left_foot_pos), Quaternion(*wxyz2xyzw(self.left_foot_quat)))
        return self.left_foot_msg

    def get_right_foot_msg(self):
        self.right_foot_msg.pose = Pose(Point(*self.right_foot_pos), Quaternion(*wxyz2xyzw(self.right_foot_quat)))
        return self.right_foot_msg

    def get_pressure_filtered_left(self):
        if len(self.pressure_sensors) == 0:
            print("No pressure sensors found in simulation model")
            return self.pressure_msg_left
        f_llb = self.pressure_sensors["LLB"].get_force()
        f_llf = self.pressure_sensors["LLF"].get_force()
        f_lrf = self.pressure_sensors["LRF"].get_force()
        f_lrb = self.pressure_sensors["LRB"].get_force()
        self.pressure_msg_left.left_back = f_llb[1]
        self.pressure_msg_left.left_front = f_llf[1]
        self.pressure_msg_left.right_front = f_lrf[1]
        self.pressure_msg_left.right_back = f_lrb[1]
        return self.pressure_msg_left

    def get_pressure_filtered_right(self):
        if len(self.pressure_sensors) == 0:
            print("No pressure sensors found in simulation model")
            return self.pressure_msg_right
        f_rlb = self.pressure_sensors["RLB"].get_force()
        f_rlf = self.pressure_sensors["RLF"].get_force()
        f_rrf = self.pressure_sensors["RRF"].get_force()
        f_rrb = self.pressure_sensors["RRB"].get_force()
        self.pressure_msg_right.left_back = f_rlb[1]
        self.pressure_msg_right.left_front = f_rlf[1]
        self.pressure_msg_right.right_front = f_rrf[1]
        self.pressure_msg_right.right_back = f_rrb[1]
        return self.pressure_msg_right

    def set_alpha(self, alpha):
        if self.alpha != alpha:
            # only change if the value changed, for better performance
            ref_col = [1, 1, 1, alpha]
            self.pybullet_client.changeVisualShape(self.robot_index, -1, rgbaColor=ref_col)
            for l in range(self.pybullet_client.getNumJoints(self.robot_index)):
                self.pybullet_client.changeVisualShape(self.robot_index, l, rgbaColor=ref_col)
            self.alpha = alpha

    def _disable_physics(self):
        self.pybullet_client.changeDynamics(self.robot_index, -1, linearDamping=0, angularDamping=0)
        self.pybullet_client.setCollisionFilterGroupMask(self.robot_index, -1, collisionFilterGroup=0,
                                                         collisionFilterMask=0)
        self.pybullet_client.changeDynamics(self.robot_index, -1,
                                            activationState=self.pybullet_client.ACTIVATION_STATE_SLEEP +
                                                            self.pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                            self.pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
        num_joints = self.pybullet_client.getNumJoints(self.robot_index)
        for j in range(num_joints):
            self.pybullet_client.setCollisionFilterGroupMask(self.robot_index, j, collisionFilterGroup=0,
                                                             collisionFilterMask=0)
            self.pybullet_client.changeDynamics(self.robot_index, j,
                                                activationState=self.pybullet_client.ACTIVATION_STATE_SLEEP +
                                                                self.pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                                self.pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)


class Joint:
    def __init__(self, joint_index, body_index, pybullet_client):
        self.pybullet_client = pybullet_client
        self.joint_index = joint_index
        self.body_index = body_index
        joint_info = self.pybullet_client.getJointInfo(self.body_index, self.joint_index)
        self.name = joint_info[1].decode('utf-8')
        self.type = joint_info[2]
        self.max_force = joint_info[10]
        self.max_velocity = joint_info[11]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        self.mid_position = 0.5 * (self.lowerLimit + self.upperLimit)
        position, velocity, forces, applied_torque = self.pybullet_client.getJointState(self.body_index,
                                                                                        self.joint_index)
        self.state = position, velocity, forces, applied_torque

    def update(self):
        """
        Called just once per step to update state from simulation. Improves performance.
        """
        position, velocity, forces, applied_torque = self.pybullet_client.getJointState(self.body_index,
                                                                                        self.joint_index)
        self.state = position, velocity, forces, applied_torque
        return position, velocity, applied_torque

    def reset_position(self, position, velocity):
        self.pybullet_client.resetJointState(self.body_index, self.joint_index, targetValue=position,
                                             targetVelocity=velocity)

    def disable_motor(self):
        self.pybullet_client.setJointMotorControl2(self.body_index, self.joint_index,
                                                   controlMode=self.pybullet_client.POSITION_CONTROL, targetPosition=0,
                                                   targetVelocity=0, positionGain=0.1, velocityGain=0.1, force=0)

    def set_position(self, position):
        # enforce limits
        position = min(self.upperLimit, max(self.lowerLimit, position))
        self.pybullet_client.setJointMotorControl2(self.body_index, self.joint_index,
                                                   self.pybullet_client.POSITION_CONTROL,
                                                   targetPosition=position, force=self.max_force,
                                                   maxVelocity=self.max_velocity)

    def set_scaled_position(self, position):
        self.set_position(self.convert_scaled_to_radiant(position))

    def reset_scaled_position(self, position):
        # sets position inside limits with a given position values in [-1, 1]
        self.reset_position(self.convert_scaled_to_radiant(position), 0)

    def get_state(self):
        return self.state

    def get_position(self):
        return self.state[0]

    def get_scaled_position(self):
        return self.convert_radiant_to_scaled(self.state[0])

    def get_velocity(self):
        return self.state[1]

    def get_scaled_velocity(self):
        return self.get_velocity() * 0.01

    def get_torque(self):
        position, velocity, forces, applied_torque = self.state
        return applied_torque

    def convert_radiant_to_scaled(self, pos):
        # helper method to convert to scaled position between [-1,1] for this joint using min max scaling
        return 2 * (pos - self.mid_position) / (self.upperLimit - self.lowerLimit)

    def convert_scaled_to_radiant(self, position):
        # helper method to convert to scaled position for this joint using min max scaling
        return position * (self.upperLimit - self.lowerLimit) / 2 + self.mid_position


class PressureSensor:
    def __init__(self, name, joint_index, body_index, cutoff, order, pybullet_client):
        self.pybullet_client = pybullet_client
        self.joint_index = joint_index
        self.name = name
        self.body_index = body_index
        nyq = 240 * 0.5  # nyquist frequency from simulation frequency
        normalized_cutoff = cutoff / nyq  # cutoff freq in hz
        self.filter_b, self.filter_a = signal.butter(order, normalized_cutoff, btype='low')
        self.filter_state = None
        self.reset()
        self.unfiltered = 0
        self.filtered = [0]

    def reset(self):
        self.filter_state = signal.lfilter_zi(self.filter_b, self.filter_a)

    def filter_step(self, unfiltered=None):
        if unfiltered is None:
            self.unfiltered = self.pybullet_client.getJointState(self.body_index, self.joint_index)[2][2] * -1
        self.filtered, self.filter_state = signal.lfilter(self.filter_b, self.filter_a, [self.unfiltered],
                                                          zi=self.filter_state)

    def get_force(self):
        return max(self.unfiltered, 0), max(self.filtered[0], 0)

    def get_value(self, type):
        if type == "filtered":
            return max(self.filtered[0], 0)
        elif type == "raw":
            return max(self.unfiltered, 0)
        elif type == "binary":
            if self.unfiltered > 10:
                return 1
            else:
                return 0
        else:
            print(f"type '{type}' not know")
