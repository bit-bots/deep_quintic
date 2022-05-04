import random
import math
import rclpy
from rclpy.node import Node
import numpy as np

from transforms3d.quaternions import quat2mat, rotate_vector, qinverse, qmult, mat2quat
from transforms3d.affines import compose
from transforms3d.euler import quat2euler, euler2quat

from bitbots_moveit_bindings import get_position_ik, get_position_fk
from bitbots_msgs.msg import FootPressure
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3

from deep_quintic.complementary_filter import ComplementaryFilter
from deep_quintic.simulation import AbstractSim, PybulletSim
from deep_quintic.utils import Rot, compute_ik
from bitbots_utils.transforms import fused2quat, sixd2quat, quat2fused, quat2sixd, wxyz2xyzw, xyzw2wxyz, compute_imu_orientation_from_world


class Robot:
    def __init__(self, node: Node, simulation: AbstractSim = None, compute_joints=False, compute_feet=False,
                 used_joints="Legs", physics=False, compute_smooth_vel=False, use_complementary_filter=True):
        self.node = node
        self.physics_active = physics
        self.sim = simulation
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
        self.imu_rpy = [0, 0, 0]
        self.imu_ang_vel = [0, 0, 0]
        self.imu_lin_acc = [0, 0, 0]
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
        self.pose_on_episode_start = [0, 0, 0.5], [1, 0, 0, 0]
        self.complementary_filter = None
        if use_complementary_filter:
            self.complementary_filter = ComplementaryFilter()
        self.last_ik_result = None

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

        self.initial_joint_positions = {"LAnklePitch": -30, "LAnkleRoll": 0, "LHipPitch": 30, "LHipRoll": 0,
                                        "LHipYaw": 0, "LKnee": 60, "RAnklePitch": 30, "RAnkleRoll": 0,
                                        "RHipPitch": -30, "RHipRoll": 0, "RHipYaw": 0, "RKnee": -60,
                                        "LShoulderPitch": 75, "LShoulderRoll": 0, "LElbow": 36,
                                        "RShoulderPitch": -75, "RShoulderRoll": 0, "RElbow": -36, "HeadPan": 0,
                                        "HeadTilt": 0}

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
        self.joint_state_msg.name = list(self.initial_joint_positions.keys())
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "imu_frame"
        self.pressure_msg = FootPressure()
        self.left_foot_msg = PoseStamped()
        self.left_foot_msg.header.frame_id = 'base_link'
        self.right_foot_msg = PoseStamped()
        self.right_foot_msg.header.frame_id = 'base_link'
        self.robot_index = None

        if self.sim is not None:
            self.links = {}
            self.torso_id = {}
            self.link_masses = []
            self.link_inertias = []
            self.joint_stall_torques = []
            self.joint_max_vels = []
            self.sim.set_initial_joint_positions(self.initial_joint_positions)
            self.robot_index = self.sim.add_robot(physics)

    def transform_world_to_robot(self, pos_in_world, quat_in_world, lin_vel_in_world, ang_vel_in_world):
        # transform from world to relative to base link
        robot_mat_in_world = quat2mat(self.quat_in_world)
        # normally we would just do this, but this takes a lot of CpU. so we do it manually
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

    def update_complementary_filter(self, dt):
        # get measured imu values if available
        if not isinstance(self.sim, PybulletSim):
            self.imu_ang_vel = self.sim.get_imu_ang_vel()
            self.imu_lin_acc = self.sim.get_imu_lin_acc()
        else:
            # otherwise use the true values from simulation
            # simple acceleration computation by using diff of velocities
            linear_acc = np.array(list(map(lambda i, j: i - j, self.last_lin_vel, self.lin_vel)))
            # adding gravity to the acceleration, following REP-145
            gravity_world_frame = np.array([0, 0, 9.81])
            gravity_robot_frame = rotate_vector(gravity_world_frame, qinverse(self.quat_in_world))
            lin_acc = linear_acc + gravity_robot_frame

            lin_vel_in_world, ang_vel_in_world = self.sim.get_base_velocity(self.robot_index)
            ang_vel = rotate_vector(ang_vel_in_world, qinverse(self.quat_in_world))

            self.imu_ang_vel = ang_vel
            self.imu_lin_acc = lin_acc

        # don't take true value from simulation but estimation from imu filter
        if (self.ang_vel[0] != 0 or self.ang_vel[1] != 0 or self.ang_vel[2] != 0) and (self.imu_lin_acc[0] != 0 or \
                                                                                       self.imu_lin_acc[1] != 0 or
                                                                                       self.imu_lin_acc[2] != 0):
            self.complementary_filter.update(*self.imu_lin_acc, *self.imu_ang_vel, dt)

        if self.complementary_filter.getDoBiasEstimation():
            self.imu_ang_vel[0] -= self.complementary_filter.getAngularVelocityBiasX()
            self.imu_ang_vel[1] -= self.complementary_filter.getAngularVelocityBiasY()
            self.imu_ang_vel[2] -= self.complementary_filter.getAngularVelocityBiasZ()

    def update(self):
        """
        Updates the state of the robot from simulation. This is only done once per step to improve performance.
        """
        self.previous_pos_in_world = self.pos_in_world
        self.previous_quat_in_world = self.quat_in_world
        self.pos_in_world, self.quat_in_world = self.sim.get_base_position_and_orientation(self.robot_index)

        # imu orientation has roll and pitch relative to gravity vector. yaw in world frame
        self.imu_rpy, yaw_quat = compute_imu_orientation_from_world(self.quat_in_world)
        # rotate velocities from world to robot frame
        self.last_lin_vel = self.lin_vel
        lin_vel_in_world, ang_vel_in_world = self.sim.get_base_velocity(self.robot_index)
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

        if self.complementary_filter is not None:
            imu_quat = self.complementary_filter.getOrientation()
            imu_rpy = quat2euler(imu_quat)
            self.imu_rpy = [imu_rpy[0], imu_rpy[1], 0]

        # compute joints
        if self.compute_joints:
            self.previous_joint_positions = self.joint_positions
            self.joint_positions, self.joint_velocities, self.joint_torques = self.sim.get_joint_values(
                self.used_joint_names, self.robot_index)

        # foot poses
        if self.compute_feet:
            # left foot
            foot_pos_in_world, foot_quat_in_world, lin_vel_in_world, ang_vel_in_world = self.sim.get_link_values(
                "l_sole", self.robot_index)
            self.left_foot_pos, self.left_foot_quat, self.left_foot_lin_vel, self.left_foot_ang_vel = self.transform_world_to_robot(
                foot_pos_in_world, foot_quat_in_world, lin_vel_in_world, ang_vel_in_world)

            # right foot
            foot_pos_in_world, foot_quat_in_world, lin_vel_in_world, ang_vel_in_world = self.sim.get_link_values(
                "r_sole", self.robot_index)
            self.right_foot_pos, self.right_foot_quat, self.right_foot_lin_vel, self.right_foot_ang_vel = self.transform_world_to_robot(
                foot_pos_in_world, foot_quat_in_world, lin_vel_in_world, ang_vel_in_world)

    def apply_action(self, action, cartesian, relative, refbot, rot_type):
        if not cartesian:
            i = 0
            # iterate through all joints, always in same order to keep the same matching between NN and simulation
            for joint_name in self.used_joint_names:
                if relative:
                    goal_position = action[i] * self.relative_scaling_joint_action  # todo makes this sense?
                else:
                    goal_position = action[i]
                self.sim.set_joint_position(joint_name, goal_position, scaled=True, relative=relative,
                                            robot_index=self.robot_index)
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
            self.last_ik_result = ik_result
            if success:
                i = 0
                for joint_name in self.leg_joints:
                    self.sim.set_joint_position(joint_name, ik_result[i], robot_index=self.robot_index)
                    i += 1
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

    def step_pressure_filters(self):
        self.sim.step_pressure_filters(self.robot_index)

    def is_alive(self):
        alive = True
        (x, y, z), _, _, _ = self.sim.get_link_values("head", self.robot_index)

        # head higher than starting position of body
        alive = alive and z > self.get_start_height()
        # angle of the robot in roll and pitch not to far from zero
        rpy = quat2euler(self.quat_in_world)
        alive = alive and abs(rpy[0]) < math.tau / 4 and abs(rpy[1]) < math.tau / 4
        return alive

    def reset(self):
        self.pose_on_episode_start = ([0, 0, 0.43], euler2quat(0, 0.25, 0))
        self.sim.reset_joints_to_init_pos(self.robot_index)

        self.sim.reset_pressure_filters(self.robot_index)
        if self.complementary_filter is not None:
            self.complementary_filter.reset(euler2quat(0, 0.25, 0))

        # reset body pose and velocity
        self.sim.reset_base_position_and_orientation(self.pose_on_episode_start[0],
                                                     self.pose_on_episode_start[1], self.robot_index)
        self.sim.reset_base_velocity([0, 0, 0], [0, 0, 0], self.robot_index)
        self.update()
        # make sure we dont get artefacts in accelerometer
        self.lin_vel = [0, 0, 0]
        self.last_lin_vel = self.lin_vel

    def reset_to_reference(self, refbot: "Robot", randomize, additional_height=0):
        self.pose_on_episode_start = (refbot.pos_in_world, refbot.quat_in_world)
        self.pose_on_episode_start[0][2] += additional_height
        self.sim.reset_base_position_and_orientation(self.pose_on_episode_start[0],
                                                     self.pose_on_episode_start[1], self.robot_index)
        self.sim.reset_base_velocity(refbot.lin_vel, refbot.ang_vel, self.robot_index)
        # set all joints to initial position since they can be modified from last fall
        self.sim.reset_joints_to_init_pos(self.robot_index)
        self.sim.reset_pressure_filters(self.robot_index)
        if self.complementary_filter is not None:
            self.complementary_filter.reset(refbot.quat_in_world)

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
            self.sim.reset_joint_to_position(joint_name, joint_pos, vel, self.robot_index)
            i += 1
        self.update()
        self.last_lin_vel = self.lin_vel

    def reset_base_to_pose(self, pos, quat):
        self.sim.reset_base_position_and_orientation(self.robot_index, pos, quat)
        self.sim.reset_base_velocity(self.robot_index, [0, 0, 0], [0, 0, 0])

    def update_ref_in_sim(self):
        self.sim.reset_base_position_and_orientation(self.pos_in_world, self.quat_in_world, self.robot_index)
        self.solve_ik_exactly()
        i = 0
        for joint_name in self.used_joint_names:
            self.sim.reset_joint_to_position(joint_name, self.joint_positions[i], velocity=0,
                                             robot_index=self.robot_index)
            i += 1

    def solve_ik_exactly(self):
        # only compute if not already done to prevent unnecessary cpu load
        joint_results = None
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
            request = GetPositionFK.Request()
            for i in range(len(self.leg_joints)):
                # todo this only works when we just use only the legs
                request.robot_state.joint_state.name.append(self.leg_joints[i])
                request.robot_state.joint_state.position.append(self.joint_positions[i])
            request.fk_link_names = ['l_sole', 'r_sole']
            result = get_position_fk(request)  # type: GetPositionFK.Response
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
        self.sim.apply_external_force_to_base(force, self.robot_index)
        self.sim.apply_external_torque_to_base(torque, self.robot_index)

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
                mu_values = self.scale_pose_to_action([l_sole.position.x, l_sole.position.y, l_sole.position.z], l_rpy,
                                                      [r_sole.position.x, r_sole.position.y, r_sole.position.z], r_rpy,
                                                      rot_type)
            else:
                # compute from reference
                mu_values = self.scale_pose_to_action(refbot.left_foot_pos, quat2euler(refbot.left_foot_quat),
                                                      refbot.right_foot_pos, quat2euler(refbot.right_foot_quat),
                                                      rot_type)
            return mu_values
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
                    mu_values.append(self.sim.convert_radiant_to_scaled(joint_name, refbot.joint_positions[i]))
                    i += 1
            return mu_values

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

    def joints_radiant_to_scaled(self, positions):
        scaled = []
        i = 0
        for joint_name in self.used_joint_names:
            scaled.append(self.sim.convert_radiant_to_scaled(joint_name, positions[i], self.robot_index))
            i += 1
        return scaled

    def joints_scaled_to_radiant(self, action):
        motor_goal = []
        i = 0
        for joint_name in self.used_joint_names:
            motor_goal.append(self.sim.convert_scaled_to_radiant(joint_name, action[i], self.robot_index))
            i += 1
        return motor_goal

    def scale_action_to_pose(self, action, rot_type):
        # todo here we convert first to euler since our scaling is in this, maybe change this directly to quat somehow
        # action is structured as left_pos, left_rot, right_pos, right_rot
        # based on rot type we have different number of values. first transform to RpY then scale
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
        # todo refactor
        # hacky method to let env run on actual robot
        for name, position, velocity in zip(self.used_joint_names, positions, velocities):
            self.joints[name].state = [position, velocity, 0, 0]

    def get_imu_msg(self):
        imu_quat = euler2quat(*self.imu_rpy, axes='sxyz')
        # change to ros standard
        quat = wxyz2xyzw(imu_quat)
        self.imu_msg.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        self.imu_msg.angular_velocity = Vector3(x=self.imu_ang_vel[0], y=self.imu_ang_vel[1], z=self.imu_ang_vel[2])
        self.imu_msg.linear_acceleration = Vector3(x=self.imu_lin_acc[0], y=self.imu_lin_acc[1], z=self.imu_lin_acc[2])
        return self.imu_msg

    def get_pose_msg(self):
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position = Point(x=self.pos_in_world[0], y=self.pos_in_world[1], z=self.pos_in_world[2])
        quat = wxyz2xyzw(self.quat_in_world)
        msg.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        return msg

    def get_joint_state_msg(self, stamped=False):
        if stamped:
            self.joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        if self.joint_positions is None:
            self.joint_positions, self.joint_velocities, self.joint_torques = self.sim.get_joint_values(
                self.used_joint_names, self.robot_index)
        self.joint_state_msg.position = self.joint_positions
        if self.joint_velocities is not None:
            self.joint_state_msg.velocity = self.joint_velocities
        if self.joint_torques is not None:
            self.joint_state_msg.effort = self.joint_torques
        return self.joint_state_msg

    def get_left_foot_msg(self):
        quat = wxyz2xyzw(self.left_foot_quat)
        self.left_foot_msg.pose = Pose(
            position=Point(x=self.left_foot_pos[0], y=self.left_foot_pos[1], z=self.left_foot_pos[2]),
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))
        return self.left_foot_msg

    def get_right_foot_msg(self):
        quat = wxyz2xyzw(self.right_foot_quat)
        self.right_foot_msg.pose = Pose(
            position=Point(x=self.right_foot_pos[0], y=self.right_foot_pos[1], z=self.right_foot_pos[2]),
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))
        return self.right_foot_msg

    def get_pressure(self, left, filtered, time=True):
        if left:
            names = ["LLB", "LLF", "LRF", "LRB"]
            self.pressure_msg.header.frame_id = 'l_sole'
        else:
            names = ["RLB", "RLF", "RRF", "RRB"]
            self.pressure_msg.header.frame_id = 'r_sole'

        self.pressure_msg.left_back = self.sim.get_sensor_force(names[0], filtered, self.robot_index)
        self.pressure_msg.left_front = self.sim.get_sensor_force(names[1], filtered, self.robot_index)
        self.pressure_msg.right_front = self.sim.get_sensor_force(names[2], filtered, self.robot_index)
        self.pressure_msg.right_back = self.sim.get_sensor_force(names[3], filtered, self.robot_index)
        if time:
            self.pressure_msg.header.stamp = self.node.get_clock().now().to_msg()
        return self.pressure_msg

    def set_alpha(self, alpha):
        if self.alpha != alpha:
            # only change if the value changed, for better performance
            self.alpha = alpha
            self.sim.set_alpha(alpha, self.robot_index)

    def set_random_head_goals(self):
        pan = random.uniform(-1, 1)
        self.sim.set_joint_position("HeadPan", pan, scaled=True, relative=False, robot_index=self.robot_index)
        tilt = random.uniform(-1, 1)
        self.sim.set_joint_position("HeadTilt", tilt, scaled=True, relative=False, robot_index=self.robot_index)
