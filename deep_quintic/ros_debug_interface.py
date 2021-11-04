import math
import time

import numpy as np
import rospy
from bitbots_msgs.msg import FootPressure, JointCommand
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32, Bool, Float32MultiArray

# from wolfgang_pybullet_sim.cfg import simConfig
from dynamic_reconfigure.server import Server

from deep_quintic.utils import wxyz2xyzw


class ROSDebugInterface:
    def __init__(self, env, init_node=True):
        if init_node:
            rospy.init_node("pybullet_sim", anonymous=True)

        self.env = env
        self.last_time = time.time()
        self.last_linear_vel = (0, 0, 0)

        # messages
        self.real_time_msg = Float32()
        self.joint_state_msg = JointState()
        self.joint_state_msg.header.frame_id = "base_link"
        self.joint_state_msg.name = self.env.robot.initial_joint_positions.keys()
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "imu_frame"
        self.clock_msg = Clock()
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "base_link"

        # publisher
        self.left_foot_pressure_publisher = rospy.Publisher("foot_pressure_raw/left", FootPressure, queue_size=1)
        self.right_foot_pressure_publisher = rospy.Publisher("foot_pressure_raw/right", FootPressure, queue_size=1)
        self.left_foot_pressure_publisher_filtered = rospy.Publisher("foot_pressure_filtered/left", FootPressure,
                                                                     queue_size=1)
        self.right_foot_pressure_publisher_filtered = rospy.Publisher("foot_pressure_filtered/right", FootPressure,
                                                                      queue_size=1)
        self.joint_publisher = rospy.Publisher("joint_states", JointState, queue_size=1)
        self.imu_publisher = rospy.Publisher("imu/data", Imu, queue_size=1)
        self.imu_rpy_publisher = rospy.Publisher("imu_rpy", Float32MultiArray, queue_size=1)
        self.clock_publisher = rospy.Publisher("clock", Clock, queue_size=1)
        self.real_time_factor_publisher = rospy.Publisher("real_time_factor", Float32, queue_size=1)
        self.true_odom_publisher = rospy.Publisher("true_odom", Odometry, queue_size=1)

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

    def publish(self):
        self.env.state.publish_debug()
        self.env.reward_function.publish_reward()

        self.publish_true_odom()
        self.publish_foot_pressure()
        self.publish_imu()
        self.publish_real_time_factor()
        self.publish_refbot()
        self.publish_clock()
        self.publish_vels()
        self.publish_action()

    def publish_action(self):
        if self.action_publisher.get_num_connections() > 0:
            action_msg = Float32MultiArray()
            action_msg.data = self.env.last_action
            self.action_publisher.publish(action_msg)

        if self.action_publisher_not_normalized.get_num_connections() > 0:
            action_msg = Float32MultiArray()
            left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = \
                self.env.robot.scale_action_to_pose(self.env.last_action, self.env.rot_type)
            action_msg.data = np.concatenate([left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy])
            self.action_publisher_not_normalized.publish(action_msg)

    def publish_vels(self):
        if self.lin_vel_publisher.get_num_connections() > 0:
            lin_msg = Float32MultiArray()
            lin_msg.data = self.env.robot.lin_vel
            self.lin_vel_publisher.publish(lin_msg)

        if self.ang_vel_publisher.get_num_connections() > 0:
            ang_msg = Float32MultiArray()
            ang_msg.data = self.env.robot.ang_vel
            self.ang_vel_publisher.publish(ang_msg)

    def publish_clock(self):
        clock_msg = Clock()
        clock_msg.clock = rospy.Time.from_sec(self.env.time)
        self.clock_publisher.publish(clock_msg)

    def publish_refbot(self):
        if self.refbot_joint_publisher.get_num_connections() > 0:
            self.env.refbot.solve_ik_exactly()
            self.refbot_joint_publisher.publish(self.env.refbot.get_joint_state_msg(True))
        if self.refbot_left_foot_publisher.get_num_connections() > 0:
            self.refbot_left_foot_publisher.publish(self.env.refbot.get_left_foot_msg())
        if self.refbot_right_foot_publisher.get_num_connections() > 0:
            self.refbot_right_foot_publisher.publish(self.env.refbot.get_right_foot_msg())
        if self.refbot_pose_publisher.get_num_connections() > 0:
            self.refbot_pose_publisher.publish(self.env.refbot.get_pose_msg())

    def publish_real_time_factor(self):
        time_now = time.time()
        self.real_time_msg.data = self.env.env_timestep / (time_now - self.last_time)
        self.last_time = time_now
        self.real_time_factor_publisher.publish(self.real_time_msg)

        if self.joint_publisher.get_num_connections() > 0:
            self.joint_publisher.publish(self.env.robot.get_joint_state_msg(stamped=True))

    def publish_imu(self):
        if self.imu_publisher.get_num_connections() > 0:
            self.imu_publisher.publish(self.env.robot.get_imu_msg())

        if self.imu_rpy_publisher.get_num_connections() > 0:
            rpy_msg = Float32MultiArray()
            rpy_msg.data.append(math.degrees(self.env.robot.imu_rpy[0]))
            rpy_msg.data.append(math.degrees(self.env.robot.imu_rpy[1]))
            rpy_msg.data.append(math.degrees(self.env.robot.imu_rpy[2]))
            self.imu_rpy_publisher.publish(rpy_msg)

    def publish_foot_pressure(self):
        if self.left_foot_pressure_publisher.get_num_connections() > 0 and self.env.foot_sensors_type != "":
            self.left_foot_pressure_publisher.publish(self.env.robot.get_pressure(left=True, filtered=False))

        if self.right_foot_pressure_publisher.get_num_connections() > 0 and self.env.foot_sensors_type != "":
            self.right_foot_pressure_publisher.publish(self.env.robot.get_pressure(left=False, filtered=False))

        if self.left_foot_pressure_publisher_filtered.get_num_connections() > 0 and \
                self.env.foot_sensors_type == "filtered":
            self.left_foot_pressure_publisher_filtered.publish(self.env.robot.get_pressure(left=True, filtered=True))

        if self.right_foot_pressure_publisher_filtered.get_num_connections() > 0 and \
                self.env.foot_sensors_type == "filtered":
            self.right_foot_pressure_publisher_filtered.publish(self.env.robot.get_pressure(left=False, filtered=True))

    def publish_true_odom(self):
        if self.true_odom_publisher.get_num_connections() > 0:
            self.odom_msg.pose.pose.position = Point(*self.env.robot.pos_in_world)
            self.odom_msg.pose.pose.orientation = Quaternion(*wxyz2xyzw(self.env.robot.quat_in_world))
            self.true_odom_publisher.publish(self.odom_msg)
