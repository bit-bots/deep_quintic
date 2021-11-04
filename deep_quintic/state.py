import math
import random
from abc import ABC
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from rospy import Publisher
from std_msgs.msg import Float32MultiArray

from typing import TYPE_CHECKING

from transforms3d.euler import euler2quat, quat2euler

from deep_quintic.utils import Rot, wxyz2xyzw, quat2fused, quat2sixd

if TYPE_CHECKING:
    from deep_quintic import DeepQuinticEnv


class State:
    def __init__(self, env: "DeepQuinticEnv"):
        self.env = env
        self.debug_names = None
        self.debug_publishers = {}

    def publish_debug(self):
        entries = self.get_state_entries(True)
        if len(self.debug_publishers.keys()) == 0:
            # initialize publishers
            for name in entries.keys():
                self.debug_publishers[name] = Publisher("state_" + name, Float32MultiArray, queue_size=1)
        for entry_name in entries.keys():
            publisher = self.debug_publishers[entry_name]
            if publisher.get_num_connections() > 0:
                publisher.publish(Float32MultiArray(data=entries[entry_name]))

    def get_state_entries(self, scaled):
        raise NotImplementedError

    def get_state_array(self, scaled):
        return np.concatenate(list(self.get_state_entries(scaled).values()))

    def get_num_observations(self):
        return len(self.get_state_array(True))


class PhaseState(State):
    def __init__(self, env: "DeepQuinticEnv"):
        super().__init__(env)

    def get_state_entries(self, scaled):
        output = dict()  # is ordered in 3.7+
        output["cmd_vel"] = self.env.current_command_speed
        if self.env.phase_in_state:
            if self.env.cyclic_phase:
                output["phase"] = [math.sin(self.env.refbot.phase * math.tau),
                                   math.cos(self.env.refbot.phase * math.tau)]
            else:
                output["phase"] = [self.env.refbot.phase]
        return output


class BaseState(PhaseState):
    def __init__(self, env: "DeepQuinticEnv", use_foot_sensors, randomize):
        super().__init__(env)
        self.use_foot_sensors = use_foot_sensors
        self.randomize = False  # todo deactivated

    def get_state_entries(self, scaled):
        output = super(BaseState, self).get_state_entries(scaled)
        # it only makes sense to display IMU rot in fused or RPY since we do not have yaw direction
        # todo they are all originally computed from euler. can this lead to problems?
        rpy = deepcopy(self.env.robot.imu_rpy)
        if self.randomize and len(rpy) == 3:
            for i in range(3):
                rpy[i] = random.gauss(rpy[i], 0.05)
        if self.env.rot_type == Rot.RPY:
            roll = rpy[0]
            pitch = rpy[1]
            output["roll"] = [roll / (math.tau / 4)] if scaled else [roll]
            output["pitch"] = [pitch / (math.tau / 4)] if scaled else [pitch]
        elif self.env.rot_type == Rot.FUSED:
            quat = euler2quat(*rpy, axes='sxyz')
            fused_roll, fused_pitch, fused_yaw, hemi = quat2fused(quat)
            roll = fused_roll
            pitch = fused_pitch
            output["roll"] = [roll / (math.tau / 4)] if scaled else [roll]
            output["pitch"] = [pitch / (math.tau / 4)] if scaled else [pitch]
        elif self.env.rot_type == Rot.QUAT:
            # quat is scaled from -1 to 1
            output["quat"] = euler2quat(*rpy, axes='sxyz')
        elif self.env.rot_type == Rot.SIXD:
            # is already scaled from -1 to 1
            output["sixd"] = quat2sixd(euler2quat(*rpy, axes='sxyz'))
        else:
            print("ROTATION NOT KNOWN")
            exit()

        ang_vel = deepcopy(self.env.robot.imu_ang_vel)
        if self.randomize:
            for i in range(3):
                ang_vel[i] = random.gauss(ang_vel[i], 0.2)
        if scaled:
            ang_vel = ang_vel / np.array(20)
        output["ang_vel"] = ang_vel

        if self.use_foot_sensors != "":
            foot_pressures = self.get_pressure_array(self.use_foot_sensors)

            if self.use_foot_sensors == "binary":
                output["foot_pressure"] = foot_pressures
            else:
                output["foot_pressure"] = foot_pressures / np.array(100) if scaled else foot_pressures

        if self.env.use_rt_in_state:
            if self.env.relative:
                print("not implemented")
                exit()
            else:
                if self.env.cartesian_action:
                    output["ref_action"] = self.env.robot.scale_pose_to_action(
                        self.env.refbot.next_left_foot_pos,
                        quat2euler(self.env.refbot.next_left_foot_quat),
                        self.env.refbot.next_right_foot_pos,
                        quat2euler(self.env.refbot.next_right_foot_quat),
                        self.env.rot_type)
                else:
                    print("not implemented")
                    exit()

        return output

    def get_pressure_array(self, type):
        robot_index = self.env.robot.robot_index
        filtered = type == "filtered"
        return [self.env.sim.get_sensor_force("LLB", filtered, robot_index),
                self.env.sim.get_sensor_force("LLF", filtered, robot_index),
                self.env.sim.get_sensor_force("LRF", filtered, robot_index),
                self.env.sim.get_sensor_force("LRB", filtered, robot_index),
                self.env.sim.get_sensor_force("RLB", filtered, robot_index),
                self.env.sim.get_sensor_force("RLF", filtered, robot_index),
                self.env.sim.get_sensor_force("RRF", filtered, robot_index),
                self.env.sim.get_sensor_force("RRB", filtered, robot_index)]


class CartesianState(BaseState):
    def __init__(self, env: "DeepQuinticEnv", use_foot_sensors, leg_vel_in_state, randomize):
        super().__init__(env, use_foot_sensors, randomize)
        self.leg_vel_in_state = leg_vel_in_state

    def get_state_entries(self, scaled):
        output = super(CartesianState, self).get_state_entries(scaled)
        output["left_pos"] = self.env.robot.left_foot_pos
        output["right_pos"] = self.env.robot.right_foot_pos

        if self.leg_vel_in_state:
            output["left_lin"] = self.env.robot.left_foot_lin_vel / np.array(5) if scaled else \
                self.env.robot.left_foot_lin_vel
            output["right_lin"] = self.env.robot.right_foot_lin_vel / np.array(5) if scaled else \
                self.env.robot.right_foot_lin_vel

            output["left_ang"] = self.env.robot.left_foot_ang_vel / np.array(4 * math.tau) if scaled else \
                self.env.robot.right_foot_ang_vel
            output["right_ang"] = self.env.robot.right_foot_ang_vel / np.array(4 * math.tau) if scaled else \
                self.env.robot.right_foot_ang_vel

        if self.env.rot_type == Rot.RPY:
            left_rot = quat2euler(self.env.robot.left_foot_quat)
            right_rot = quat2euler(self.env.robot.left_foot_quat)
            output["left_rot"] = left_rot / np.array(math.tau / 2) if scaled else left_rot
            output["right_rot"] = right_rot / np.array(math.tau / 2) if scaled else right_rot
        elif self.env.rot_type == Rot.FUSED:
            # without hemi
            left_rot = quat2fused(self.env.robot.left_foot_quat)[:3]
            right_rot = quat2fused(self.env.robot.left_foot_quat)[:3]
            output["left_rot"] = left_rot / np.array(math.tau / 2) if scaled else left_rot
            output["right_rot"] = right_rot / np.array(math.tau / 2) if scaled else right_rot
        elif self.env.rot_type == Rot.QUAT:
            left_rot = self.env.robot.left_foot_quat
            right_rot = self.env.robot.left_foot_quat
            output["left_rot"] = left_rot
            output["right_rot"] = right_rot
        elif self.env.rot_type == Rot.SIXD:
            left_rot = quat2sixd(self.env.robot.left_foot_quat)
            right_rot = quat2sixd(self.env.robot.left_foot_quat)
            output["left_rot"] = left_rot
            output["right_rot"] = right_rot

        # for key, value in output.items():
        #    if np.min(value) < -1 or np.max(value) > 1:
        #        print(f"Value for {key} was {value}")
        return output


class JointSpaceState(BaseState):
    def __init__(self, env: "DeepQuinticEnv", use_foot_sensors, leg_vel_in_state, randomize):
        super().__init__(env, use_foot_sensors, randomize)
        self.leg_vel_in_state = leg_vel_in_state

    def get_state_entries(self, scaled):
        output = super(JointSpaceState, self).get_state_entries(scaled)
        joint_positions, joint_velocities, joint_torrques = self.env.sim.get_joint_values(
            self.env.robot.used_joint_names, scaled, self.env.robot.robot_index)
        output["joint_positions"] = joint_positions
        if self.leg_vel_in_state:
            output["joint_velocities"] = joint_velocities
        for key, value in output.items():
            if np.min(value) < -1 or np.max(value) > 1:
                print(f"Value for {key} was {value}")
        return output
