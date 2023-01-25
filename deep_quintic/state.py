import math
import random
from abc import ABC
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from typing import TYPE_CHECKING

from transforms3d.euler import euler2quat, quat2euler

from deep_quintic.utils import Rot, scale_joint_position
from bitbots_utils.transforms import wxyz2xyzw, quat2fused, quat2sixd

if TYPE_CHECKING:
    from deep_quintic import DeepQuinticEnv


class State:
    def __init__(self, env: "DeepQuinticEnv"):
        self.env = env
        self.debug_names = None
        self.debug_publishers = {}
        self.debug_publishers_unscaled = {}

    def publish_debug(self):
        entries_scaled = self.get_state_entries(True)
        entries_unscaled = self.get_state_entries(False)
        if len(self.debug_publishers.keys()) == 0:
            # initialize publishers
            for name in entries_scaled.keys():
                self.debug_publishers[name] = self.env.node.create_publisher(Float32MultiArray, "state_" + name, 1)
                self.debug_publishers_unscaled[name] = self.env.node.create_publisher(Float32MultiArray, "state_" + name + "_unscaled", 1)
        for entry_name in entries_scaled.keys():
            publisher = self.debug_publishers[entry_name]
            if publisher.get_subscription_count() > 0:
                publisher.publish(Float32MultiArray(data=entries_scaled[entry_name]))
            publisher_unscaled = self.debug_publishers_unscaled[entry_name]
            if publisher_unscaled.get_subscription_count() > 0:
                publisher_unscaled.publish(Float32MultiArray(data=entries_unscaled[entry_name]))

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
        # out of some reason we need to do deepcopy
        output["cmd_vel"] = deepcopy(self.env.current_command_speed)
        # scale
        if scaled:
            output["cmd_vel"][1] = output["cmd_vel"][1] * 2.0
            output["cmd_vel"][2] = output["cmd_vel"][2] / 4.0
        phase = deepcopy(self.env.refbot.phase)
        if self.env.phase_in_state:
            if self.env.cyclic_phase:
                output["phase"] = [math.sin(phase * math.tau),
                                   math.cos(phase * math.tau)]
            else:
                output["phase"] = [phase]
        return output


class BaseState(PhaseState):
    def __init__(self, env: "DeepQuinticEnv", use_foot_sensors, randomize):
        super().__init__(env)
        self.use_foot_sensors = use_foot_sensors
        self.randomize = randomize

    def get_state_entries(self, scaled):
        output = super(BaseState, self).get_state_entries(scaled)
        # it only makes sense to display IMU rot in fused or RPY since we do not have yaw direction
        if self.env.use_imu_orientation:
            rpy = deepcopy(self.env.robot.imu_rpy)
            if self.randomize and len(rpy) == 3:
                for i in range(3):
                    rpy[i] = random.gauss(rpy[i], 0.20)
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

        if self.env.use_gyro:
            ang_vel = deepcopy(self.env.robot.imu_ang_vel)
            if self.randomize:
                for i in range(3):
                    ang_vel[i] = random.gauss(ang_vel[i], 2.0)
            if scaled:
                ang_vel = ang_vel / np.array(20)
            output["ang_vel"] = ang_vel

        if self.use_foot_sensors != "":
            foot_pressures = deepcopy(self.get_pressure_array(self.use_foot_sensors))

            if self.use_foot_sensors == "binary":
                output["foot_pressure"] = []
                for pressure in foot_pressures:
                    output.append(pressure>0.0) #todo usage of a threshould would maybe make sens
            else:
                output["foot_pressure"] = foot_pressures / np.array(100) if scaled else foot_pressures

        if self.env.use_rt_in_state:
            if self.env.relative:
                print("not implemented")
                exit()
            else:
                if self.env.cartesian_action:
                    output["ref_action"] = self.env.robot.scale_pose_to_action(
                        deepcopy(self.env.refbot.next_left_foot_pos),
                        quat2euler(deepcopy(self.env.refbot.next_left_foot_quat)),
                        deepcopy(self.env.refbot.next_right_foot_pos),
                        quat2euler(deepcopy(self.env.refbot.next_right_foot_quat)),
                        deepcopy(self.env.rot_type))
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
        output["left_pos"] = deepcopy(self.env.robot.left_foot_pos)
        output["right_pos"] = deepcopy(self.env.robot.right_foot_pos)

        if self.leg_vel_in_state:
            output["left_lin"] = deepcopy(self.env.robot.left_foot_lin_vel) / np.array(5) if scaled else \
                deepcopy(self.env.robot.left_foot_lin_vel)
            output["right_lin"] = deepcopy(self.env.robot.right_foot_lin_vel) / np.array(5) if scaled else \
                deepcopy(self.env.robot.right_foot_lin_vel)

            output["left_ang"] = deepcopy(self.env.robot.left_foot_ang_vel) / np.array(4 * math.tau) if scaled else \
                deepcopy(self.env.robot.right_foot_ang_vel)
            output["right_ang"] = deepcopy(self.env.robot.right_foot_ang_vel) / np.array(4 * math.tau) if scaled else \
                deepcopy(self.env.robot.right_foot_ang_vel)

        if self.env.rot_type == Rot.RPY:
            left_rot = quat2euler(deepcopy(self.env.robot.left_foot_quat))
            right_rot = quat2euler(deepcopy(self.env.robot.left_foot_quat))
            output["left_rot"] = left_rot / np.array(math.tau / 4) if scaled else left_rot
            output["right_rot"] = right_rot / np.array(math.tau / 4) if scaled else right_rot
        elif self.env.rot_type == Rot.FUSED:
            # without hemi
            left_rot = quat2fused(deepcopy(self.env.robot.left_foot_quat))[:3]
            right_rot = quat2fused(deepcopy(self.env.robot.left_foot_quat))[:3]
            output["left_rot"] = left_rot / np.array(math.tau / 4) if scaled else left_rot
            output["right_rot"] = right_rot / np.array(math.tau / 4) if scaled else right_rot
        elif self.env.rot_type == Rot.QUAT:
            left_rot = deepcopy(self.env.robot.left_foot_quat)
            right_rot = deepcopy(self.env.robot.left_foot_quat)
            output["left_rot"] = left_rot
            output["right_rot"] = right_rot
        elif self.env.rot_type == Rot.SIXD:
            left_rot = quat2sixd(deepcopy(self.env.robot.left_foot_quat))
            right_rot = quat2sixd(deepcopy(self.env.robot.left_foot_quat))
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
        if env.ros and env.robot_type != "Wolfang":
            print("no hardcoded joint scaling implemented for this robot")
            exit("")
        else:
            self.joint_scaling = {"LAnklePitch": (0.2618, 1.74533, -1.22173),
                                    "LAnkleRoll": (0.0, 1.0472, -1.0472),
                                    "LHipPitch": (0.08727, 2.0944, -1.91986),
                                    "LHipRoll": (0.0, 1.5708, -1.5708),
                                    "LHipYaw": (0.0, 1.5708, -1.5708),
                                    "LKnee": (1.48353, 2.96706, 0.0),
                                    "RAnklePitch": (-0.2618, 1.22173, -1.74533),
                                    "RAnkleRoll": (0.0, 1.0472, -1.0472),
                                    "RHipPitch": (-0.08727, 1.91986, -2.0944),
                                    "RHipRoll": (0.0, 1.5708, -1.5708),
                                    "RHipYaw": (0.0, 1.5708, -1.5708),
                                    "RKnee": (-1.48353, 0.0, -2.96706)}


    def get_state_entries(self, scaled):
        output = super(JointSpaceState, self).get_state_entries(scaled)
        if not self.env.ros:
            # we are running in simulation, get the values
            joint_positions, joint_velocities, joint_torques = self.env.sim.get_joint_values(
                self.env.robot.used_joint_names, scaled, self.env.robot.robot_index)
            joint_positions = deepcopy(joint_positions)
            joint_velocities = deepcopy(joint_velocities)
        else:
            # values have been provided by ROS
            joint_positions = deepcopy(self.env.robot.joint_positions)
            joint_velocities = deepcopy(self.env.robot.joint_velocities)
        if scaled:
            i = 0
            positions = []
            velocities = []
            for joint_name in self.env.robot.used_joint_names:
                if not self.env.ros:
                    # get scaling from simulation
                    scaled_position = self.env.sim.convert_radiant_to_scaled(joint_name, joint_positions[i])
                else:
                    # use hardcoded scaling as we dont have a simulation
                    scaled_position = scale_joint_position(joint_positions[i], *self.joint_scaling[joint_name])
                positions.append(scaled_position)
                velocities.append(joint_velocities[i] / 10.0)
                i+=1
            joint_positions = positions
            joint_velocities = velocities
        output["joint_positions"] = joint_positions
        if self.leg_vel_in_state:
            output["joint_velocities"] = joint_velocities
        if scaled and False:
            for key, value in output.items():
                if np.min(value) < -1 or np.max(value) > 1:
                    print(f"Value for {key} was {value}")
        return output    