from __future__ import absolute_import
from __future__ import division
import argparse
import math
import threading
from enum import Enum

import gym
import numpy as np
from geometry_msgs.msg import Point, Quaternion
from moveit_msgs.srv import GetPositionIK


from bitbots_moveit_bindings import get_position_ik, get_position_fk
from rclpy.duration import Duration
from bitbots_utils.transforms import wxyz2xyzw
from rclpy.time import Time


class Rot(Enum):
    # enum for different possible rotation representation
    RPY = 1
    FUSED = 2
    QUAT = 3
    SIXD = 4


def substract_tuples(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result

def compute_ik(left_foot_pos, left_foot_quat, right_foot_pos, right_foot_quat, used_joint_names, joint_indexes,
               collision=False, approximate=False):
    request = GetPositionIK.Request()
    request.ik_request.timeout = Duration(seconds=int(0.01), nanoseconds=0.01 % 1 * 1e9).to_msg()
    # request.ik_request.attempts = 1
    request.ik_request.avoid_collisions = collision

    request.ik_request.group_name = "LeftLeg"
    request.ik_request.pose_stamped.pose.position = Point(x=left_foot_pos[0], y=left_foot_pos[1], z=left_foot_pos[2])
    # quaternion needs to be in ros style xyzw
    quat = wxyz2xyzw(left_foot_quat)
    request.ik_request.pose_stamped.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    ik_result = get_position_ik(request, approximate=approximate)
    first_error_code = ik_result.error_code.val
    if right_foot_pos is not None and right_foot_quat is not None:
        # maybe we just want to solve the left leg
        request.ik_request.group_name = "RightLeg"
        request.ik_request.pose_stamped.pose.position = Point(x=right_foot_pos[0],y=right_foot_pos[1], z=right_foot_pos[2])
        quat = wxyz2xyzw(right_foot_quat)
        request.ik_request.pose_stamped.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        ik_result = get_position_ik(request, approximate=approximate)
    # check if no solution or collision
    error_codes = [-31]
    success = not (first_error_code in error_codes or ik_result.error_code.val in error_codes)
    # only return the leg joints in specified order
    joint_positions = np.empty(len(used_joint_names))
    for i in range(len(ik_result.solution.joint_state.name)):
        joint_name = ik_result.solution.joint_state.name[i]
        if joint_name in used_joint_names:
            index = joint_indexes[joint_name]
            joint_positions[index] = ik_result.solution.joint_state.position[i]
    if np.isnan(joint_positions).any():
        print(f"IK had NaN values.\n left quat {left_foot_quat}\n right quat {right_foot_quat}\n left pos {left_foot_pos}\n right pos {right_foot_pos}")
        print(f"ik result {ik_result}")
        success = False
    return joint_positions, success
