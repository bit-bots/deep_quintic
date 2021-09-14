from __future__ import absolute_import
from __future__ import division
import argparse
import math
import threading
from enum import Enum

import gym
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from moveit_msgs.srv import GetPositionIKRequest, GetPositionIKResponse, GetPositionFKRequest
from stable_baselines3.common.vec_env import DummyVecEnv

import functools
import inspect
import pybullet

from bitbots_moveit_bindings import get_position_ik, get_position_fk
from tf.transformations import quaternion_from_euler
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import rotate_vector, qinverse, quat2mat, mat2quat


class Rot(Enum):
    # enum for different possible rotation representation
    RPY = 1
    FUSED = 2
    QUAT = 3
    SIXD = 4


def wxyz2xyzw(quat_wxyz):
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def xyzw2wxyz(quat_xyzw):
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def substract_tuples(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result


def compute_imu_orientation_from_world(robot_quat_in_world):
    # imu orientation has roll and pitch relative to gravity vector. yaw in world frame
    # get global yaw
    yrp_world_frame = quat2euler(robot_quat_in_world, axes='szxy')
    # remove global yaw rotation from roll and pitch
    yaw_quat = euler2quat(yrp_world_frame[0], 0, 0, axes='szxy')
    rp = rotate_vector((yrp_world_frame[1], yrp_world_frame[2], 0), qinverse(yaw_quat))
    # save in correct order
    return [rp[0], rp[1], 0], yaw_quat


def compute_ik(left_foot_pos, left_foot_quat, right_foot_pos, right_foot_quat, used_joint_names, joint_indexes,
               collision=False, approximate=False):
    request = GetPositionIKRequest()
    request.ik_request.timeout = rospy.Time.from_seconds(0.01)
    # request.ik_request.attempts = 1
    request.ik_request.avoid_collisions = collision

    request.ik_request.group_name = "LeftLeg"
    request.ik_request.pose_stamped.pose.position = Point(*left_foot_pos)
    # quaternion needs to be in ros style xyzw
    request.ik_request.pose_stamped.pose.orientation = Quaternion(*wxyz2xyzw(left_foot_quat))
    ik_result = get_position_ik(request, approximate=approximate)
    first_error_code = ik_result.error_code.val
    request.ik_request.group_name = "RightLeg"
    request.ik_request.pose_stamped.pose.position = Point(*right_foot_pos)
    request.ik_request.pose_stamped.pose.orientation = Quaternion(*wxyz2xyzw(right_foot_quat))
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
    return joint_positions, success


class BulletClient(object):
    """A wrapper for pybullet to manage different clients.
    from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/bullet_client.py"""

    def __init__(self, connection_mode=None):
        """Creates a Bullet client and connects to a simulation.
        Args:
          connection_mode:
            `None` connects to an existing simulation or, if fails, creates a
              new headless simulation,
            `pybullet.GUI` creates a new simulation with a GUI,
            `pybullet.DIRECT` creates a headless simulation,
            `pybullet.SHARED_MEMORY` connects to an existing simulation.
        """
        self._shapes = {}
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT
        self._client = pybullet.connect(connection_mode)

    def __del__(self):
        """Clean up connection if not already done."""
        if self._client >= 0:
            try:
                pybullet.disconnect(physicsClientId=self._client)
                self._client = -1
            except pybullet.error:
                pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        if name == "disconnect":
            self._client = -1
        return attribute


def quat2sixd(quat_wxyz):
    # see https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Zhou_On_the_Continuity_CVPR_2019_supplemental.pdf
    # first get matrix
    m = quat2mat(quat_wxyz)
    # 6D represenation is first 2 coloumns of matrix
    return [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1]]


def sixd2quat(sixd):
    # see https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Zhou_On_the_Continuity_CVPR_2019_supplemental.pdf
    # compute the three column vectors
    a_1 = sixd[:3]
    a_2 = sixd[3:]
    b_1 = a_1 / np.linalg.norm(a_1)
    b_2_no_norm = a_2 - (np.dot(b_1, a_2) * b_1)
    b_2 = b_2_no_norm / np.linalg.norm(b_2_no_norm)
    b_3 = np.cross(b_1, b_2)
    # create matrix from column vectors
    mat = np.stack((b_1, b_2, b_3), axis=-1)
    return mat2quat(mat)


def quat2fused(q):
    q_xyzw = xyzw2wxyz(q)
    # Fused yaw of Quaternion
    fused_yaw = 2.0 * math.atan2(q_xyzw[2],
                                 q_xyzw[3])  # Output of atan2 is [-tau/2,tau/2], so this expression is in [-tau,tau]
    if fused_yaw > math.tau / 2:
        fused_yaw -= math.tau  # fused_yaw is now in[-2* pi, pi]
    if fused_yaw <= -math.tau / 2:
        fused_yaw += math.tau  # fused_yaw is now in (-pi, pi]

    # Calculate the fused pitch and roll
    stheta = 2.0 * (q_xyzw[1] * q_xyzw[3] - q_xyzw[0] * q_xyzw[2])
    sphi = 2.0 * (q_xyzw[1] * q_xyzw[2] + q_xyzw[0] * q_xyzw[3])
    if stheta >= 1.0:  # Coerce stheta to[-1, 1]
        stheta = 1.0
    elif stheta <= -1.0:
        stheta = -1.0
    if sphi >= 1.0:  # Coerce sphi to[-1, 1]
        sphi = 1.0
    elif sphi <= -1.0:
        sphi = -1.0
    fused_pitch = math.asin(stheta)
    fused_roll = math.asin(sphi)

    # compute hemi parameter
    hemi = (0.5 - (q_xyzw[0] * q_xyzw[0] + q_xyzw[1] * q_xyzw[1]) >= 0.0)
    return fused_roll, fused_pitch, fused_yaw, hemi


# Conversion: Fused angles (3D/4D) --> Quaternion
def fused2quat(fusedRoll, fusedPitch, fusedYaw, hemi):
    # Precalculate the sine values
    sth = math.sin(fusedPitch)
    sphi = math.sin(fusedRoll)

    # Calculate the sine sum criterion
    crit = sth * sth + sphi * sphi

    # Calculate the tilt angle alpha
    if crit >= 1.0:
        alpha = math.pi / 2
    else:
        if hemi:
            alpha = math.acos(math.sqrt(1.0 - crit))
        else:
            alpha = math.acos(-math.sqrt(1.0 - crit))

    # Calculate the tilt axis angle gamma
    gamma = math.atan2(sth, sphi)

    # Evaluate the required intermediate angles
    halpha = 0.5 * alpha
    hpsi = 0.5 * fusedYaw
    hgampsi = gamma + hpsi

    # Precalculate trigonometric terms involved in the quaternion expression
    chalpha = math.cos(halpha)
    shalpha = math.sin(halpha)
    chpsi = math.cos(hpsi)
    shpsi = math.sin(hpsi)
    chgampsi = math.cos(hgampsi)
    shgampsi = math.sin(hgampsi)

    # Calculate and return the required quaternion
    return np.array([chalpha * chpsi, shalpha * chgampsi, shalpha * shgampsi, chalpha * shpsi])  # Order: (w,x,y,z)
