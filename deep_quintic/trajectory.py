import json

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
import random

from moveit_msgs.srv import GetPositionFK
from tf_transformations import euler_from_quaternion, quaternion_from_euler

class Trajectory:

    def __init__(self):
        self.loop = False
        self.joint_names = None
        self.frames = []
        self.duration = 0
        self.current_frame_index = 0
        self.start_frame_index = 0
        self.start_frame = None
        self.loop_count = 0
        self.loop_translation = None

    def load_from_json(self, filepath):
        # load json
        with open(filepath, "r") as json_file:
            json_data = json.load(json_file)

        self.loop = json_data["Loop"]
        self.joint_names = json_data["joint_order"]
        jsons_frames = json_data["frames"]
        start_time = jsons_frames[0][0]
        for json_frame in jsons_frames:
            frame = Frame(time=json_frame[0] - start_time,
                          phase=json_frame[1],
                          position=[json_frame[2], json_frame[3], json_frame[4]],
                          orientation=[json_frame[5], json_frame[6], json_frame[7], json_frame[8]],
                          joint_names=self.joint_names,
                          joint_positions=json_frame[9:9 + len(self.joint_names)],
                          joint_velocities=json_frame[9 + len(self.joint_names):])
            self.frames.append(frame)
        self.duration = self.frames[-1].time - self.frames[0].time
        self.loop_translation = [a - b for a, b in
                                 zip(self.frames[-1].robot_world_position, self.frames[0].robot_world_position)]

    def reset(self):
        # reset to a random frame and remember it
        self.loop_count = 0
        self.set_frame_loop_positions()
        self.current_frame_index = int(random.uniform(0, len(self.frames) - 2))
        self.start_frame_index = self.current_frame_index
        self.start_frame = self.frames[self.current_frame_index]
        previous_frame = self.frames[self.current_frame_index - 1]
        return self.start_frame, previous_frame

    def get_current_frame(self):
        return self.frames[self.current_frame_index]

    def get_previous_frame(self):
        if self.current_frame_index == 0:
            # handle edge case depending on looping or not
            if self.loop:
                return self.frames[-1]
            else:
                return self.frames[0]
        else:
            return self.frames[self.current_frame_index - 1]

    def get_current_pose(self):
        current_frame = self.frames[self.current_frame_index]
        return current_frame.position, current_frame.pose

    def get_start_pose(self):
        return self.start_frame.position, self.start_frame.pose

    def step_to_time(self, time_since_start):
        """
        Steps the trajectory forward from current index until finding best match for the time already spend in trajectory.
        Will automatically loop if looping is true.

        @param time_since_start: seconds passed since beginning the trajectory
        @return: new current frame
        """
        # add time of starting frame and subtract the time spend in the previous loops
        time_in_this_loop = time_since_start + self.start_frame.time - self.duration * self.loop_count

        # iterate through the next frames to see which one is closest in time
        closest_time = 100000000
        closest_index = self.current_frame_index
        for i in range(self.current_frame_index, len(self.frames)):
            time_diff = abs(self.frames[i].time - time_in_this_loop)
            if time_diff < closest_time:
                # save better index
                closest_time = time_diff
                closest_index = i
            else:
                # if the step forward in trajectory didn't improve the result, we can stop searching
                # we are only going further away
                # the previous result was the correct one
                self.current_frame_index = closest_index
                return self.frames[closest_index]

        if self.loop:
            self.loop_count += 1
            self.set_frame_loop_positions()
            # if we are looping and we reached the end, we start searching from the beginning
            # subtract the time spend in last iteration
            time = time_in_this_loop - self.duration
            closest_time = 100000000
            closest_index = 0
            for i in range(0, len(self.frames)):
                time_diff = abs(self.frames[i].time - time)
                if time_diff < closest_time:
                    closest_time = time_diff
                    closest_index = i
                else:
                    self.current_frame_index = closest_index
                    return self.frames[closest_index]
        return None

    def set_frame_loop_positions(self):
        # set all frame positions as they are after the current number of loops
        for frame in self.frames:
            frame.robot_world_position = [a + b * self.loop_count for a, b in
                                          zip(frame.robot_original_position, self.loop_translation)]


class Frame:

    def __init__(self, time, phase, position, orientation, joint_names=[], joint_positions=[], joint_velocities=[],
                 left_foot=None, right_foot=None, robot_velocity=(0, 0, 0)):
        i = 0
        for name in joint_names:
            self.joint_positions[name] = joint_positions[i]
            # self.joint_velocities[name] = joint_velocities[i] #todo
            i += 1
        # poses of feet are relative to robot
        if left_foot is not None:
            self.left_foot_pos = np.array([left_foot.position.x, left_foot.position.y, left_foot.position.z])
            self.left_foot_rpy = euler_from_quaternion(
                [left_foot.orientation.x, left_foot.orientation.y, left_foot.orientation.z, left_foot.orientation.w])
        else:
            self.left_foot_pos = None
            self.left_foot_rpy = None
        if right_foot is not None:
            self.right_foot_pos = np.array([right_foot.position.x, right_foot.position.y, right_foot.position.z])
            self.right_foot_rpy = euler_from_quaternion([
                right_foot.orientation.x, right_foot.orientation.y, right_foot.orientation.z, right_foot.orientation.w])
        else:
            self.right_foot_pos = None
            self.right_foot_rpy = None

