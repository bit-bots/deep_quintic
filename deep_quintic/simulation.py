import math
import os
from scipy import signal
import signal as sig
import subprocess
import time
from abc import ABC
import pybullet as p
import numpy as np
import transforms3d.axangles
import psutil

try:
    # this import is from webots not from the controller python package. if the controller package is installed it will also fail!
    from controller import Keyboard
except:
    print("Keyboard package from webots not found. Only pybullet sim will work")
from nav_msgs.msg import Odometry
from wolfgang_pybullet_sim.simulation import Simulation

from bitbots_utils.transforms import xyzw2wxyz, wxyz2xyzw
from ament_index_python import get_package_share_directory
from ros2param.api import load_parameter_file

try:
    from wolfgang_webots_sim.webots_robot_supervisor_controller import SupervisorController, RobotController
    from controller import Supervisor
except:
    print("Could not load webots sim. If you want to use it, source the setenvs.sh")


class AbstractSim:

    def __init__(self, node):
        self.node = node
        self.time_step = None

    def get_base_velocity(self, robot_index):
        raise NotImplementedError

    def get_base_position_and_orientation(self, robot_index):
        raise NotImplementedError

    def get_joint_values(self, used_joint_names, scaled=False, robot_index=1):
        raise NotImplementedError

    def get_link_values(self, link_name, robot_index):
        raise NotImplementedError

    def get_sensor_force(self, sensor_name, filtered, robot_index=1):
        raise NotImplementedError

    def set_joint_position(self, joint_name, position, scaled=False, relative=False, robot_index=1):
        raise NotImplementedError

    def set_alpha(self, alpha, robot_index=1):
        raise NotImplementedError

    def reset_joints_to_init_pos(self, robot_index=1):
        raise NotImplementedError

    def reset_base_position_and_orientation(self, pos, quat, robot_index=1):
        raise NotImplementedError

    def reset_base_velocity(self, lin_vel, ang_vel, robot_index=1):
        raise NotImplementedError

    def reset_joint_to_position(self, joint_name, pos_in_rad, velocity=0, robot_index=1):
        raise NotImplementedError

    def reset_pressure_filters(self, robot_index=1):
        raise NotImplementedError

    def apply_external_force_to_base(self, force, robot_index=1):
        raise NotImplementedError

    def apply_external_torque_to_base(self, torque, robot_index=1):
        raise NotImplementedError

    def convert_radiant_to_scaled(self, joint_name, radiant, robot_index=1):
        raise NotImplementedError

    def convert_scaled_to_radiant(self, joint_name, scaled, robot_index=1):
        raise NotImplementedError

    def randomize_links(self, mass_bounds, inertia_bounds, robot_index=1):
        raise NotImplementedError

    def randomize_joints(self, torque_bounds, vel_bounds, robot_index=1):
        raise NotImplementedError

    def randomize_foot_friction(self, restitution_bounds, lateral_friction_bounds, spinning_friction_bounds,
                                rolling_friction_bounds, robot_index=1):
        raise NotImplementedError

    def randomize_floor(self, restitution_bounds, lateral_friction_bounds, spinning_friction_bounds,
                        rolling_friction_bounds):
        raise NotImplementedError

    def randomize_terrain(self, terrain_height):
        raise NotImplementedError

    def read_command_vel_from_gui(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def step_pressure_filters(self, robot_index=1):
        raise NotImplementedError

    def handle_gui(self):
        raise NotImplementedError

    def get_alpha(self):
        raise NotImplementedError

    def is_fixed_position(self):
        raise NotImplementedError

    def get_render(self, render_width, render_height, camera_distance, camera_pitch, camera_yaw, robot_pos):
        raise NotImplementedError

    def set_initial_joint_positions(self, joint_position_dict):
        self.initial_joint_positions = joint_position_dict

    def add_robot(self, physics_active=True):
        raise NotImplementedError

    def get_imu_ang_vel(self):
        raise NotImplementedError

    def get_imu_lin_acc(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class PybulletSim(Simulation, AbstractSim):

    def __init__(self, node, gui, terrain_height, robot_type="wolfgang"):
        AbstractSim.__init__(self, node)
        # time step should be at 240Hz (due to pyBullet documentation)
        self.time_step = (1 / 240)

        self.node.declare_parameter("simulation_active", True)
        self.node.declare_parameter("contact_stiffness", 0.0)
        self.node.declare_parameter("joint_damping", 0.0)
        self.node.declare_parameter("spinning_friction", 0.0)
        self.node.declare_parameter("contact_damping", 0.0)
        self.node.declare_parameter("lateral_friction", 0.0)
        self.node.declare_parameter("rolling_friction", 0.0)
        self.node.declare_parameter("cutoff", 0)
        self.node.declare_parameter("order", 0)
        self.node.declare_parameter("restitution", 0.0)
        # load simulation params
        load_parameter_file(node=self.node, node_name=self.node.get_name(),
                            parameter_file=f'{get_package_share_directory("wolfgang_pybullet_sim")}/config/config.yaml',
                            use_wildcard=True)
        Simulation.__init__(self, gui, urdf_path=None, terrain_height=terrain_height, field=False, robot=robot_type,
                            load_robot=False)

        self.gui = gui
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self.debug_alpha_index = p.addUserDebugParameter("display reference", 0, 1, 0.5)
            self.debug_refbot_fix_position = p.addUserDebugParameter("fixed refbot position", 0, 1, 0)
            self.debug_random_vel = p.addUserDebugParameter("random vel", 0, 1, 0)
            self.debug_cmd_vel = [p.addUserDebugParameter("cmd vel x", -1, 1, 0.1),
                                  p.addUserDebugParameter("cmd vel y", -1, 1, 0.0),
                                  p.addUserDebugParameter("cmd vel yaw", -2, 2, 0.0)]

    def get_base_velocity(self, robot_index):
        return p.getBaseVelocity(robot_index)

    def get_base_position_and_orientation(self, robot_index):
        (x, y, z), (qx, qy, qz, qw) = Simulation.get_base_position_and_orientation(self, robot_index)
        pos_in_world = np.array([x, y, z])
        quat_in_world = xyzw2wxyz([qx, qy, qz, qw])
        return pos_in_world, quat_in_world

    def get_link_values(self, link_name, robot_index):
        pos_in_world, xyzw_in_world, lin_vel_in_world, ang_vel_in_world = Simulation.get_link_values(self, link_name,
                                                                                                     robot_index)
        return pos_in_world, xyzw2wxyz(xyzw_in_world), lin_vel_in_world, ang_vel_in_world

    def reset_base_position_and_orientation(self, pos, quat, robot_index=1):
        xyzw = wxyz2xyzw(quat)
        Simulation.reset_base_position_and_orientation(self, pos, xyzw, robot_index)

    def read_command_vel_from_gui(self):
        if self.gui and p.readUserDebugParameter(self.debug_random_vel) < 0.5:
            # manual setting parameters via gui for testing
            return [p.readUserDebugParameter(self.debug_cmd_vel[0]), p.readUserDebugParameter(self.debug_cmd_vel[1]),
                    p.readUserDebugParameter(self.debug_cmd_vel[2])]
        else:
            return None

    def get_alpha(self):
        return p.readUserDebugParameter(self.debug_alpha_index)

    def is_fixed_position(self):
        return p.readUserDebugParameter(self.debug_refbot_fix_position) >= 0.5

    def get_render(self, render_width, render_height, camera_distance, camera_pitch, camera_yaw, robot_pos):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=robot_pos,
                                                          distance=camera_distance,
                                                          yaw=camera_yaw,
                                                          pitch=camera_pitch, roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(render_width) / render_height, nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=render_width, height=render_height,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            viewMatrix=view_matrix, projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        pass


class WebotsSim(SupervisorController, AbstractSim):

    def __init__(self, node, gui, robot_type=None, world="", start_webots=False, fast_physics=False):
        AbstractSim.__init__(self, node)
        self.robot_type = robot_type
        self.alpha = 0.5
        self.fixed_position = False
        self.show_refbot = True
        self.free_camera_active = False
        self.command_vel = [0.1, 0, 0]
        self.velocity_warning_printed = False
        if world == "":
            world = f"deep_quintic_{robot_type}"

        if start_webots:
            # start webots
            path = get_package_share_directory("wolfgang_webots_sim")

            fast = ""
            if fast_physics:
                fast = "_fast"

            arguments = ["webots", #/usr/local/webots/bin/webots-bin
                         "--batch",
                         path + "/worlds/" + world + fast + ".wbt"]
            if not gui:
                arguments.append("--minimize")
                arguments.append("--no-rendering")
                arguments.append("--stdout")
                arguments.append("--stderr")
            self.sim_proc = subprocess.Popen(arguments)

            os.environ["WEBOTS_PID"] = str(self.sim_proc.pid)

        if gui or True: #this is a hack because fast mode has a bug https://github.com/cyberbotics/webots/issues/3504
            mode = 'normal'
        else:
            mode = 'fast'
        os.environ["WEBOTS_ROBOT_NAME"] = "lernbot"
        SupervisorController.__init__(self, ros_node=self.node, ros_active=False, mode=mode, base_ns='/',
                                      model_states_active=False, robot=self.robot_type)

        self.pressure_filters = {}
        self.time_step = self.world_info.getField("basicTimeStep").getSFFloat() / 1000.0
        # compute frequency based on timestep which is represented in ms
        self.simulator_freq = 1 / self.time_step

        # this is currently only a solution for this case and could be made more generic
        # webots does not allow to have more than one robot controller from a python process.
        # Therefore we only create a controller for the actual robot and handle the refbot differently by directly
        # accessing the nodes
        self.robot_controller = RobotController(ros_node=self.node, ros_active=False, robot=self.robot_type,
                                                robot_node=self.supervisor, base_ns='', recognize=False,
                                                camera_active=False)

        for sensor_name in self.robot_controller.pressure_sensor_names:
            self.pressure_filters[sensor_name] = WebotsPressureFilter(self.simulator_freq)

        self.refbot_node = self.robot_nodes["refbot"]

    def add_robot(self, physics_active=True):
        # robots are already added in __init__()
        if not physics_active:
            name = 'refbot'
        else:
            name = 'lernbot'
        return name

    def get_base_velocity(self, robot_index):
        if robot_index == "refbot":
            # refbot needs to be handled differently since it can not have a robot controller
            # currently not needed but catch this if this will be used later
            raise NotImplementedError
        lin_vel, ang_vel = self.get_robot_velocity(robot_index)
        return [lin_vel, ang_vel]

    def get_base_position_and_orientation(self, robot_index):
        if robot_index == "refbot":
            raise NotImplementedError
        pos, quat = self.get_robot_pose_quat(robot_index)
        quat = xyzw2wxyz(quat)
        return pos, quat

    def get_joint_values(self, used_joint_names, scaled=False, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        return self.robot_controller.get_joint_values(used_joint_names, scaled)

    def get_link_values(self, link_name, robot_index):
        if robot_index == "refbot":
            raise NotImplementedError
        pos, xyzw = self.get_link_pose(link_name, robot_index)
        lin_vel, ang_vel = self.get_link_velocities(link_name, robot_index)
        return pos, xyzw2wxyz(xyzw), lin_vel, ang_vel

    def get_sensor_force(self, sensor_name, filtered, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        unfiltered, filtered = self.pressure_filters[sensor_name.lower()].get_force()
        if filtered:
            return filtered
        else:
            return unfiltered

    def set_joint_position(self, joint_name, position, scaled=False, relative=False, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        self.robot_controller.set_joint_goal_position(joint_name, position, scaled=scaled)

    def set_alpha(self, alpha, robot_index=1):
        if robot_index != "refbot":
            raise NotImplementedError
        transparency_field = self.robot_nodes["refbot"].getField("transparency")
        # not all robot models can be set to transparent
        if transparency_field is not None:
            transparency_field.setSFFloat(alpha)

    def reset_joints_to_init_pos(self, robot_index=1):
        # to a webots based reset first, there could be an issue with one of the joints
        self.reset_robot_init(robot_index)
        for name in self.initial_joint_positions.keys():
            self.reset_joint_to_position(name, math.radians(self.initial_joint_positions[name]),
                                         velocity=0, robot_index=robot_index)

    def reset_base_position_and_orientation(self, pos, quat, robot_index=1):
        quat = wxyz2xyzw(quat)
        self.reset_robot_pose(pos, quat, name=robot_index)

    def reset_base_velocity(self, lin_vel, ang_vel, robot_index=1):
        self.robot_nodes[robot_index].setVelocity([*lin_vel, *ang_vel])

    def reset_joint_to_position(self, joint_name, pos_in_rad, velocity=0, robot_index=1):
        # we directly reset the joint position, not the PID controlled motor
        # get the joint from the defined robot by using the weird webots syntax
        # joint_name = self.robot_controller.external_motor_names_to_motor_names[joint_name]
        joint_node = self.joint_nodes[robot_index][joint_name]
        if joint_node is None:
            print(f"Joint {joint_name} not found in robot {robot_index}. Can not set position")
        else:
            joint_node.setJointPosition(pos_in_rad)
        if velocity != 0 and not self.velocity_warning_printed:
            print("Resetting a joint to a specific velocity is not possible in Webots. Will only set Position.")
            self.velocity_warning_printed = True

    def reset_pressure_filters(self, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        for filter in self.pressure_filters.values():
            filter.reset()

    def apply_external_force_to_base(self, force, robot_index=1):
        self.robot_nodes[robot_index].addForce(force, False)

    def apply_external_torque_to_base(self, torque, robot_index=1):
        self.robot_nodes[robot_index].addTorque(torque, False)

    def convert_radiant_to_scaled(self, joint_name, radiant, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        return self.robot_controller.convert_joint_radiant_to_scaled(joint_name, radiant)

    def convert_scaled_to_radiant(self, joint_name, scaled, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        return self.robot_controller.convert_scaled_to_radiant(joint_name, scaled)

    def randomize_links(self, mass_bounds, inertia_bounds, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        raise NotImplementedError  # todo

    def randomize_joints(self, torque_bounds, vel_bounds, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        raise NotImplementedError  # todo

    def randomize_foot_friction(self, restitution_bounds, lateral_friction_bounds, spinning_friction_bounds,
                                rolling_friction_bounds, robot_index=1):
        if robot_index == "refbot":
            raise NotImplementedError
        raise NotImplementedError  # todo

    def randomize_floor(self, restitution_bounds, lateral_friction_bounds, spinning_friction_bounds,
                        rolling_friction_bounds):
        raise NotImplementedError  # todo

    def randomize_terrain(self, terrain_height):
        raise NotImplementedError  # todo

    def read_command_vel_from_gui(self):
        return self.command_vel

    def step(self):
        SupervisorController.step(self)

    def step_pressure_filters(self, robot_index=1):
        i = 0
        for sensor_name in list(self.pressure_filters.keys()):
            self.pressure_filters[sensor_name].filter_step(self.robot_controller.pressure_sensors[i])
            i += 1

    def get_alpha(self):
        return self.alpha

    def is_fixed_position(self):
        return self.fixed_position

    def get_render(self, render_width, render_height, camera_distance, camera_pitch, camera_yaw, robot_pos):        
        if not self.free_camera_active:
            self.free_camera = self.robot_controller.robot_node.getDevice("free_camera")
            self.free_camera.enable(30)
            self.free_camera_active = True

        image = self.free_camera.getImage()
        if image is None:
            # first image is invalid return black image
            return np.zeros(shape=(600, 800, 3), dtype=np.uint8)
        # we get a byte array from the simulation. shape it to a np array that has shape (width, height, color)
        # reshape from one dimensional vector to image with 4 channels
        image_array = np.frombuffer(image, dtype=np.uint8)
        image_reshape = np.reshape(image_array, (self.free_camera.getHeight(), self.free_camera.getWidth(), 4))
        # cut last channel as alpha has no important value
        image_reshape = image_reshape[..., :3]
        # image from simulation is bgr, need rgb
        image_reshape_rgb = image_reshape[...,::-1]        
        return image_reshape_rgb


    def handle_gui(self):
        key = SupervisorController.handle_gui(self)
        if key == ord('T'):
            self.alpha -= 0.05
            self.alpha = max(0, self.alpha)
        elif key == Keyboard.SHIFT + ord('T'):
            self.alpha += 0.05
            self.alpha = min(1, self.alpha)
        elif key == ord('F'):
            self.fixed_position = not self.fixed_position
        elif key == ord('B'):
            if self.show_refbot:
                self.set_alpha(1, "refbot")
            else:
                self.set_alpha(0, "refbot")
            self.show_refbot = not self.show_refbot
        elif key == ord('W'):
            self.command_vel[0] += 0.1
        elif key == ord('S'):
            self.command_vel[0] -= 0.1
        elif key == ord('A'):
            self.command_vel[1] += 0.1
        elif key == ord('D'):
            self.command_vel[1] -= 0.1
        elif key == ord('Q'):
            self.command_vel[2] += 0.1
        elif key == ord('E'):
            self.command_vel[2] -= 0.1

        if key in [ord('W'), ord('A'), ord('S'), ord('D'), ord('Q'), ord('E')]:
            print(
                f"Command vel: x:{round(self.command_vel[0], 2)} y:{round(self.command_vel[1], 2)} yaw:{round(self.command_vel[2], 2)}")

    def get_imu_ang_vel(self):
        return self.robot_controller.gyro.getValues()

    def get_imu_lin_acc(self):
        return self.robot_controller.accel.getValues()

    def close(self):
        process = psutil.Process(self.sim_proc.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()


class WebotsPressureFilter:
    def __init__(self, simulation_freq, cutoff=10, order=5):
        nyq = simulation_freq * 0.5  # nyquist frequency from simulation frequency
        normalized_cutoff = cutoff / nyq  # cutoff freq in hz
        self.filter_b, self.filter_a = signal.butter(order, normalized_cutoff, btype='low')
        self.filter_state = None
        self.reset()
        self.unfiltered = 0
        self.filtered = [0]

    def reset(self):
        self.filter_state = signal.lfilter_zi(self.filter_b, self.filter_a)

    def filter_step(self, unfiltered):
        self.filtered, self.filter_state = signal.lfilter(self.filter_b, self.filter_a, [self.unfiltered],
                                                          zi=self.filter_state)

    def get_force(self):
        return max(self.unfiltered, 0), max(self.filtered[0], 0)
