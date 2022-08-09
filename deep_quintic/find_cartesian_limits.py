import rclpy
from rclpy.node import Node

from ament_index_python import get_package_share_directory
from bitbots_moveit_bindings import set_moveit_parameters
from bitbots_moveit_bindings.libbitbots_moveit_bindings import initRos
from bitbots_utils.utils import load_moveit_parameter, get_parameters_from_ros_yaml
from numpy import random
from deep_quintic.utils import compute_ik
from transforms3d.euler import euler2quat
import math

robot_type = "mrl_hsl"
experiment_number = 1000
threshold = 0.0001
sample_bounds = (((-0.5, 0.5), (-0.5, 0.5), (-0.5, -0.0), 
                  (-math.tau / 8, math.tau / 8), (-math.tau / 8, math.tau / 8), (-math.tau / 8, math.tau / 8)))
sample_bounds = [(-0.12, 0.2), (0, 0.2), (-0.25, -0.15), (-math.tau / 12, math.tau / 12),
                            (-math.tau / 12, math.tau / 12), (-math.tau / 12, math.tau / 12)]

initRos()
rclpy.init()
node = Node("find_cartesian_limits")
moveit_parameters = load_moveit_parameter(robot_type)
set_moveit_parameters(moveit_parameters)

print(f"\nRandom sampling {experiment_number} Cartesian goal poses for left foot of robot {robot_type}.")
print(f"Sample space is {sample_bounds}")
print(f"Threshold is {threshold}")

results = []
cartesian_bounds = [[-math.inf,math.inf], [0,0], [0,0], [0,0], [0,0], [0,0]]

def print_results():
    global results, successes
    fraction = successes/ experiment_number
    print(f"A fraction of {fraction} of the poses in this space are reachable")

successes = 0

for i in range(experiment_number):
    target_pos = (random.uniform(*sample_bounds[0]),
                    random.uniform(*sample_bounds[1]),
                    random.uniform(*sample_bounds[2]))
    target_rpy = (random.uniform(*sample_bounds[3]),
                    random.uniform(*sample_bounds[4]),
                    random.uniform(*sample_bounds[5]))
    target_quat = euler2quat(*target_rpy, axes='sxyz')
                                                                    
    joint_results, success = compute_ik(target_pos, target_quat, None, None, [], [], collision=False, approximate=False)
    results.append((success, (*target_pos, *target_rpy)))    
    if success:
        successes += 1
print_results()