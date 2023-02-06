# a modified Walker2DBullet env to test the initial network bias

import numpy as np
import pandas as pd
import pybullet
import pybullet_envs.robot_bases
import sys
from time import sleep
from time import time
# hacky solution to fix the relative imports inside pybullet
sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages/pybullet_envs/")
sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages/")
from pybullet_envs.robot_locomotors import WalkerBase
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv


class Walker2DBulletBiasEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False, bias_pose="bend", control_type="position", bias_type="network", realtime=False):
    self.robot = Walker2DBiased(bias_pose=bias_pose, control_type=control_type, bias_type=bias_type, realtime=realtime)
    self.bias_pose = bias_pose
    self.control_type = control_type
    self.bias_type = bias_type    
    WalkerBaseBulletEnv.__init__(self, self.robot, render)    

  def get_action_biases(self):
    if self.bias_type == "network":
      return self.robot.get_action_biases()
    else:
      return [0] * 6

class Walker2DBiased(WalkerBase):
  foot_list = ["foot", "foot_left"]

  def __init__(self, bias_pose="bend", control_type="position", bias_type="network", realtime=False):
    WalkerBase.__init__(self, "walker2d.xml", "torso", action_dim=6, obs_dim=22, power=0.40)
    self.bias_pose = bias_pose
    self.control_type = control_type
    self.bias_type = bias_type
    self.realtime = realtime
    self.timestep = 1 / 240
    self.last_wall_time = time()
    self.applied_torques = []    

  def alive_bonus(self, z, pitch):
    return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    for n in ["foot_joint", "foot_left_joint"]:
      self.jdict[n].power_coef = 30.0
    if self.applied_torques != []:
      df = pd.DataFrame(self.applied_torques, columns =['0', '1', '2', '3', '4', '5'])
      df.to_csv("torques.csv")
      exit(0)

  #name: thigh_joint lower: -2.6179940700531006, upper: 0.0
  #name: leg_joint lower: -2.6179940700531006, upper: 0.0
  #name: foot_joint lower: -0.7853981852531433, upper: 0.7853981852531433
  #name: thigh_left_joint lower: -2.6179940700531006, upper: 0.0
  #name: leg_left_joint lower: -2.6179940700531006, upper: 0.0
  #name: foot_left_joint lower: -0.7853981852531433, upper: 0.7853981852531433
  
  def apply_action(self, action):
    if self.control_type == "position":
      self.apply_action_position(action)
    elif self.control_type == "torque":
      self.apply_action_torque(action)
    else:
      print(f"control type {self.control_type} not defined")
      exit(1)

    if self.realtime:
      # wait till one timestep has actually passed in real time
      time_to_sleep = max(0, self.timestep - (time() - self.last_wall_time)) * 2
      sleep(time_to_sleep)
    self.last_wall_time = time()

  def apply_action_position(self, action):
    # some stuff for testing    
    # normal init exploration
    if False:
      action = [0]*6
    # stable standing straight legs
    if False:
      action_rad = [0] * 6
      action = []
      for n, j in enumerate(self.ordered_joints):
        action.append(2 * (action_rad[n] - (0.5 * (j.lowerLimit + j.upperLimit))) / (j.upperLimit - j.lowerLimit))      
    # stable standing bend legs
    if False:
      action_rad = [-0.4, -0.6, 0.4, -0.4, -0.6, 0.4]
      action = []
      for n, j in enumerate(self.ordered_joints):
        action.append(2 * (action_rad[n] - (0.5 * (j.lowerLimit + j.upperLimit))) / (j.upperLimit - j.lowerLimit))         

    # add bias if required
    if self.bias_type == "add":
      if self.bias_pose == "straight":
        pose = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
      elif self.bias_pose == "bend":
        pose = [0.6944225316813748, 0.5416337975220622, 0.5092958037216182, 0.6944225316813748, 0.5416337975220622, 0.5092958037216182]
      for i in range(6):
        action[i] = min(max(action[i] + pose[i], -1), 1)

    #scale from [-1,1] to rad
    a = []
    for n, j in enumerate(self.ordered_joints):
      a.append(action[n] * (j.upperLimit - j.lowerLimit) / 2 + (0.5 * (j.lowerLimit + j.upperLimit)))        
    
    #print(a)
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_position(a[n])


  def apply_action_torque(self, a):    
    assert (np.isfinite(a).all())
    torque = []    
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
      torque.append(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))      
        
    # for recording torque
    if False:
      self.applied_torques.append(torque)

  def get_action_biases(self):    
    if self.control_type == "position":
      if self.bias_pose == "straight":
        # stable standing straight legs
        action_mu = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
      elif self.bias_pose == "bend":
        # stable bend legs
        action_mu = [0.6944225316813748, 0.5416337975220622, 0.5092958037216182, 0.6944225316813748, 0.5416337975220622, 0.5092958037216182]
      else:
        print(f"bias pose {self.bias_pose} not defined")
        exit(1)      
    elif self.control_type == "torque":
      if self.bias_pose == "bend":        
        mean_torques = [6.87910984210465, 5.703097397649507, -1.5885176309726174, 6.780863944953735, 7.256871279170993, -1.8898604377215609]
        #precomputed normalized mean torques
        action_mu = [0.17197774605261623, 0.14257743494123767, -0.13237646924771812, 0.16952159862384336, 0.18142178197927483, -0.15748836981013006]        
      else:
        print(f"bias pose {self.bias_pose} not defined for torque control")
        exit(1)
    return action_mu
