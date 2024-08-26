import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import random

from pyquaternion import Quaternion
from tabletop.constants import DT, XML_DIR, START_ARM_POSE, ONEARM_START_ARM_POSE
from tabletop.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from tabletop.constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from tabletop.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from tabletop.constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

import math
from scipy.spatial.transform import Rotation

def quat_to_rpy(w, x, y, z, mode='abs'):
    try:
        r = Rotation.from_quat([w, x, y, z], scalar_first=True)
    except:
        return np.array([0.0, 0.0, 0.0])
    return np.array(r.as_euler('zyx', degrees=False))

def rpy_to_quat(roll, pitch, yaw):
    try:
        r = Rotation.from_euler('zyx', [roll, pitch, yaw], degrees=False)
    except:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array(r.as_quat(scalar_first=True))

def get_ee_vel_wrapper(target_class):
    class ee_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)
            self.prev_action[:7] += action[:7]
            self.prev_action[7] = action[7]
            self.prev_action[8:-1] += action[8:-1]
            self.prev_action[-1] = action[-1]
            super().before_step(self.prev_action, physics)

    return ee_vel_class(False)

def get_onearm_ee_vel_wrapper(target_class):
    class ee_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)
            self.prev_action[:7] += action[:7]
            self.prev_action[7] = action[7]
            super().before_step(self.prev_action, physics)

    return ee_vel_class(False)


def get_onearm_ee_rpy_vel_wrapper(target_class):
    class ee_rpy_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)

            self.prev_action[:3] += action[:3]
            self.prev_action[7] = action[6]
            delta_quat = Quaternion(rpy_to_quat(*action[3:6]))
            curr_quat = Quaternion(self.prev_action[3:7])
            next_quat = (curr_quat * delta_quat).elements
            self.prev_action[3:7] = next_quat
            
            super().before_step(self.prev_action, physics)
    return ee_rpy_vel_class(False)

def get_onearm_ee_rpy_pos_wrapper(target_class):
    class ee_rpy_pos_class(target_class):
        def before_step(self, action, physics):
            curr_ee = self.get_eepos(physics)
            curr_quat = curr_ee[3:7]
            
            ee_pos_raw = action[:3]
            ee_rpy_raw = action[3:6]
            grp = action[6]
            
            ee_quat_raw = rpy_to_quat(ee_rpy_raw[0], ee_rpy_raw[1], ee_rpy_raw[2])
            
            real_action = np.concatenate([ee_pos_raw, ee_quat_raw, [grp]], axis=0)
            
            super().before_step(real_action, physics)
    return ee_rpy_pos_class(False)

def get_joint_vel_wrapper(target_class):
    class joint_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
            
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_qpos(physics)
            self.prev_action = self.get_qpos(physics)
            self.prev_action[:6] += action[:6]
            self.prev_action[:7:-1] += action[7:-1]
            self.prev_action[6] = action[6]
            self.prev_action[-1] = action[-1]
            
            super().before_step(self.prev_action, physics)

    return joint_vel_class(False)

def get_onearm_joint_vel_wrapper(target_class):
    class joint_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
            
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_qpos(physics)
            self.prev_action[:6] += action[:6]
            self.prev_action[6] = action[6]
            super().before_step(self.prev_action, physics)

    return joint_vel_class(False)
            
            
            