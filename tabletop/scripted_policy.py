import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from tabletop.constants import SIM_TASK_CONFIGS
from tabletop.ee_sim_env import make_ee_sim_env
import random

import IPython
e = IPython.embed

class OneArmBasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        # t_frac = 1
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']

        if curr_quat.dot(next_quat) < 0.0:
            next_quat = - next_quat
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts, inst=False):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])

        self.step_count += 1
        if inst:
            return action_left, next_left_waypoint['instruction']
        return action_left

class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        # t_frac = 1
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']

        if curr_quat.dot(next_quat) < 0.0:
            next_quat = - next_quat
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class CleanPolicy(BasePolicy):
    def get_angle(self, vector):
        x, y = vector
        angle = np.arctan2(y, x)
        degrees = angle / np.pi * 180.0
        return degrees

    def norm_vector(self, vector, ret_norm=False):
        norm = np.linalg.norm(vector, axis=0)
        if ret_norm:
            return vector / norm, norm
        else:
            return vector / norm

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        obj_dict = ts_first.observation['obj_dict']
        # print(obj_dict)
        # make a order of plan
        plan = {}
        for arm, obj in obj_dict.items():
            sorted_objects = sorted(obj.items(), key=lambda x: x[1][1])
            plan[arm] = [ob for ob, _ in sorted_objects]
        # print(plan)
        ####################################### CONSTANTS ###################################################
        left_pos = np.array([-0.469, 0.5])
        right_pos = np.array([+0.469, 0.5])
        basket_pos_left = np.array([-0.10, 0.18])
        basket_pos_right = np.array([0.10, 0.18])
        radius = {
            'bowl' : 0.06,
            'bottle' : 0.03,
            'cup1' : 0.019,
            'cup2' : 0.025,
            'can' : 0,
        }
        distance_const = 200.0
        grasp_time = 10
        apporach_time = 30

        grasp_approach_height = 0.15

        grasp_can_height = 0.07
        grasp_cup_height = {'cup2' : 0.06, 'cup1': 0.04}
        grasp_height = 0.03
        grasp_bottle_height = 0.12

        pick_height = 0.2
        left_quat = Quaternion([1, 0, 0, 0])
        right_quat = Quaternion([1, 0, 0, 0]) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=180)
        maxtimestep = 1200
        ####################################### END OF CONSTANTS ###################################################
        
        # make a approach (5 stage) -> (pre approach, approach grasp, pick, up, move to basket, place)  
        ################### LEFT ARM PLAN #######################################################################    
        self.left_trajectory = [{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0, 'instruction':'clean the table'}]
        left_t = 0
        pos = left_pos
        for i, obj in enumerate(plan['left']):
            inst = f'pick up the {obj} and put it in the basket'
            if 'bottle' in obj:
                ######### pre approach ################
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                a, b = d

                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm
                r = da * radius['bottle']
                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj] - 4 * r, np.array([grasp_approach_height * 1.2])])
                quat = left_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=degrees) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += int(dt * 1.3)
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[:2] = pos[:2] + r * 5
                pos[2] = grasp_bottle_height
                left_t += apporach_time * 2
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time * 2
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
            elif 'cup' in obj:
                ######### pre approach ################
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize

                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm

                a, b = da
                if random.random() < 0.5:
                    r = np.array([-b, a]) * radius[obj]
                else:
                    r = np.array([b, -a]) * radius[obj]

                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj] + r, np.array([grasp_approach_height])])
                quat = left_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-degrees)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += dt
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[2] = grasp_cup_height[obj]
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######################################## 
            elif 'can' in obj:
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                a, b = d
                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm
                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj], np.array([grasp_approach_height])])
                quat = left_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=degrees)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += dt
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[2] = grasp_can_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time * 2
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######################################## 
            elif 'bowl' in obj:
                ######### pre approach ################
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                
                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm

                a, b = da
                if random.random() < 0.5:
                    r = np.array([-b, a]) * radius['bowl']
                    tilt = -10
                else:
                    r = np.array([b, -a]) * radius['bowl']
                    tilt = 10

                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj] + r, np.array([grasp_approach_height])])
                quat = left_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-degrees) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=tilt)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += dt
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[2] = grasp_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######################################## 


        self.left_trajectory.append({"t": maxtimestep, "xyz": pos.copy(), "quat": init_mocap_pose_left[3:], "gripper": 0, 'instruction':'stay'})
        ################### END OF LEFT ARM PLAN #######################################################################  

        ################### RIGHT ARM PLAN #######################################################################      
        self.right_trajectory = [{"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}]
        right_t = 0
        pos = right_pos
        for i, obj in enumerate(plan['right']):
            if 'bottle' in obj:
                ######### pre approach ################
                d = (obj_dict['right'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                a, b = d

                da = (obj_dict['right'][obj] - right_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm
                r = da * radius['bottle']
                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['right'][obj] - 4 * r, np.array([grasp_approach_height * 1.2])])
                quat = right_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=degrees) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['right'][obj] - basket_pos_right))
                # dt = int(distance_const * dnorm)
                right_t += int(dt * 1.3)
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### approach with picking ################
                pos[:2] = pos[:2] + r * 5
                pos[2] = grasp_bottle_height
                right_t += apporach_time * 2
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### pick ################
                right_t += grasp_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### up ################
                pos[2] = pick_height
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### move to basket ##############
                right_t += int(np.linalg.norm(basket_pos_right - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_right, np.array([pick_height])])
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######## Place #########################
                right_t += grasp_time * 2
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
            elif 'cup' in obj:
                ######### pre approach ################
                d = (obj_dict['right'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize

                da = (obj_dict['right'][obj] - right_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm

                a, b = da
                if random.random() < 0.5:
                    r = np.array([-b, a]) * radius[obj]
                else:
                    r = np.array([b, -a]) * radius[obj]

                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['right'][obj] + r, np.array([grasp_approach_height])])
                quat = right_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=degrees)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['right'][obj] - basket_pos_right))
                # dt = int(distance_const * dnorm)
                right_t += dt
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### approach with picking ################
                pos[2] = grasp_cup_height[obj]
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### pick ################
                right_t += grasp_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### up ################
                pos[2] = pick_height
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### move to basket ##############
                right_t += int(np.linalg.norm(basket_pos_right - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_right, np.array([pick_height])])
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######## Place #########################
                right_t += grasp_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######################################## 
            elif 'can' in obj:
                d = (obj_dict['right'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                a, b = d
                da = (obj_dict['right'][obj] - right_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm
                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['right'][obj], np.array([grasp_approach_height])])
                quat = right_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=degrees)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['right'][obj] - basket_pos_right))
                # dt = int(distance_const * dnorm)
                right_t += dt
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### approach with picking ################
                pos[2] = grasp_can_height
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### pick ################
                right_t += grasp_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### up ################
                pos[2] = pick_height
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### move to basket ##############
                right_t += int(np.linalg.norm(basket_pos_right - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_right, np.array([pick_height])])
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######## Place #########################
                right_t += grasp_time * 2
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######################################## 
            elif 'bowl' in obj:
                ######### pre approach ################
                d = (obj_dict['right'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                
                da = (obj_dict['right'][obj] - right_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm

                a, b = da
                if random.random() < 0.5:
                    r = np.array([-b, a]) * radius['bowl']
                    tilt = -10
                else:
                    r = np.array([b, -a]) * radius['bowl']
                    tilt = 10

                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['right'][obj] + r, np.array([grasp_approach_height])])
                quat = right_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=degrees) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-tilt)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['right'][obj] - basket_pos_right))
                # dt = int(distance_const * dnorm)
                right_t += dt
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### approach with picking ################
                pos[2] = grasp_height
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######### pick ################
                right_t += grasp_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### up ################
                pos[2] = pick_height
                right_t += apporach_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######### move to basket ##############
                right_t += int(np.linalg.norm(basket_pos_right - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_right, np.array([pick_height])])
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0})
                ######## Place #########################
                right_t += grasp_time
                self.right_trajectory.append({'t':right_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1})
                ######################################## 


        self.right_trajectory.append({"t": maxtimestep, "xyz": pos.copy(), "quat": init_mocap_pose_right[3:], "gripper": 0})
        ################### END OF RIGHT ARM PLAN #######################################################################
        for p in self.left_trajectory:
            print(p)

        for p in self.right_trajectory:
            print(p)

class OneArmCleanPolicy(OneArmBasePolicy):
    def get_angle(self, vector):
        x, y = vector
        angle = np.arctan2(y, x)
        degrees = angle / np.pi * 180.0
        return degrees

    def norm_vector(self, vector, ret_norm=False):
        norm = np.linalg.norm(vector, axis=0)
        if ret_norm:
            return vector / norm, norm
        else:
            return vector / norm

    def generate_trajectory(self, ts_first):
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        obj_dict = ts_first.observation['obj_dict']
        # print(obj_dict)
        # make a order of plan
        plan = {}
        for arm, obj in obj_dict.items():
            sorted_objects = sorted(obj.items(), key=lambda x: x[1][1])
            plan[arm] = [ob for ob, _ in sorted_objects]
        import random
        numbers = list(range(4))  # [0, 1, 2, 3]
        # 리스트를 랜덤하게 섞기
        random.shuffle(numbers)
        # print(plan)
        ####################################### CONSTANTS ###################################################
        left_pos = np.array([-0.469, 0.5])
        basket_pos_left = np.array([-0.0, 0.18])
        radius = {
            'bowl' : 0.06,
            'cup' : 0.025,
            'red can' : 0,
            'green can' : 0,
        }
        distance_const = 200.0
        grasp_time = 10
        apporach_time = 30

        grasp_approach_height = 0.15

        grasp_can_height = 0.07
        grasp_cup_height = {'cup': 0.04}
        grasp_height = 0.03
        grasp_bottle_height = 0.12

        pick_height = 0.2
        left_quat = Quaternion([1, 0, 0, 0])
        # right_quat = Quaternion([1, 0, 0, 0]) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=180)
        maxtimestep = 950
        ####################################### END OF CONSTANTS ###################################################
        
        # make a approach (5 stage) -> (pre approach, approach grasp, pick, up, move to basket, place)  
        ################### LEFT ARM PLAN #######################################################################    
        self.left_trajectory = [{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0, 'instruction':'clean the table'}]
        left_t = 0
        pos = left_pos
        for i, obj_idx in enumerate(numbers):
            obj = plan['left'][obj_idx]
            inst = f'pick up the {obj} and put it in the basket'
            if 'bottle' in obj:
                ######### pre approach ################
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                a, b = d

                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm
                r = da * radius['bottle']
                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj] - 4 * r, np.array([grasp_approach_height * 1.2])])
                quat = left_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=degrees) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += int(dt * 1.3)
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[:2] = pos[:2] + r * 5
                pos[2] = grasp_bottle_height
                left_t += apporach_time * 2
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time * 2
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
            elif 'cup' in obj:
                ######### pre approach ################
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize

                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm

                a, b = da
                if random.random() < 0.5:
                    r = np.array([-b, a]) * radius[obj]
                else:
                    r = np.array([b, -a]) * radius[obj]

                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj] + r, np.array([grasp_approach_height])])
                quat = left_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-degrees)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += dt
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[2] = grasp_cup_height[obj]
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######################################## 
            elif 'can' in obj:
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                a, b = d
                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm
                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj], np.array([grasp_approach_height])])
                quat = left_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=degrees)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += dt
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[2] = grasp_can_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time * 2
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######################################## 
            elif 'bowl' in obj:
                ######### pre approach ################
                d = (obj_dict['left'][obj] - pos[:2])
                dnorm = np.linalg.norm(d, axis=0)
                d = d / dnorm # normalize
                
                da = (obj_dict['left'][obj] - left_pos[:2])
                danorm = np.linalg.norm(da, axis=0)
                da = da / danorm

                a, b = da
                if random.random() < 0.5:
                    r = np.array([-b, a]) * radius['bowl']
                    tilt = -10
                else:
                    r = np.array([b, -a]) * radius['bowl']
                    tilt = 10

                degrees = self.get_angle(da)
                pos = np.concatenate([obj_dict['left'][obj] + r, np.array([grasp_approach_height])])
                quat = left_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-degrees) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=tilt)
                dt = int(distance_const * dnorm) if i == 0 else int(distance_const * np.linalg.norm(obj_dict['left'][obj] - basket_pos_left))
                # dt = int(distance_const * dnorm)
                left_t += dt
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### approach with picking ################
                pos[2] = grasp_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######### pick ################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### up ################
                pos[2] = pick_height
                left_t += apporach_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######### move to basket ##############
                left_t += int(np.linalg.norm(basket_pos_left - pos[:2]) * distance_const)
                pos = np.concatenate([basket_pos_left, np.array([pick_height])])
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':0, 'instruction':inst})
                ######## Place #########################
                left_t += grasp_time
                self.left_trajectory.append({'t':left_t, 'xyz':pos.copy(), 'quat':quat.elements, 'gripper':1, 'instruction':inst})
                ######################################## 


        self.left_trajectory.append({"t": maxtimestep, "xyz": pos.copy(), "quat": init_mocap_pose_left[3:], "gripper": 0, 'instruction':'stay'})
        ################### END OF LEFT ARM PLAN #######################################################################  
        for p in self.left_trajectory:
            print(p)

class PickAndTransferPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

