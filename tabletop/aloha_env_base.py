import numpy as np
import collections
from tabletop.constants import *
from tabletop.utils import *
from tabletop.wrappers import GSOWrapper
from tabletop.aloha_ik import AlohaIK
from dm_control.suite import base
from scipy.spatial.transform import Rotation

class AlohaTask(base.Task):
    def __init__(self, random=None, single_arm=False, single_arm_dir=None):
        self.obj_dict = {}
        self.obj_id_counter = 0
        self.time_limit = 20
        self.reward = 0
        self.reward_counter = 0
        self.max_reward = 1
        self.single_arm = single_arm
        self.single_arm_dir = single_arm_dir
        assert (self.single_arm and single_arm_dir in ['right', 'left']) or not self.single_arm, 'Wrong single arm direction'
        self.robot_offset = 8 if self.single_arm else 16
        self.aloha_ik = AlohaIK()
        self.use_joint_vel_ctrl = False
        self.instruction = ''
        super().__init__(random=random)

    def add_object(self, nick, name, pos=[0, 0, 0.02], rpy=[0, 0, 0], scale=[1, 1, 1], mass=1.0, inertial=[0, 0, 0]):
        r = Rotation.from_euler('zyx', rpy, degrees=True)
        quat = np.array(r.as_quat())
        obj = GSOWrapper(name, pos, quat, scale, mass, self.obj_id_counter, inertial=inertial)        
        self.obj_dict[nick] = obj
        self.obj_id_counter += 1

    def before_step(self, action, physics):
        if self.single_arm:
            if self.action_space == 'ee_quat_pos':
                g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
                qpos = self.aloha_ik.get_joint_pos(
                    target_pos=action[0:3],
                    target_quat=action[3:7],
                    curr_qpos=self.get_qpos(physics)[:-1],
                )
                np.copyto(physics.data.ctrl, np.concatenate([qpos, [g_right_ctrl]]))
            elif self.action_space == 'ee_6d_pos':
                g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
                target_quat = sixd_to_quat(action[3:9])
                qpos = self.aloha_ik.get_joint_pos(
                    target_pos=action[0:3],
                    target_quat=target_quat,
                    curr_qpos=self.get_qpos(physics)[:-1],
                )
                np.copyto(physics.data.ctrl, np.concatenate([qpos, [g_right_ctrl]]))
            else:
                g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
                np.copyto(physics.data.ctrl, np.concatenate([action[:6], [g_right_ctrl]]))
        else:
            if self.action_space == 'ee_quat_pos':
                g_left_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[7])
                g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
                qpos_left = self.aloha_ik.get_joint_pos(
                    target_pos=action[0:3],
                    target_quat=action[3:7],
                    curr_qpos=self.get_qpos(physics)[:6],
                )
                qpos_right = self.aloha_ik.get_joint_pos(
                    target_pos=action[8:11],
                    target_quat=action[11:-1],
                    curr_qpos=self.get_qpos(physics)[7:13],
                )
                np.copyto(physics.data.ctrl, np.concatenate([qpos_left, [g_left_ctrl], qpos_right, [g_right_ctrl]]))
            elif self.action_space == 'ee_6d_pos':
                g_left_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[9])
                g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
                target_quat_left = sixd_to_quat(action[3:9])
                target_quat_right = sixd_to_quat(action[13:19])
                qpos_left = self.aloha_ik.get_joint_pos(
                    target_pos=action[0:3],
                    target_quat=target_quat_left,
                    curr_qpos=self.get_qpos(physics)[:6],
                )
                qpos_right = self.aloha_ik.get_joint_pos(
                    target_pos=action[10:13],
                    target_quat=target_quat_right,
                    curr_qpos=self.get_qpos(physics)[7:13],
                )
                np.copyto(physics.data.ctrl, np.concatenate([qpos_left, [g_left_ctrl], qpos_right, [g_right_ctrl]]))
            else:
                g_left_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[6])
                g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
                np.copyto(physics.data.ctrl, np.concatenate([action[:6], [g_left_ctrl], action[7:-1], [g_right_ctrl]]))

    def after_step(self, physics):
        self.update_contact(physics)

    def initialize_robots(self, physics):
        if self.single_arm:
            np.copyto(physics.data.qpos[:self.robot_offset], np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084]))
        else:
            np.copyto(physics.data.qpos[:self.robot_offset], np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084, 0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084]))

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        self.reward = 0
        self.reward_counter = 0
        super().initialize_episode(physics)

    def set_object_pose(self, physics, nick, pos=None, rpy=None):
        obj_id = self.obj_dict[nick].id
        joint_id = self.robot_offset + 7 * obj_id
        np.copyto(physics.data.qpos[joint_id:joint_id + 7], np.concatenate([pos, rpy_to_quat(*rpy)]))

    def get_object_pose(self, physics, nick):
        obj_id = self.obj_dict[nick].id
        joint_id = self.robot_offset + 7 * obj_id
        pos = physics.data.qpos[joint_id : joint_id + 3].copy()
        quat = physics.data.qpos[joint_id + 3 : joint_id + 7].copy()
        return pos, quat

    def get_relative_pose(self, physics, target_site, reference_site):
        target_site_name = physics.model.site(target_site).id
        reference_site_name = physics.model.site(reference_site).id
        pos_ref = physics.named.data.site_xpos[reference_site_name].copy()

        rot_ref = physics.named.data.site_xmat[reference_site_name].copy().reshape(3, 3)
        rot_ref_inv = rot_ref.T

        # Get world pose of the target site
        pos_target = physics.named.data.site_xpos[target_site_name].copy()
        rot_target = physics.named.data.site_xmat[target_site_name].copy().reshape(3, 3)
        
        # Calculate relative position
        translation_world = pos_target - pos_ref
        relative_position = rot_ref_inv @ translation_world

        # Calculate relative orientation
        relative_rotation = rot_ref_inv @ rot_target
        relative_quaternion = Rotation.from_matrix(relative_rotation).as_quat(scalar_first=True)

        # compensate
        # relative_position -= np.array([0.19, 0, 0])
        return relative_position, relative_quaternion

    def get_qpos(self, physics):
        qpos_raw = physics.data.qpos.copy()
        if self.single_arm:
            right_qpos_raw = qpos_raw[:8]
            right_arm_qpos = right_qpos_raw[:6]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
            return np.concatenate([right_arm_qpos, right_gripper_qpos])
        else:
            left_qpos_raw = qpos_raw[:8]
            right_qpos_raw = qpos_raw[8:16]
            left_arm_qpos = left_qpos_raw[:6]
            right_arm_qpos = right_qpos_raw[:6]
            left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
            return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_qvel(self, physics):
        qvel_raw = physics.data.qvel.copy()
        if self.single_arm:
            right_qvel_raw = qvel_raw[:8]
            right_arm_qvel = right_qvel_raw[:6]
            right_gripper_qvel = [ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
            return np.concatenate([right_arm_qvel, right_gripper_qvel])
        else:
            left_qvel_raw = qvel_raw[:8]
            right_qvel_raw = qvel_raw[8:16]
            left_arm_qvel = left_qvel_raw[:6]
            right_arm_qvel = right_qvel_raw[:6]
            left_gripper_qvel = [ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
            right_gripper_qvel = [ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
            return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_eepos(self, physics):
        if self.single_arm:
            ## Fix this to singlearm_dir
            right_ee_pos_raw, right_ee_quat_raw = self.get_relative_pose(
                physics,
                f'{self.single_arm_dir}/gripper',
                f'{self.single_arm_dir}/actuation_center'
            )
            qpos_raw = physics.data.qpos.copy()
            right_qpos_raw = qpos_raw[:8]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]

            return np.concatenate([right_ee_pos_raw, right_ee_quat_raw, right_gripper_qpos])
        else:
            right_ee_pos_raw, right_ee_quat_raw = self.get_relative_pose(
                physics,
                f'right/gripper',
                f'right/actuation_center'
            )

            left_ee_pos_raw, left_ee_quat_raw = self.get_relative_pose(
                physics,
                f'left/gripper',
                f'left/actuation_center'
            )
            # print(left_ee_pos_raw, right_ee_pos_raw)
            qpos_raw = physics.data.qpos.copy()
            left_qpos_raw = qpos_raw[:8]
            right_qpos_raw = qpos_raw[8:16]
            left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]

            return np.concatenate([left_ee_pos_raw, left_ee_quat_raw, left_gripper_qpos, right_ee_pos_raw, right_ee_quat_raw, right_gripper_qpos])
        
    def get_eepos_rpy(self, physics):
        if self.single_arm:
            right_ee_pos_raw, right_ee_quat_raw = self.get_relative_pose(
                physics,
                f'{self.single_arm_dir}/gripper',
                f'{self.single_arm_dir}/actuation_center'
            )
            right_ee_rpy_raw = quat_to_rpy(*right_ee_quat_raw)
            qpos_raw = physics.data.qpos.copy()
            right_qpos_raw = qpos_raw[:8]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
            return np.concatenate([right_ee_pos_raw, right_ee_rpy_raw, right_gripper_qpos])
        else:
            right_ee_pos_raw, right_ee_quat_raw = self.get_relative_pose(
                physics,
                f'right/gripper',
                f'right/actuation_center'
            )
            right_ee_rpy_raw = quat_to_rpy(*right_ee_quat_raw)
            left_ee_pos_raw, left_ee_quat_raw = self.get_relative_pose(
                physics,
                f'left/gripper',
                f'left/actuation_center'
            )
            left_ee_rpy_raw = quat_to_rpy(*left_ee_quat_raw)
            
            qpos_raw = physics.data.qpos.copy()
            left_qpos_raw = qpos_raw[:8]
            right_qpos_raw = qpos_raw[8:16]
            left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
            return np.concatenate([left_ee_pos_raw, left_ee_rpy_raw, left_gripper_qpos, right_ee_pos_raw, right_ee_rpy_raw, right_gripper_qpos])
        
    def get_eepos_6d(self, physics):
        if self.single_arm:
            ## Fix this to singlearm_dir
            right_ee_pos_raw, right_ee_quat_raw = self.get_relative_pose(
                physics,
                f'{self.single_arm_dir}/gripper',
                f'{self.single_arm_dir}/actuation_center'
            )
            qpos_raw = physics.data.qpos.copy()
            right_qpos_raw = qpos_raw[:8]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
            right_ee_6d_raw = quat_to_6d(*right_ee_quat_raw)

            return np.concatenate([right_ee_pos_raw, right_ee_6d_raw, right_gripper_qpos])
        else:
            right_ee_pos_raw, right_ee_quat_raw = self.get_relative_pose(
                physics,
                f'right/gripper',
                f'right/actuation_center'
            )

            left_ee_pos_raw, left_ee_quat_raw = self.get_relative_pose(
                physics,
                f'left/gripper',
                f'left/actuation_center'
            )
            qpos_raw = physics.data.qpos.copy()
            left_qpos_raw = qpos_raw[:8]
            right_qpos_raw = qpos_raw[8:16]
            left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
            right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
            left_ee_6d_raw = quat_to_6d(*left_ee_quat_raw)
            right_ee_6d_raw = quat_to_6d(*right_ee_quat_raw)

            return np.concatenate([left_ee_pos_raw, left_ee_6d_raw, left_gripper_qpos, right_ee_pos_raw, right_ee_6d_raw, right_gripper_qpos])

    def get_env_state(self, physics):
        return physics.data.qpos.copy()[self.robot_offset:]

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['ee_pos'] = self.get_eepos(physics)
        obs['ee_rpy_pos'] = self.get_eepos_rpy(physics)
        obs['ee_6d_pos'] = self.get_eepos_6d(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['back'] = physics.render(height=480, width=640, camera_id='teleoperator_pov')
        obs['images']['front'] = physics.render(height=480, width=640, camera_id='collaborator_pov')
        if not self.single_arm:
            obs['images']['wrist_left'] = physics.render(height=240, width=320, camera_id='wrist_cam_left')
            obs['images']['wrist_right'] = physics.render(height=240, width=320, camera_id='wrist_cam_right')
        else:
            obs['images'][f'wrist'] = physics.render(height=240, width=320, camera_id=f'wrist_cam_{self.single_arm_dir}')
        obs['language_instruction'] = self.get_instruction(self.reward)
        return obs

    def update_contact(self, physics):
        self.all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            self.all_contact_pairs.append(contact_pair)

    def get_reward(self, physics, reward_condition_list=[[False, 0]]):
        self.max_reward = len(reward_condition_list)
        if not self.reward == self.max_reward:
            if reward_condition_list[self.reward][0]:
                if self.reward_counter == reward_condition_list[self.reward][1]:
                    self.reward += 1
                    self.reward_counter = 0
                else:
                    self.reward_counter += 1
            else:
                self.reward_counter = 0
        return self.reward
    
    def get_instruction(self, reward):
        return self.instruction
    
    def get_touch_condition(self, physics, name1, name2): ## Call after update_contact
        name1 = self.get_geoms(physics, name1)
        name2 = self.get_geoms(physics, name2)
        for contact_pair in self.all_contact_pairs:
            if (contact_pair[0] in name1 and contact_pair[1] in name2) or \
            (contact_pair[0] in name2 and contact_pair[1] in name1):
                return True  # Contact detected
        return False  # No contact

    def get_geoms(self, physics, name):
        if name == 'right_arm':
            geoms = ['right/right_g0', 'right/right_g1', 'right/right_g2', 'right/left_g0', 'right/left_g1', 'right/left_g2']
        elif name == 'left_arm':
            geoms = ['left/right_g0', 'left/right_g1', 'left/right_g2', 'left/left_g0', 'left/left_g1', 'left/left_g2']
        elif name == 'table':
            geoms = ['table']
        elif name in self.obj_dict.keys():
            geoms = self.obj_dict[name].get_geoms()
        else:
            geoms = []
        return geoms

    def get_pos_condition(self, physics, pos1, pos2, delta=0.1):
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) < delta

    def get_rpy_condtion(self, physics, rpy1, rpy2, delta=0.1, mask=[1, 1, 1]):
        diff = np.abs(np.array(rpy1) - np.array(rpy2))
        diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)  # Handle angle wrapping
        masked_diff = diff * np.array(mask)
        return np.all(masked_diff < delta)

    
