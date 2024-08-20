import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import random

from tabletop.constants import DT, XML_DIR, START_ARM_POSE, ONEARM_START_ARM_POSE
from tabletop.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from tabletop.constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from tabletop.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from tabletop.constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_clean' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_clean.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CleanTask(random=False)
        env = control.Environment(physics, task, time_limit=40, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_onearm_clean' in task_name:
        xml_path = os.path.join(XML_DIR, f'onearm_viperx_clean.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = OneArmCleanTask(random=False)
        env = control.Environment(physics, task, time_limit=40, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class OneArmViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        normalized_left_gripper_action = action[6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        left_arm_qpos = left_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        left_arm_qvel = left_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

class CleanTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 8
        self.obj_names = ['bowl1', 'bowl2', 'bottle1', 'bottle2', 'cup1', 'cup2', 'can1', 'can2']
        self.obj_quats = {
            'bowl1' : [0, 0, 0, 1],
            'bowl2' : [0, 0, 0, 1],
            'cup1' : [0.7071068, 0, 0, 0.7071068],
            'cup2' : [0.7071068, 0, 0, 0.7071068],
            'bottle1' : [0, 0, 0, 1],
            'bottle2' : [0, 0, 0, 1],
            'can1' : [0, 0, 0, 1],
            'can2' : [0, 0, 0, 1],
        }
        self.start_idx = dict()
        for i, name in enumerate(self.obj_quats):
            self.start_idx[name] = 16 + 7 * i
        self.center_points = np.array([
            [-0.15, 0.3 + 0.075, 0.01],
            [-0.15, 0.45 + 0.075, 0.01],
            [-0.15, 0.6 + 0.075, 0.01],
            [-0.15, 0.75 + 0.075, 0.01],
            [0.15, 0.3 + 0.075, 0.01],
            [0.15, 0.45 + 0.075, 0.01],
            [0.15, 0.6 + 0.075, 0.01],
            [0.15, 0.75 + 0.075, 0.01],
        ])
        self.obj_order = list(range(len(self.obj_names)))
        self.random_init = True
        

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # self.initialize_robots(physics)
        # randomize box position
        print('init ep', self.random_init)
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            if self.random_init:
                poses = self.sample_obj_poses()
                for name in self.obj_names:
                    start_idx = self.start_idx[name]
                    np.copyto(physics.data.qpos[start_idx : start_idx + 7], poses[name])
            self.random_init = True
        super().initialize_episode(physics)

    def init_objects(self, physics, obj_dict):
        self.random_init = False
        print('init', obj_dict)
        for arm in obj_dict.values():
            for name, pos in arm.items():
                start_idx = self.start_idx[name]
                pos = np.concatenate([pos, [0.01]])
                np.copyto(physics.data.qpos[start_idx : start_idx + 3], pos)

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['obj_dict'] = self.get_obj_dict(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        
        obs['language_instruction'] = 'clean the table'
        return obs

    @staticmethod
    def get_env_state(physics):
        pos_list = []
        offset = 2
        for i in range(8):
            pos_list = pos_list + list(physics.data.qpos.copy()[16 + i * 7 : 16 + i * 7 + offset])
        return np.array(pos_list)
    
    def get_obj_dict(self, physics):
        pos_dict = {'left' : {}, 'right': {}}
        for i in self.obj_order[:4]:
            start_idx = self.start_idx[self.obj_names[i]]
            # print(i, self.obj_names[i], physics.data.qpos.copy()[start_idx : start_idx + 3])
            pos_dict['left'][self.obj_names[i]] = physics.data.qpos.copy()[start_idx : start_idx + 2]
        for i in self.obj_order[4:]:
            start_idx = self.start_idx[self.obj_names[i]]
            # print(i, self.obj_names[i], physics.data.qpos.copy()[start_idx : start_idx + 3])
            pos_dict['right'][self.obj_names[i]] = physics.data.qpos.copy()[start_idx : start_idx + 2]
        return pos_dict

    def get_reward(self, physics):
        reward = 0
        for i in range(8):
            if physics.data.qpos.copy()[16 + i * 7 + 2] < -0.15:
                reward += 1
        return reward

    def sample_obj_poses(self):
        self.obj_order = list(range(len(self.obj_names)))
        random.shuffle(self.obj_order)
        poses = dict()
        for i in range(len(self.obj_names)):
            noise = 0.03 * np.random.normal(size=(3,))
            noise[1] = noise[1] * 0.1
            noise[-1] = 0
            pos = self.center_points[i] + noise
            quats = np.array(self.obj_quats[self.obj_names[self.obj_order[i]]])
            poses[self.obj_names[self.obj_order[i]]] = np.concatenate([pos, quats])
            # print(self.obj_names[self.obj_order[i]], pos)
        return poses

class OneArmCleanTask(OneArmViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.obj_names = ['bowl', 'cup', 'red can', 'green can']
        self.obj_quats = {
            'bowl' : [0, 0, 0, 1],
            'cup' : [0.7071068, 0, 0, 0.7071068],
            'red can' : [0, 0, 0, 1],
            'green can' : [0, 0, 0, 1],
        }
        self.start_idx = dict()
        for i, name in enumerate(self.obj_quats):
            self.start_idx[name] = 8 + 7 * i
        self.center_points = np.array([
            [-0.2, 0.3 + 0.075, 0.01],
            [-0.2, 0.45 + 0.075, 0.01],
            [0.0, 0.3 + 0.075, 0.01],
            [0.0, 0.45 + 0.075, 0.01],
        ])
        self.obj_order = list(range(len(self.obj_names)))
        self.random_init = True
        self.instruction = 'clean the table'
        

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # self.initialize_robots(physics)
        # randomize box position
        print('init ep', self.random_init)
        with physics.reset_context():
            physics.named.data.qpos[:8] = ONEARM_START_ARM_POSE
            np.copyto(physics.data.ctrl, ONEARM_START_ARM_POSE)
            if self.random_init:
                poses = self.sample_obj_poses()
                for name in self.obj_names:
                    start_idx = self.start_idx[name]
                    np.copyto(physics.data.qpos[start_idx : start_idx + 7], poses[name])
            self.random_init = True
        super().initialize_episode(physics)

    def init_objects(self, physics, obj_dict):
        self.random_init = False
        print('init', obj_dict)
        for arm in obj_dict.values():
            for name, pos in arm.items():
                start_idx = self.start_idx[name]
                pos = np.concatenate([pos, [0.01]])
                np.copyto(physics.data.qpos[start_idx : start_idx + 3], pos)

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['obj_dict'] = self.get_obj_dict(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        
        obs['language_instruction'] = 'clean the table'
        return obs

    @staticmethod
    def get_env_state(physics):
        pos_list = []
        offset = 2
        for i in range(4):
            pos_list = pos_list + list(physics.data.qpos.copy()[8 + i * 7 : 8 + i * 7 + offset])
        return np.array(pos_list)
    
    def get_obj_dict(self, physics):
        pos_dict = {'left' : {}}
        for i in self.obj_order[:4]:
            start_idx = self.start_idx[self.obj_names[i]]
            # print(i, self.obj_names[i], physics.data.qpos.copy()[start_idx : start_idx + 3])
            pos_dict['left'][self.obj_names[i]] = physics.data.qpos.copy()[start_idx : start_idx + 2]
        return pos_dict

    def get_reward(self, physics):
        reward = 0
        for i in range(4):
            if physics.data.qpos.copy()[8 + i * 7 + 2] < -0.15:
                reward += 1
        return reward

    def sample_obj_poses(self):
        self.obj_order = list(range(len(self.obj_names)))
        random.shuffle(self.obj_order)
        poses = dict()
        for i in range(len(self.obj_names)):
            noise = 0.03 * np.random.normal(size=(3,))
            noise[1] = noise[1] * 0.1
            noise[-1] = 0
            pos = self.center_points[i] + noise
            quats = np.array(self.obj_quats[self.obj_names[self.obj_order[i]]])
            poses[self.obj_names[self.obj_order[i]]] = np.concatenate([pos, quats])
            # print(self.obj_names[self.obj_order[i]], pos)
        return poses


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

