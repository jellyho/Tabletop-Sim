import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from tabletop.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from tabletop.ee_sim_env import make_ee_sim_env
from tabletop.sim_env import make_sim_env, BOX_POSE
from tabletop.scripted_policy import PickAndTransferPolicy, InsertionPolicy, CleanPolicy, OneArmCleanPolicy

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'angle'

    real_dix = 0

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    
    if task_name == 'sim_onearm_clean':
        policy_cls = OneArmCleanPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        episode_inst = ['clean the table']
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action, inst = policy(ts, True)
            ts = env.step(action)
            # print(step, ts.reward)
            episode.append(ts)
            episode_inst.append(inst)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.001)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        eepos_traj = [ts.observation['ee_pos'] for ts in episode]
        eevel_traj = [ts.observation['ee_vel'] for ts in episode]
        inst_traj = episode_inst
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            joint[6] = left_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0
        if task_name == 'sim_clean' or task_name == 'sim_onearm_clean':
            subtask_info = episode[0].observation['obj_dict']

        # clear unused variables
        del env
        del policy
        del episode
        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        if task_name == 'sim_clean' or task_name == 'sim_onearm_clean':
            ts = env.reset()
            env.task.init_objects(env.physics, subtask_info)

        else:
            BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")
            continue

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'
        - state                 (2*8, )       'float64'
        - instructions (1, )         '???'

        action                  (14,)         'float64'
        actions
        - joint_pos
        - joint_vel
        - ee_pos
        - ee_vel
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/state': [],
            '/observations/instructions': [],
            '/action': [],
            '/actions/joint_pos' : [],
            '/actions/joint_vel' : [],
            '/actions/ee_pos' : [],
            '/actions/ee_vel' : [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[1:-1]
        episode_replay = episode_replay[1:-1]
        eepos_traj = eepos_traj[1:-1]
        eevel_traj = eevel_traj[1:-1]
        inst_traj[1:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            eepos_action = eepos_traj.pop(0)
            eevel_action = eevel_traj.pop(0)
            ts = episode_replay.pop(0)
            inst = inst_traj.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            if task_name == 'sim_onearm_clean':
                data_dict['/observations/state'].append(ts.observation['env_state'])
                data_dict['/observations/instructions'].append(inst)
            data_dict['/action'].append(action)
            data_dict['/actions/joint_pos'].append(action)
            data_dict['/actions/joint_vel'].append(ts.observation['qvel'])
            data_dict['/actions/ee_pos'].append(eepos_action)
            data_dict['/actions/ee_vel'].append(eevel_action)

            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{real_idx}')
        real_idx += 1
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 240, 320, 3), dtype='uint8',
                                         chunks=(1, 240, 320, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 7))
            qvel = obs.create_dataset('qvel', (max_timesteps, 7))
            action = root.create_dataset('action', (max_timesteps, 7))
            actions = root.create_group('actions')
            if task_name == 'sim_onearm_clean':
                state = obs.create_dataset('state', (max_timesteps, 2 * 4))
                action_joint_pos = actions.create_dataset('joint_pos', (max_timesteps, 7))
                action_joint_vel = actions.create_dataset('joint_vel', (max_timesteps, 7))
                action_ee_pos = actions.create_dataset('ee_pos', (max_timesteps, 8))
                action_ee_vel = actions.create_dataset('ee_vel', (max_timesteps, 8))
                instructions = obs.create_dataset('instructions', (max_timesteps,), dtype=h5py.string_dtype(encoding='utf-8'))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

