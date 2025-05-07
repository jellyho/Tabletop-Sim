# import cv2
import sys
import os
import numpy as np
import argparse
import h5py
import glob
from PIL import Image

def replay(task_name, episode_dir, save_dir):
    accumulator = None
    episode_files = glob.glob(f'{episode_dir}/{task_name}/episode_*.hdf5')
    count = 0
    for files in episode_files:
        episode = h5py.File(files, 'r')
        first_frame = episode['/observations/images/back'][0]
        frame = first_frame.astype(np.float32) / 255.0  # 정규화 (0~1)

        if accumulator is None:
            accumulator = np.zeros_like(frame)

        accumulator += frame
        count += 1
    overlay = accumulator / count
    overlay_path = os.path.join(save_dir, f'{task_name}_overlay.png')
    overlay_image = (overlay * 255).astype(np.uint8)  # Convert back to 0-255 range
    Image.fromarray(overlay_image).save(overlay_path)
    print(f"Overlay saved to {overlay_path}")
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', action='store', type=str, default='aloha_dish_drainer', required=False)
    parser.add_argument('-d', '--episode_dir', action='store', type=str, default='datasets')
    parser.add_argument('-s', '--save_dir', action='store', type=str, default='.')
    
    args = parser.parse_args()

    replay(args.task_name, args.episode_dir, args.save_dir)