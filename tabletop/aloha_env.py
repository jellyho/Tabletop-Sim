import numpy as np
from random import sample
from tabletop.constants import *
from tabletop.utils import *
from tabletop.aloha_env_base import AlohaTask
from scipy.spatial.transform import Rotation as R
import dm_env

import itertools

class DishDrainer(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('drainer', 'Rubbermaid_Large_Drainer', pos=[-0.1, 0.1, 0.01], rpy=[0, 0, -60], scale=[0.8, 0.8, 0.8])
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.1, 0, 0.01], scale=[0.8, 0.8, 0.8], mass=0.2)
        self.instruction = 'Pick up the dish and put on to the drainer'

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        plate_pos = np.array([0.15, -0.17, 0.01])
        plate_pos[:2] += random_vector * 0.015
        plate_rpy = np.array([0, 0, 0],)
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=plate_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'drainer', 'plate'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
class HandoverBox(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('basket', 'Room_Essentials_Fabric_Cube_Lavender', pos=[0.2, 0.0, 0.01], rpy=[0, 0, -40], scale=[0.6, 0.6, 0.6])
        self.add_object('box', 'Fresca_Peach_Citrus_Sparkling_Flavored_Soda_12_PK', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 90], scale=[0.4, 0.25, 0.4], mass=0.3)
        self.instruction = 'Handover the box and place into the pink basket'

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        basket_pos = np.array([0.2, 0.1, 0.01])
        basket_pos[:2] += random_vector * 0.02
        basket_rpy = np.array([0.0, 0.0, 0.0],)

        random_vector = np.random.randn(2)
        box_pos = np.array([-0.2, -0.2, 0.01])
        box_pos[:2] += random_vector * 0.025
        box_rpy = np.array([0.0, 0.0, 0.0],)
        random_vector = np.random.randn(1) / (2 * np.pi)
        box_rpy[0] += random_vector
        self.set_object_pose(physics, 'basket', pos=basket_pos, rpy=basket_rpy)
        self.set_object_pose(physics, 'box', pos=box_pos, rpy=box_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'box', 'basket'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first

class BoxIntoPot(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first

        self.combination_list = self._generate_combination(num_repeat=1)
        self.benchmark_info = [self._generate_combination(num_repeat=5)] # NOTE: hard coded.
        self.instruction_template = "Put the {object} into the {pot}"
        self.instruction = None
        self.objects = ['brown box', 'white box', 'yellow box']
        self.pots = ['blue pot', 'green pot']
        self.target_object = None
        self.target_pot = None
        self.use_fixed_combination = False
        self.objects_poses = [
            [-0.1, 0.05, 0.05],
            [-0.2, -0.05, 0.05],
            [0.00, -0.05, 0.05],
        ]
        self.pot_poses = {
            'blue pot': [0.25, 0.2, 0.01],
            'green pot': [0.25, -0.2, 0.01]
        }

        self.add_object('blue pot', 'Ecoforms_Garden_Pot_GP16ATurquois', pos=self.pot_poses['blue pot'], rpy=[0, 0, 0], scale=[0.9, 0.9, 0.9])
        self.add_object('green pot', 'Ecoforms_Plant_Container_12_Pot_Nova', pos=self.pot_poses['green pot'], rpy=[0, 0, 0], scale=[0.5, 0.5, 0.5])
        self.add_object('brown box', 'Perricone_MD_Neuropeptide_Firming_Moisturizer', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[1.3, 0.6, 0.7], mass=0.1)
        self.add_object('white box', 'Perricone_MD_Firming_Neck_Therapy_Treatment', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[1.3, 0.5, 0.6], mass=0.1)
        self.add_object('yellow box', 'Ultra_JarroDophilus', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[1.3, 0.6, 0.4], mass=0.1)

    def _generate_combination(self, num_repeat:int = 1):
        objects = ["yellow box", "white box", "brown box"]
        pots = ["blue pot", "green pot"]
        all_combinations = []
        
        for pot in pots:
            for target in objects:
                for perm in itertools.permutations(objects):
                    combo = (target, pot, perm)
                    for _ in range(num_repeat):
                        all_combinations.append(combo)
        
        return all_combinations
    
    def benchmark_init(self, physics, idx):
        comb = self.combination_list[idx % len(self.combination_list)]
        self.set_combination(comb)
        self.initialize_episode(physics)
        physics.step()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=self.get_observation(physics)
        )

    def set_combination(self, combination):
        """
        Set a specific combination for this task.
        combination: tuple of (target_object, target_pot, object_order)
        """
        if combination:
            self.use_fixed_combination = True
            self.fixed_target_object = combination[0]
            self.fixed_target_pot = combination[1]
            self.fixed_object_order = combination[2]
            # Update instruction immediately
            self.instruction = self.instruction_template.format(object=self.fixed_target_object, pot=self.fixed_target_pot)

    def initialize_episode(self, physics):
        self.set_object_pose(physics, 'blue pot', pos=self.pot_poses['blue pot'], rpy=[0, 0, 0])
        self.set_object_pose(physics, 'green pot', pos=self.pot_poses['green pot'], rpy=[0, 0, 0])
        
        # Use fixed combination if available, otherwise randomize
        if self.use_fixed_combination:
            self.target_object = self.fixed_target_object
            self.target_pot = self.fixed_target_pot
            for i, obj in enumerate(self.fixed_object_order):
                if i < len(self.objects_poses):
                    self.set_object_pose(physics, obj, pos=self.objects_poses[i], rpy=[np.pi/2, 0, 0])
        else:
            # Randomize objects positions
            objects_poses_copy = self.objects_poses.copy()
            for obj in self.objects:
                random_index = np.random.randint(0, len(objects_poses_copy))
                chosen_pose = objects_poses_copy.pop(random_index)
                self.set_object_pose(physics, obj, pos=chosen_pose, rpy=[np.pi/2, 0, 0])
            
            # Randomize target object and pot
            self.target_object = self.objects[np.random.randint(0, len(self.objects))]
            self.target_pot = self.pots[np.random.randint(0, len(self.pots))]
            
        self.instruction = self.instruction_template.format(object=self.target_object, pot=self.target_pot)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        corret_pot_moved = abs(self.get_object_pose(physics, self.target_pot)[0][0] - self.pot_poses[self.target_pot][0]) > 0.05 and abs(self.get_object_pose(physics, self.target_pot)[0][1] - self.pot_poses[self.target_pot][1]) > 0.05
        ## [condition, counter]
        # Check if the target object is inside the target pot
        target_in_pot = self.get_touch_condition(physics, self.target_object, self.target_pot) and abs(self.get_object_pose(physics, self.target_object)[0][2] - self.get_object_pose(physics, self.target_pot)[0][2]) <= 0.1
        
        # Check that other objects are not in the target pot
        others_not_in_pot = True
        for obj in self.objects:
            if obj != self.target_object:
                obj_in_pot = self.get_touch_condition(physics, obj, self.target_pot) and abs(self.get_object_pose(physics, obj)[0][2] - self.get_object_pose(physics, self.target_pot)[0][2]) <= 0.1
                if obj_in_pot:
                    others_not_in_pot = False
                    break
        
        reward_condition_list = [
            [corret_pot_moved, 20],
            [target_in_pot and others_not_in_pot, 10],
        ]
        # print(f"inst: {self.instruction}\treward: {target_in_pot and others_not_in_pot}")
        return super().get_reward(physics, reward_condition_list) ### always first

class BoxIntoPotEasy(BoxIntoPot):
    def _generate_combination(self, num_repeat: int = 1):
        objects = ["yellow box", "white box", "brown box"]
        pots = ["blue pot", "green pot"]
        all_combinations = []
        for pot in pots:
            for target in objects:
                combo = (target, pot, tuple(objects))
                for _ in range(num_repeat):
                    all_combinations.append(combo)
        return all_combinations

class ShoesTable(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('toy_table', '3D_Dollhouse_TablePurple', pos=[0.0, 0.1, 0.01], rpy=[0, 0, -80], scale=[3.0, 4.0, 2.0], mass=10.0)
        self.add_object('shoe_right', 'Womens_Canvas_Bahama_in_Black_vnJULsDVyq5', pos=[0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[0.65, 0.65, 0.65], mass=0.3)
        self.add_object('shoe_left', 'Womens_Canvas_Bahama_in_Black', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, -90], scale=[0.65, 0.65, 0.65], mass=0.3)

        self.instruction = 'Pick up the black shoes and put them side by side on the purple table'

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        tabletop_pos = np.array([0.0, 0.1, 0.01])
        tabletop_pos[:2] += random_vector * 0.01
        tabletop_rpy = np.array([-100/180.0*np.pi, 0, 0])

        shoe_right_pos = np.array([0.15, -0.15, 0.01])
        random_vector = np.random.randn(2)
        shoe_right_pos[:2] += random_vector * 0.05
        shoe_right_rpy = np.array([-np.pi, 0, 0])
        random_vector = np.random.randn(1)
        shoe_right_rpy[0] += random_vector * (1 / 9) * np.pi
        shoe_left_pos = np.array([-0.15, -0.15, 0.01])
        random_vector = np.random.randn(2)
        shoe_left_pos[:2] += random_vector * 0.05
        shoe_left_rpy = np.array([-90/180.0*np.pi, 0, 0])
        random_vector = np.random.randn(1)
        shoe_left_rpy[0] += random_vector * (1 / 9) * np.pi
        self.set_object_pose(physics, 'toy_table', tabletop_pos, tabletop_rpy)
        self.set_object_pose(physics, 'shoe_right', shoe_right_pos, shoe_right_rpy)
        self.set_object_pose(physics, 'shoe_left', shoe_left_pos, shoe_left_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'shoe_right', 'toy_table') and self.get_touch_condition(physics, 'shoe_left', 'toy_table'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first

class LiftBox(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('box', 'Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz', pos=[0.0, 0.0, 0.1], rpy=[0, 0, 0], scale=[5, 2, 2], mass=1)

        self.instruction = 'Lift the box with the front facing the camera'

    def initialize_episode(self, physics):
        # Generate random position for box within [-1.0, 1.0] range
        box_pos = np.array([
            np.random.uniform(-0.1, 0.1),  # x-coordinate
            np.random.uniform(-0.05, 0.05),  # y-coordinate
            0.02  # z-coordinate (kept at default)
        ])

        rotation = np.random.uniform(-np.pi/6.0, np.pi/6.0)
        self.set_object_pose(physics, 'box', pos=box_pos, rpy=[rotation, 0, 0])
        
        # Always call the parent's initialize_episode at the end
        super().initialize_episode(physics)  # always last

    def get_reward(self, physics):
        ## [condition, counter]
        rotation_rpy = R.from_quat(self.get_object_pose(physics, 'box')[1]).as_euler('xyz')
        rotation_rpy = list(rotation_rpy)
        is_box_float = self.get_object_pose(physics, 'box')[0][2] > 0.1
        is_rotation_okay = abs(abs(rotation_rpy[0]) - np.pi) <= 0.15  and abs(rotation_rpy[1]) < 0.1 and abs(rotation_rpy[2]) < 0.1
        reward_condition_list = [
            [is_box_float and is_rotation_okay, 10],
        ]
        # print(f"cond1: {abs(abs(rotation_rpy[0]) - np.pi) <= 0.15}\tcond2: {abs(rotation_rpy[1]) < 0.1}\tcond3: {abs(rotation_rpy[2]) < 0.1}")
        return super().get_reward(physics, reward_condition_list)
    
############# Generalization tasks #############
class DishDrainerNew(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.original_task = 'aloha_dish_drainer' # For benchmark init
        self.add_object('drainer', 'Poppin_File_Sorter_Pink', pos=[-0.1, 0.1, 0.01], rpy=[0, 0, -60], scale=[0.8, 0.8, 0.8])
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.1, 0, 0.01], rpy=[0, 0, -40], scale=[0.8, 0.8, 0.8], mass=0.2)
        self.instruction = 'Pick up the dish and put on to the drainer'

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        plate_pos = np.array([0.15, -0.17, 0.01])
        plate_pos[:2] += random_vector * 0.015
        plate_rpy = np.array([0, 0, 0],)
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=plate_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'drainer', 'plate'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
class HandoverBoxNew(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.original_task = 'aloha_handover_box'
        self.add_object('basket', 'Room_Essentials_Fabric_Cube_Lavender', pos=[0.2, 0.0, 0.01], rpy=[0, 0, -40], scale=[0.6, 0.6, 0.6])
        self.add_object('box', 'Fresca_Peach_Citrus_Sparkling_Flavored_Soda_12_PK', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 90], scale=[0.4, 0.25, 0.4], mass=0.3)
        self.add_object('toy1', 'Vtech_Stack_Sing_Rings_636_Months', pos=[-0.1, 0.1, 0.01], scale=[0.6, 0.6, 0.6])
        self.add_object('toy2', 'Vtech_Roll_Learn_Turtle', pos=[-0.2, -0.02, 0.01], scale=[0.4, 0.4, 0.4], rpy=[0, 0, 120])
        self.add_object('toy3', 'Toysmith_Windem_Up_Flippin_Animals_Dog', pos=[0.2, -0.2, 0.01], scale=[2.5, 2.5, 2.5], rpy=[0, 0, 180])
        self.instruction = 'Handover the box and place into the pink basket'

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        basket_pos = np.array([0.2, 0.1, 0.01])
        basket_pos[:2] += random_vector * 0.02
        basket_rpy = np.array([0.0, 0.0, 0.0],)

        random_vector = np.random.randn(2)
        box_pos = np.array([-0.2, -0.2, 0.01])
        box_pos[:2] += random_vector * 0.025
        box_rpy = np.array([0.0, 0.0, 0.0],)
        random_vector = np.random.randn(1) / (2 * np.pi)
        box_rpy[0] += random_vector
        self.set_object_pose(physics, 'basket', pos=basket_pos, rpy=basket_rpy)
        self.set_object_pose(physics, 'box', pos=box_pos, rpy=box_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'box', 'basket'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
class LiftBoxNew(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.original_task = 'aloha_lift_box' # For benchmark init
        self.add_object('box', 'Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz', pos=[0.0, 0.0, 0.1], rpy=[0, 0, 0], scale=[5, 2, 2], mass=1)

        self.instruction = 'Lift the box with the front facing the camera'

    def initialize_episode(self, physics):
        # Generate random position for box within [-1.0, 1.0] range
        box_pos = np.array([
            np.random.uniform(-0.1, 0.1),  # x-coordinate
            np.random.uniform(-0.05, 0.05),  # y-coordinate
            0.02  # z-coordinate (kept at default)
        ])

        rotation = np.random.uniform(-np.pi/6.0, np.pi/6.0)
        self.set_object_pose(physics, 'box', pos=box_pos, rpy=[rotation, 0, 0])
        
        # Always call the parent's initialize_episode at the end
        super().initialize_episode(physics)  # always last

    def get_reward(self, physics):
        ## [condition, counter]
        rotation_rpy = R.from_quat(self.get_object_pose(physics, 'box')[1]).as_euler('xyz')
        rotation_rpy = list(rotation_rpy)
        is_box_float = self.get_object_pose(physics, 'box')[0][2] > 0.1
        is_rotation_okay = abs(abs(rotation_rpy[0]) - np.pi) <= 0.15  and abs(rotation_rpy[1]) < 0.1 and abs(rotation_rpy[2]) < 0.1
        reward_condition_list = [
            [is_box_float and is_rotation_okay, 10],
        ]
        # print(f"cond1: {abs(abs(rotation_rpy[0]) - np.pi) <= 0.15}\tcond2: {abs(rotation_rpy[1]) < 0.1}\tcond3: {abs(rotation_rpy[2]) < 0.1}")
        return super().get_reward(physics, reward_condition_list)
    
class ShoesTableNew(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.original_task = 'aloha_shoes_table'
        self.add_object('toy_table', '3D_Dollhouse_TablePurple', pos=[0.0, 0.1, 0.01], rpy=[0, 0, -80], scale=[3.0, 4.0, 2.0], mass=10.0)
        self.add_object('shoe_right', 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_w34KNQ41csH', pos=[0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[0.65, 0.65, 0.65], mass=0.3)
        self.add_object('shoe_left', 'Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_kqi81aojcOR', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[0.65, 0.65, 0.65], mass=0.3)

        self.instruction = 'Pick up the brown shoes and put them side by side on the purple table'

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        tabletop_pos = np.array([0.0, 0.1, 0.01])
        tabletop_pos[:2] += random_vector * 0.01
        tabletop_rpy = np.array([-100/180.0*np.pi, 0, 0])

        shoe_right_pos = np.array([0.15, -0.15, 0.01])
        random_vector = np.random.randn(2)
        shoe_right_pos[:2] += random_vector * 0.05
        shoe_right_rpy = np.array([-np.pi, 0, 0])
        random_vector = np.random.randn(1)
        shoe_right_rpy[0] += random_vector * (1 / 9) * np.pi
        shoe_left_pos = np.array([-0.15, -0.15, 0.01])
        random_vector = np.random.randn(2)
        shoe_left_pos[:2] += random_vector * 0.05
        shoe_left_rpy = np.array([0, 0, 0])
        random_vector = np.random.randn(1)
        shoe_left_rpy[0] += random_vector * (1 / 9) * np.pi
        self.set_object_pose(physics, 'toy_table', tabletop_pos, tabletop_rpy)
        self.set_object_pose(physics, 'shoe_right', shoe_right_pos, shoe_right_rpy)
        self.set_object_pose(physics, 'shoe_left', shoe_left_pos, shoe_left_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'shoe_right', 'toy_table') and self.get_touch_condition(physics, 'shoe_left', 'toy_table'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first

# NOTE: episode_len is actually seconds. so actual episode_len will be episode_len / DT
ALOHA_TASK_CONFIGS = {
    'aloha_dish_drainer': {
        'task_class': DishDrainer,
        'episode_len': 10,
    },
    'aloha_handover_box': {
        'task_class':HandoverBox,
        'episode_len': 15 ,
    },
    'aloha_shoes_table': {
        'task_class': ShoesTable,
        'episode_len': 15,
    },
    'aloha_lift_box': {
        'task_class': LiftBox,
        'episode_len': 15,
    },
    'aloha_box_into_pot': {
        'task_class': BoxIntoPot,
        'episode_len': 10,
    },
    'aloha_box_into_pot_easy': {
        'task_class': BoxIntoPotEasy,
        'episode_len': 10,
    },
    'aloha_dish_drainer_new': {
        'task_class': DishDrainerNew,
        'episode_len': 10
    },
    'aloha_lift_box_new': {
        'task_class': LiftBoxNew,
        'episode_len': 15,
        'table_color': 'marble'
    },
    'aloha_handover_box_new': {
        'task_class':HandoverBoxNew,
        'episode_len': 15,
    },
    'aloha_shoes_table_new': {
        'task_class': ShoesTableNew,
        'episode_len': 15,
    },
}