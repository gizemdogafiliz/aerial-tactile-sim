"""
Watch data collection in GUI.
20 samples, slow mode — see the TacTip sensor press against the edge.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.data_collection.collect_data import collect_data
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_2d_data_collection

stimuli_path = os.path.expanduser(
    '~/aerial-tactile-sim/servo_control/tactile_gym_servo_control/stimuli'
)
output_dir = os.path.expanduser('~/aerial-tactile-sim/output/data_collection_demo')
os.makedirs(output_dir, exist_ok=True)

num_samples = 20

target_df, image_dir, workframe_pos, workframe_rpy = setup_edge_2d_data_collection(
    num_samples=num_samples,
    apply_shear=False,
    shuffle_data=False,
    collect_dir_name="demo_gui",
)

print(f"Collecting {num_samples} samples...")
print(f"Images will be saved to: {image_dir}")
print(f"\nTarget poses (first 5):")
print(target_df[['pose_2', 'pose_6']].head())
print("\npose_2 = y position (mm), pose_6 = Rz angle (degrees)")

tactip_params = {
    "name": "tactip",
    "type": "standard",
    "core": "no_core",
    "dynamics": {},
    "image_size": [128, 128],
    "turn_off_border": False,
}

stimulus_pos = [0.6, 0.0, 0.0125]
stimulus_rpy = [0, 0, 0]
stim_path = os.path.join(stimuli_path, "square/square.urdf")

embodiment, _ = setup_pybullet_env(
    stim_path,
    tactip_params,
    stimulus_pos,
    stimulus_rpy,
    workframe_pos,
    workframe_rpy,
    show_gui=True,
    show_tactile=True,
)

collect_data(
    embodiment,
    target_df,
    image_dir,
    workframe_pos,
    workframe_rpy,
    tactip_params,
    show_gui=True,
    show_tactile=True,
    quick_mode=False,
)

print(f"\nDone! {num_samples} images saved to {image_dir}")
