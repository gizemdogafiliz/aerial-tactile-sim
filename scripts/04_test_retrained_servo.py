"""
Test servo control with our retrained model (50 epochs).
Compare with pretrained model by changing model_dir below.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from tactile_gym.utils.general_utils import load_json_obj
from tactile_gym_servo_control.learning.learning_utils import import_task, POSE_LABEL_NAMES
from tactile_gym_servo_control.learning.networks import create_model
from tactile_gym_servo_control.servo_control.servo_control import (
    load_embodiment_and_env,
    run_servo_control,
)
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_2d_servo_control

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# SWITCH HERE: 'tap' = pretrained, 'tap_retrained' = our 50-epoch model
model_dir = os.path.expanduser(
    '~/aerial-tactile-sim/servo_control/tactile_gym_servo_control/learned_models/edge_2d/tap_retrained'
)
print(f"Model: {model_dir}")

task = 'edge_2d'
out_dim, label_names = import_task(task)

pose_limits_dict = load_json_obj(os.path.join(model_dir, 'pose_limits'))
pose_limits = [pose_limits_dict['pose_llims'], pose_limits_dict['pose_ulims']]
model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
image_processing_params = load_json_obj(os.path.join(model_dir, 'image_processing_params'))

move_init_pose, stim_names, ep_len, init_ref_pose, p_gains = setup_edge_2d_servo_control()

stim_name = stim_names[0]
embodiment, ref_pose_ids = load_embodiment_and_env(init_ref_pose, stim_name)
move_init_pose(embodiment, stim_name)

trained_model = create_model(
    image_processing_params['dims'],
    out_dim,
    model_params,
    saved_model_dir=model_dir,
    device=device
)
trained_model.eval()

print(f"\nRunning servo control with retrained model...")
print(f"Use PyBullet GUI sliders to change reference pose.")
print(f"Press 'q' in PyBullet window to quit.\n")

RECORD_VIDEO = True

run_servo_control(
    embodiment,
    trained_model,
    image_processing_params,
    p_gains=p_gains,
    label_names=label_names,
    pose_limits=pose_limits,
    ref_pose_ids=ref_pose_ids,
    device=device,
    ep_len=ep_len,
    quick_mode=False,
    record_vid=RECORD_VIDEO,
)
