"""
Train edge_2d CNN from scratch.
Saves to a separate directory to preserve pretrained weights.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from tactile_gym.utils.general_utils import save_json_obj, check_dir
from tactile_gym_servo_control.learning.learning_utils import import_task, seed_everything
from tactile_gym_servo_control.learning.networks import create_model
from tactile_gym_servo_control.learning.train_cnn import train_cnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

task = 'edge_2d'

model_params = {
    'model_type': 'nature_cnn',
    'model_kwargs': {
        'fc_layers': [512, 512],
        'dropout': 0.0,
    },
}

learning_params = {
    'seed': 42,
    'batch_size': 128,
    'epochs': 50,
    'lr': 1e-4,
    'lr_factor': 0.5,
    'lr_patience': 10,
    'shuffle': True,
    'n_cpu': 4,
    'plot_during_training': False,
}

image_processing_params = {
    'dims': (128, 128),
    'bbox': None,
    'thresh': False,
    'stdiz': False,
    'normlz': True,
}

augmentation_params = {
    'rshift': (0.025, 0.025),
    'rzoom': None,
    'brightlims': None,
    'noise_var': None,
}

seed_everything(learning_params['seed'])

save_dir = os.path.expanduser(
    '~/aerial-tactile-sim/servo_control/tactile_gym_servo_control/learned_models/edge_2d/tap_retrained'
)
os.makedirs(save_dir, exist_ok=True)

save_json_obj(model_params, os.path.join(save_dir, 'model_params'))
save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
save_json_obj(image_processing_params, os.path.join(save_dir, 'image_processing_params'))
save_json_obj(augmentation_params, os.path.join(save_dir, 'augmentation_params'))

out_dim, label_names = import_task(task)
print(f"Task: {task}, out_dim: {out_dim}, labels: {label_names}")

model = create_model(
    image_processing_params['dims'],
    out_dim,
    model_params,
    saved_model_dir=None,
    device=device
)

train_cnn(
    task,
    model,
    label_names,
    learning_params,
    image_processing_params,
    augmentation_params,
    save_dir,
    device
)

print("\nDone! Model saved to:", save_dir)
