"""
Phase 4: Train ridge_4d CNN — tactile image → (d_cross_x, d_cross_y, depth, ridge angle).

Finds the latest collected dataset, splits into 80/20 train/val,
then trains NatureCNN using the servo_control pipeline.

CNN outputs:
  x  → d_cross_x_mm  (distance to nearest vertical ridge, ±19mm)
  y  → d_cross_y_mm  (distance to nearest horizontal ridge, ±19mm)
  z  → depth_mm      (contact depth, -2 to 5.5mm)
  Rz → yaw_deg       (ridge angle, ±20°)

Usage:
  conda activate tactile
  python 07_train_ridge_cnn.py               # CPU, 50 epochs
  python 07_train_ridge_cnn.py --epochs 100  # more epochs
  python 07_train_ridge_cnn.py --gpu         # use CUDA
"""

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from tactile_gym.utils.general_utils import save_json_obj
from tactile_gym_servo_control.learning.learning_utils import import_task, seed_everything
from tactile_gym_servo_control.learning.networks import create_model
import tactile_gym_servo_control.learning.train_cnn as train_cnn_module
from tactile_gym_servo_control.learning.train_cnn import train_cnn

# Override accuracy tolerances for aerial tactile task
# Original: 0.25mm / 1.0° — too tight for drone scenario
train_cnn_module.POS_TOL = 2.0   # mm
train_cnn_module.ROT_TOL = 3.5   # deg

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, 'output', 'ridge_dataset')
DATA_DIR = os.path.join(
    PROJECT_DIR, 'servo_control', 'tactile_gym_servo_control', 'data', 'ridge_4d', 'tap'
)
MODEL_BASE = os.path.join(
    PROJECT_DIR, 'servo_control', 'tactile_gym_servo_control', 'learned_models', 'ridge_4d'
)


def find_latest_dataset():
    """Find the most recent collect_* directory."""
    pattern = os.path.join(DATASET_DIR, 'collect_*')
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No dataset found in {DATASET_DIR}")
    return dirs[-1]


def split_dataset(collect_dir, train_ratio=0.8, seed=42):
    """Split collected dataset into train/val and copy to data/ directory."""
    targets = pd.read_csv(os.path.join(collect_dir, 'targets.csv'))
    n = len(targets)

    np.random.seed(seed)
    indices = np.random.permutation(n)
    split = int(train_ratio * n)

    for split_name, idx in [('train', indices[:split]), ('val', indices[split:])]:
        split_dir = os.path.join(DATA_DIR, split_name)
        img_dir = os.path.join(split_dir, 'images')

        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(img_dir)

        split_df = targets.iloc[idx].reset_index(drop=True)
        split_df.to_csv(os.path.join(split_dir, 'targets.csv'), index=False)

        for _, row in split_df.iterrows():
            src = os.path.join(collect_dir, 'images', row['sensor_image'])
            shutil.copy2(src, os.path.join(img_dir, row['sensor_image']))

        shutil.copy2(
            os.path.join(collect_dir, 'collection_params.json'),
            os.path.join(split_dir, 'collection_params.json')
        )

        print(f"  {split_name}: {len(split_df)} samples")

    return int(train_ratio * n), n - int(train_ratio * n)


def main():
    parser = argparse.ArgumentParser(description='Train ridge_4d CNN')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--model', choices=['nature_cnn', 'resnet'], default='nature_cnn')
    args = parser.parse_args()

    device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
    print(f"Device: {device}")

    # --- Step 1: Find dataset and split ---
    collect_dir = find_latest_dataset()
    print(f"\nDataset: {collect_dir}")
    n_train, n_val = split_dataset(collect_dir)
    print(f"  Split: {n_train} train / {n_val} val\n")

    # --- Step 2: Train ---
    task = 'ridge_4d'

    if args.model == 'nature_cnn':
        model_params = {
            'model_type': 'nature_cnn',
            'model_kwargs': {
                'fc_layers': [512, 512],
                'dropout': 0.0,
            },
        }
    else:
        model_params = {
            'model_type': 'resnet',
            'model_kwargs': {
                'layers': [2, 2, 2, 2],
            },
        }

    learning_params = {
        'seed': 42,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
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

    save_dir = os.path.join(MODEL_BASE, args.model)
    os.makedirs(save_dir, exist_ok=True)
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))
    save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
    save_json_obj(image_processing_params, os.path.join(save_dir, 'image_processing_params'))
    save_json_obj(augmentation_params, os.path.join(save_dir, 'augmentation_params'))
    print(f"Model: {args.model} -> {save_dir}")

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

    print(f"\nDone! Model saved to: {save_dir}")


if __name__ == '__main__':
    main()
