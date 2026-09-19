"""
Hybrid v2: CNN orient + Radon distance evaluation.

Instead of using Radon's own orient detection (Sobel energy),
feed the CNN's orient prediction into Radon's single-angle projection
for distance estimation.

Hypothesis: CNN orient (99.0%) > Radon orient (97.1%),
so distance accuracy should also improve on corner samples
where Radon gets orient wrong.

Usage:
  conda activate tactile
  python 11_hybrid_cnn_orient_radon_distance.py
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from tactile_gym.utils.general_utils import load_json_obj
from tactile_gym_servo_control.learning.networks import create_model
from tactile_gym_servo_control.learning.learning_utils import (
    import_task, decode_pose, get_pose_limits
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(
    PROJECT_DIR, 'servo_control', 'tactile_gym_servo_control', 'data', 'ridge_4d', 'tap'
)
MODEL_BASE = os.path.join(
    PROJECT_DIR, 'servo_control', 'tactile_gym_servo_control', 'learned_models', 'ridge_4d'
)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'evaluation')

IMG_SIZE = 128
DOME_RADIUS_PX = 50
MM_PER_PX = 32.0 / (2 * DOME_RADIUS_PX)


def radon_distance_with_orient(img_gray, orient_deg):
    """Radon distance using externally provided orientation.

    Same projection as detect_ridge_radon() but skips Sobel orient detection,
    uses the given orient_deg instead.
    """
    h, w = img_gray.shape
    cx, cy = w // 2, h // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), DOME_RADIUS_PX, 255, -1)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5).astype(np.float32)
    blurred[mask == 0] = 0

    bg = cv2.GaussianBlur(img_gray, (31, 31), 15).astype(np.float32)
    bg[mask == 0] = 0
    ridge_signal = np.clip(blurred - bg, 0, 255)

    theta = 0 if orient_deg < 45 else 90

    rotated = cv2.warpAffine(
        ridge_signal,
        cv2.getRotationMatrix2D((cx, cy), -theta, 1.0),
        (w, h)
    )
    projection = rotated.sum(axis=0)
    rho_px = projection.argmax() - cx
    rho_mm = rho_px * MM_PER_PX

    return abs(rho_mm)


def detect_ridge_radon(img_gray):
    """Original Radon: own orient + own distance."""
    h, w = img_gray.shape
    cx, cy = w // 2, h // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), DOME_RADIUS_PX, 255, -1)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5).astype(np.float32)
    blurred[mask == 0] = 0

    bg = cv2.GaussianBlur(img_gray, (31, 31), 15).astype(np.float32)
    bg[mask == 0] = 0
    ridge_signal = np.clip(blurred - bg, 0, 255)

    gx = cv2.Sobel(ridge_signal, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(ridge_signal, cv2.CV_64F, 0, 1, ksize=5)
    energy_x = (gx ** 2).sum()
    energy_y = (gy ** 2).sum()

    if energy_x > energy_y:
        ridge_orient_deg = 0.0
        theta = 0
    else:
        ridge_orient_deg = 90.0
        theta = 90

    rotated = cv2.warpAffine(
        ridge_signal,
        cv2.getRotationMatrix2D((cx, cy), -theta, 1.0),
        (w, h)
    )
    projection = rotated.sum(axis=0)
    rho_px = projection.argmax() - cx
    rho_mm = rho_px * MM_PER_PX

    return {
        'abs_distance_mm': abs(rho_mm),
        'ridge_orient_deg': ridge_orient_deg,
    }


def load_cnn_model(model_name):
    model_dir = os.path.join(MODEL_BASE, model_name)
    if not os.path.exists(os.path.join(model_dir, 'best_model.pth')):
        return None, None, None

    model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
    image_params = load_json_obj(os.path.join(model_dir, 'image_processing_params'))
    pose_limits = load_json_obj(os.path.join(model_dir, 'pose_limits'))

    out_dim, label_names = import_task('ridge_4d')
    model = create_model(
        image_params['dims'], out_dim, model_params,
        saved_model_dir=model_dir, device='cpu'
    )
    model.eval()

    limits = (pose_limits['pose_llims'], pose_limits['pose_ulims'])
    return model, label_names, limits


def orient_accuracy(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    gt_bin = (gt < 45).astype(int)
    pred_bin = (pred < 45).astype(int)
    return (gt_bin == pred_bin).mean()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading NatureCNN...")
    model, label_names, limits = load_cnn_model('nature_cnn')
    if model is None:
        print("NatureCNN not found!")
        return

    val_dir = os.path.join(DATA_DIR, 'val')
    targets = pd.read_csv(os.path.join(val_dir, 'targets.csv'))
    img_dir = os.path.join(val_dir, 'images')

    gt_abs_dist = []
    gt_orient = []

    cnn_orient_pred = []
    radon_orient_pred = []

    radon_only_dist = []
    hybrid_v2_dist = []
    cnn_only_dist = []

    print(f"Evaluating {len(targets)} val samples...")

    for _, row in targets.iterrows():
        img_path = os.path.join(img_dir, row['sensor_image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Ground truth
        gt_abs_d = abs(row['pose_2'])
        gt_or = row['pose_4']
        gt_abs_dist.append(gt_abs_d)
        gt_orient.append(gt_or)

        # CNN prediction
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            output = model(img_t)
        decoded = decode_pose(output, label_names, limits)
        cnn_or = decoded['Rx'].item()
        cnn_dist = abs(decoded['y'].item())
        cnn_orient_pred.append(cnn_or)
        cnn_only_dist.append(cnn_dist)

        # Radon-only (original)
        radon_det = detect_ridge_radon(img)
        radon_orient_pred.append(radon_det['ridge_orient_deg'])
        radon_only_dist.append(radon_det['abs_distance_mm'])

        # Hybrid v2: CNN orient → Radon distance
        hybrid_dist = radon_distance_with_orient(img, cnn_or)
        hybrid_v2_dist.append(hybrid_dist)

    gt_abs_dist = np.array(gt_abs_dist)
    gt_orient = np.array(gt_orient)
    cnn_orient_pred = np.array(cnn_orient_pred)
    radon_orient_pred = np.array(radon_orient_pred)
    radon_only_dist = np.array(radon_only_dist)
    hybrid_v2_dist = np.array(hybrid_v2_dist)
    cnn_only_dist = np.array(cnn_only_dist)

    n = len(gt_abs_dist)

    # Orient accuracy
    cnn_orient_acc = orient_accuracy(gt_orient, cnn_orient_pred)
    radon_orient_acc = orient_accuracy(gt_orient, radon_orient_pred)

    # Distance MAE
    radon_dist_mae = np.abs(gt_abs_dist - radon_only_dist).mean()
    hybrid_v2_dist_mae = np.abs(gt_abs_dist - hybrid_v2_dist).mean()
    cnn_dist_mae = np.abs(gt_abs_dist - cnn_only_dist).mean()

    # Corner vs straight breakdown
    labels_csv = os.path.join(
        PROJECT_DIR, 'output', 'ridge_dataset',
        sorted(os.listdir(os.path.join(PROJECT_DIR, 'output', 'ridge_dataset')))[-1],
        'labels.csv'
    )
    labels_df = pd.read_csv(labels_csv)
    val_start = len(labels_df) - n
    is_corner = labels_df['is_corner'].values[val_start:]

    straight = is_corner == 0
    corner = is_corner == 1

    print(f"\n{'='*70}")
    print(f"  HYBRID v2 EVALUATION — {n} val samples")
    print(f"{'='*70}")

    print(f"\n  Orient accuracy:")
    print(f"    Radon (Sobel energy):   {radon_orient_acc*100:.1f}%")
    print(f"    NatureCNN:              {cnn_orient_acc*100:.1f}%")

    print(f"\n  Distance MAE (|distance|):")
    print(f"    {'Method':<35} {'All':>8} {'Straight':>10} {'Corner':>10}")
    print(f"    {'-'*35} {'-'*8} {'-'*10} {'-'*10}")

    for name, dist_pred in [
        ('Radon-only (Radon orient)', radon_only_dist),
        ('Hybrid v2 (CNN orient + Radon dist)', hybrid_v2_dist),
        ('CNN-only (NatureCNN)', cnn_only_dist),
    ]:
        mae_all = np.abs(gt_abs_dist - dist_pred).mean()
        mae_str = np.abs(gt_abs_dist[straight] - dist_pred[straight]).mean()
        mae_cor = np.abs(gt_abs_dist[corner] - dist_pred[corner]).mean()
        print(f"    {name:<35} {mae_all:>7.2f}mm {mae_str:>9.2f}mm {mae_cor:>9.2f}mm")

    # Where CNN orient differs from Radon orient
    cnn_bin = (cnn_orient_pred < 45).astype(int)
    radon_bin = (radon_orient_pred < 45).astype(int)
    differ = cnn_bin != radon_bin
    n_differ = differ.sum()
    print(f"\n  Samples where CNN and Radon orient disagree: {n_differ} ({n_differ/n*100:.1f}%)")
    if n_differ > 0:
        gt_bin = (gt_orient < 45).astype(int)
        cnn_correct_on_differ = (cnn_bin[differ] == gt_bin[differ]).sum()
        radon_correct_on_differ = (radon_bin[differ] == gt_bin[differ]).sum()
        print(f"    CNN correct:   {cnn_correct_on_differ}/{n_differ}")
        print(f"    Radon correct: {radon_correct_on_differ}/{n_differ}")

        mae_radon_differ = np.abs(gt_abs_dist[differ] - radon_only_dist[differ]).mean()
        mae_hybrid_differ = np.abs(gt_abs_dist[differ] - hybrid_v2_dist[differ]).mean()
        print(f"    Radon dist MAE on these: {mae_radon_differ:.2f}mm")
        print(f"    Hybrid v2 dist MAE on these: {mae_hybrid_differ:.2f}mm")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, dist_pred, color) in zip(axes, [
        ('Radon-only', radon_only_dist, '#4C72B0'),
        ('Hybrid v2 (CNN orient)', hybrid_v2_dist, '#C44E52'),
        ('CNN-only', cnn_only_dist, '#55A868'),
    ]):
        mae = np.abs(gt_abs_dist - dist_pred).mean()
        ax.scatter(gt_abs_dist, dist_pred, s=3, alpha=0.3, c=color)
        lims = [0, max(gt_abs_dist.max(), dist_pred.max())]
        ax.plot(lims, lims, 'r-', linewidth=1)
        ax.set_xlabel('Ground truth |distance| (mm)')
        ax.set_ylabel('Predicted |distance| (mm)')
        ax.set_title(f'{name}\nMAE = {mae:.2f}mm')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'hybrid_v2_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved: {out_path}")
    print(f"  Done!")


if __name__ == '__main__':
    main()
