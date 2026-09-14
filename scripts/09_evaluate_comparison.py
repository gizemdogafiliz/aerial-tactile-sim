"""
Hybrid evaluation: Radon (distance/orient) + CNN (depth/yaw) on the same val set.

Compares 4 approaches:
  1. CNN-only (NatureCNN)
  2. CNN-only (ResNet)
  3. Radon-only
  4. Hybrid: Radon distance + CNN depth/yaw + best orient

Produces:
  - Per-output MAE bar chart
  - Scatter plots (pred vs gt) for each method
  - Summary table

Usage:
  conda activate tactile
  python 09_evaluate_comparison.py
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
from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, 'output', 'ridge_dataset')
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


def detect_ridge_radon(img_gray):
    """Radon detection — same as 08_hough_ridge_detect.py"""
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
    """Load trained CNN model and its params."""
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


def run_cnn_on_val(model, label_names, limits):
    """Run CNN on val set, return decoded predictions."""
    val_dir = os.path.join(DATA_DIR, 'val')
    targets = pd.read_csv(os.path.join(val_dir, 'targets.csv'))
    img_dir = os.path.join(val_dir, 'images')

    predictions = {'y': [], 'z': [], 'Rx': [], 'Rz': []}
    ground_truth = {'y': [], 'z': [], 'Rx': [], 'Rz': []}

    for _, row in targets.iterrows():
        img_path = os.path.join(img_dir, row['sensor_image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(img_t)

        decoded = decode_pose(output, label_names, limits)

        for key in predictions:
            predictions[key].append(decoded[key].item())

        ground_truth['y'].append(row['pose_2'])
        ground_truth['z'].append(row['pose_3'])
        ground_truth['Rx'].append(row['pose_4'])
        ground_truth['Rz'].append(row['pose_6'])

    return ground_truth, predictions


def run_radon_on_val():
    """Run Radon on val set images."""
    val_dir = os.path.join(DATA_DIR, 'val')
    targets = pd.read_csv(os.path.join(val_dir, 'targets.csv'))
    img_dir = os.path.join(val_dir, 'images')

    predictions = {'abs_distance': [], 'orient': []}
    ground_truth = {'abs_distance': [], 'orient': []}

    for _, row in targets.iterrows():
        img_path = os.path.join(img_dir, row['sensor_image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        det = detect_ridge_radon(img)
        predictions['abs_distance'].append(det['abs_distance_mm'])
        predictions['orient'].append(det['ridge_orient_deg'])

        ground_truth['abs_distance'].append(abs(row['pose_2']))
        ground_truth['orient'].append(row['pose_4'])

    return ground_truth, predictions


def compute_metrics(gt, pred):
    """Compute MAE for each output."""
    gt = np.array(gt)
    pred = np.array(pred)
    return np.abs(gt - pred).mean()


def orient_accuracy(gt, pred):
    """Binary orientation accuracy."""
    gt = np.array(gt)
    pred = np.array(pred)
    gt_bin = (gt < 45).astype(int)
    pred_bin = (pred < 45).astype(int)
    return (gt_bin == pred_bin).mean()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading models...")
    ncnn_model, ncnn_labels, ncnn_limits = load_cnn_model('nature_cnn')
    res_model, res_labels, res_limits = load_cnn_model('resnet')

    print("Running NatureCNN on val set...")
    ncnn_gt, ncnn_pred = run_cnn_on_val(ncnn_model, ncnn_labels, ncnn_limits)

    if res_model is not None:
        print("Running ResNet on val set...")
        res_gt, res_pred = run_cnn_on_val(res_model, res_labels, res_limits)
    else:
        print("ResNet model not found, skipping.")
        res_gt, res_pred = None, None

    print("Running Radon on val set...")
    radon_gt, radon_pred = run_radon_on_val()

    # --- Compute metrics ---
    results = {}

    # NatureCNN
    results['NatureCNN'] = {
        '|distance| (mm)': compute_metrics(np.abs(ncnn_gt['y']), np.abs(ncnn_pred['y'])),
        'depth (mm)': compute_metrics(ncnn_gt['z'], ncnn_pred['z']),
        'orient (°)': compute_metrics(ncnn_gt['Rx'], ncnn_pred['Rx']),
        'yaw (°)': compute_metrics(ncnn_gt['Rz'], ncnn_pred['Rz']),
        'orient_acc': orient_accuracy(ncnn_gt['Rx'], ncnn_pred['Rx']),
    }

    # ResNet
    if res_pred is not None:
        results['ResNet'] = {
            '|distance| (mm)': compute_metrics(np.abs(res_gt['y']), np.abs(res_pred['y'])),
            'depth (mm)': compute_metrics(res_gt['z'], res_pred['z']),
            'orient (°)': compute_metrics(res_gt['Rx'], res_pred['Rx']),
            'yaw (°)': compute_metrics(res_gt['Rz'], res_pred['Rz']),
            'orient_acc': orient_accuracy(res_gt['Rx'], res_pred['Rx']),
        }

    # Radon
    results['Radon'] = {
        '|distance| (mm)': compute_metrics(radon_gt['abs_distance'], radon_pred['abs_distance']),
        'depth (mm)': None,
        'orient (°)': None,
        'yaw (°)': None,
        'orient_acc': orient_accuracy(radon_gt['orient'], radon_pred['orient']),
    }

    # Hybrid: Radon distance + best CNN's depth/yaw + Radon orient (straight) / CNN orient (corner)
    best_cnn_name = 'NatureCNN'
    best_cnn_pred = ncnn_pred
    best_cnn_gt = ncnn_gt
    if res_pred is not None:
        ncnn_d_mae = results['NatureCNN']['depth (mm)']
        res_d_mae = results['ResNet']['depth (mm)']
        if res_d_mae < ncnn_d_mae:
            best_cnn_name = 'ResNet'
            best_cnn_pred = res_pred
            best_cnn_gt = res_gt

    results[f'Hybrid (Radon+{best_cnn_name})'] = {
        '|distance| (mm)': results['Radon']['|distance| (mm)'],
        'depth (mm)': results[best_cnn_name]['depth (mm)'],
        'orient (°)': results[best_cnn_name]['orient (°)'],
        'yaw (°)': results[best_cnn_name]['yaw (°)'],
        'orient_acc': results['Radon']['orient_acc'],
    }

    # --- Print summary table ---
    print(f"\n{'='*75}")
    print(f"  COMPARISON — Val set ({len(ncnn_gt['y'])} samples)")
    print(f"{'='*75}")
    print(f"\n  {'Method':<25} {'|dist| mm':>10} {'depth mm':>10} {'orient °':>10} {'yaw °':>10} {'orient%':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, m in results.items():
        dist = f"{m['|distance| (mm)']:.2f}" if m['|distance| (mm)'] is not None else "--"
        dep = f"{m['depth (mm)']:.2f}" if m['depth (mm)'] is not None else "--"
        ori = f"{m['orient (°)']:.2f}" if m['orient (°)'] is not None else "--"
        yaw = f"{m['yaw (°)']:.2f}" if m['yaw (°)'] is not None else "--"
        acc = f"{m['orient_acc']*100:.1f}%" if m['orient_acc'] is not None else "--"
        print(f"  {name:<25} {dist:>10} {dep:>10} {ori:>10} {yaw:>10} {acc:>10}")

    # --- Plot 1: MAE bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['|distance| (mm)', 'depth (mm)', 'yaw (°)']
    x = np.arange(len(metrics))
    width = 0.18
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for i, (name, m) in enumerate(results.items()):
        vals = []
        for metric in metrics:
            v = m[metric]
            vals.append(v if v is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i])
        for bar, v, raw in zip(bars, vals, [m[metric] for metric in metrics]):
            if raw is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(results)-1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('MAE')
    ax.set_title('Ridge Detection: Method Comparison (MAE)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_bar.png'), dpi=150)
    print(f"\n  Bar chart saved: output/evaluation/comparison_bar.png")

    # --- Plot 2: Scatter plots ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    scatter_data = [
        ('NatureCNN', ncnn_gt, ncnn_pred),
    ]
    if res_pred is not None:
        scatter_data.append(('ResNet', res_gt, res_pred))

    outputs = [
        ('y', '|distance|', 'mm', True),
        ('z', 'depth', 'mm', False),
        ('Rx', 'orient', '°', False),
        ('Rz', 'yaw', '°', False),
    ]

    for row_idx, (cnn_name, gt, pred) in enumerate(scatter_data):
        for col_idx, (key, label, unit, use_abs) in enumerate(outputs):
            ax = axes[row_idx, col_idx]
            gt_arr = np.abs(np.array(gt[key])) if use_abs else np.array(gt[key])
            pred_arr = np.abs(np.array(pred[key])) if use_abs else np.array(pred[key])
            mae = np.abs(gt_arr - pred_arr).mean()

            ax.scatter(gt_arr, pred_arr, s=3, alpha=0.3, c='steelblue')
            lims = [min(gt_arr.min(), pred_arr.min()), max(gt_arr.max(), pred_arr.max())]
            ax.plot(lims, lims, 'r-', linewidth=1)
            ax.set_xlabel(f'target {label}')
            ax.set_ylabel(f'predicted {label}')
            ax.set_title(f'{cnn_name}: {label} (MAE={mae:.2f}{unit})')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.3)

    # Add Radon distance scatter to both rows
    for row_idx in range(min(2, len(scatter_data))):
        ax = axes[row_idx, 0]
        radon_gt_arr = np.array(radon_gt['abs_distance'])
        radon_pred_arr = np.array(radon_pred['abs_distance'])
        ax.scatter(radon_gt_arr, radon_pred_arr, s=3, alpha=0.3, c='orange', label='Radon')
        ax.legend(fontsize=7)

    if len(scatter_data) < 2:
        for col_idx in range(4):
            axes[1, col_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_scatter.png'), dpi=150)
    print(f"  Scatter plots saved: output/evaluation/comparison_scatter.png")

    # --- Plot 3: Orientation confusion ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    orient_methods = [
        ('Radon', radon_gt['orient'], radon_pred['orient']),
        ('NatureCNN', ncnn_gt['Rx'], ncnn_pred['Rx']),
    ]
    if res_pred is not None:
        orient_methods.append(('ResNet', res_gt['Rx'], res_pred['Rx']))

    for idx, (name, gt, pred) in enumerate(orient_methods):
        gt_bin = (np.array(gt) < 45).astype(int)
        pred_bin = (np.array(pred) < 45).astype(int)
        tp = ((gt_bin == 1) & (pred_bin == 1)).sum()
        tn = ((gt_bin == 0) & (pred_bin == 0)).sum()
        fp = ((gt_bin == 0) & (pred_bin == 1)).sum()
        fn = ((gt_bin == 1) & (pred_bin == 0)).sum()
        cm = np.array([[tn, fp], [fn, tp]])
        acc = (tp + tn) / (tp + tn + fp + fn)

        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Vert(90°)', 'Horiz(0°)'])
        ax.set_yticklabels(['Vert(90°)', 'Horiz(0°)'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title(f'{name} Orient ({acc*100:.1f}%)')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

    if len(orient_methods) < 3:
        axes[2].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_orient.png'), dpi=150)
    print(f"  Orientation plots saved: output/evaluation/comparison_orient.png")

    print(f"\n  Done!")


if __name__ == '__main__':
    main()
