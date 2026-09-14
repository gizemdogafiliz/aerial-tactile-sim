"""
Phase 4 (alt): Classical ridge detection via Hough Transform.

Detects ridge orientation (0°/90°) and signed distance from sensor center
using Hough Transform on tactile images — no CNN, no training data needed.

Outputs per image:
  rho     — perpendicular distance from image center to ridge line (pixels → mm)
  theta   — ridge orientation (≈0° horizontal, ≈90° vertical)
  depth   — NOT estimated (needs CNN or separate calibration)

Usage:
  conda activate tactile
  python 08_hough_ridge_detect.py                        # evaluate on latest dataset
  python 08_hough_ridge_detect.py --show 20              # visualize 20 samples
  python 08_hough_ridge_detect.py --image path/to/img    # single image
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, 'output', 'ridge_dataset')

# TacTip image: 128x128, dome ≈ 100px diameter, ~32mm → 0.32 mm/px
IMG_SIZE = 128
DOME_RADIUS_PX = 50
MM_PER_PX = 32.0 / (2 * DOME_RADIUS_PX)  # ~0.32 mm/px


def detect_ridge_hough(img_gray, return_debug=False):
    """Detect dominant ridge line using Hough Transform.

    Returns (rho_mm, theta_deg, confidence) or None if no line found.
    rho_mm: signed perpendicular distance from image center to ridge (mm).
    theta_deg: ridge orientation (0°=horizontal, 90°=vertical).
    """
    h, w = img_gray.shape
    cx, cy = w // 2, h // 2

    # Circular mask to exclude sensor border
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), DOME_RADIUS_PX, 255, -1)
    masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)

    # Blur to reduce pin noise
    blurred = cv2.GaussianBlur(masked, (5, 5), 1.5)

    # Canny edge detection — ridge appears as bright line
    med = np.median(blurred[mask > 0])
    edges = cv2.Canny(blurred, int(med * 0.5), int(med * 1.0))
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Hough Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=20)

    if lines is None:
        if return_debug:
            return None, edges
        return None

    # Take the strongest line
    rho_px, theta_rad = lines[0][0]

    # Convert: Hough theta is angle of the normal to the line
    # Ridge orientation = theta + 90° (perpendicular to normal)
    ridge_angle_deg = np.degrees(theta_rad)

    # rho is distance from origin (top-left corner) to line.
    # Shift to image center: signed distance = rho - (cx*cos(theta) + cy*sin(theta))
    rho_from_center = rho_px - (cx * np.cos(theta_rad) + cy * np.sin(theta_rad))
    rho_mm = rho_from_center * MM_PER_PX

    # TacTip perpendicular mapping: bright band in image is PERPENDICULAR to physical ridge.
    # Hough detects the bright band direction, not the ridge direction.
    #   theta ≈ 0°  → normal along x → detected line is vertical → ridge is HORIZONTAL (0°)
    #   theta ≈ 90° → normal along y → detected line is horizontal → ridge is VERTICAL (90°)
    ridge_orient_deg = 90.0 if ridge_angle_deg >= 45 else 0.0

    confidence = len(lines)

    result = {
        'rho_mm': rho_mm,
        'ridge_orient_deg': ridge_orient_deg,
        'hough_theta_deg': ridge_angle_deg,
        'confidence': confidence,
    }

    if return_debug:
        return result, edges
    return result


def detect_ridge_radon(img_gray):
    """Detect dominant ridge using gradient energy + single-angle Radon projection.

    TacTip physics: dome wraps around ridge, creating a bright band PERPENDICULAR
    to the physical ridge direction. So:
      - Horizontal ridge (0°) → vertical bright band → high horizontal gradient energy
      - Vertical ridge (90°) → horizontal bright band → high vertical gradient energy

    Orientation: gradient energy comparison (Sobel gx² vs gy²).
    Distance: column projection at 0° (horiz ridge) or 90° rotation (vert ridge).
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

    # Gradient energy for orientation
    gx = cv2.Sobel(ridge_signal, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(ridge_signal, cv2.CV_64F, 0, 1, ksize=5)
    energy_x = (gx ** 2).sum()
    energy_y = (gy ** 2).sum()

    # High gx → vertical band in image → HORIZONTAL ridge (perpendicular mapping)
    if energy_x > energy_y:
        ridge_orient_deg = 0.0
        theta = 0
    else:
        ridge_orient_deg = 90.0
        theta = 90

    # Single-angle Radon: rotate image by -theta, then column projection
    rotated = cv2.warpAffine(
        ridge_signal,
        cv2.getRotationMatrix2D((cx, cy), -theta, 1.0),
        (w, h)
    )
    projection = rotated.sum(axis=0)
    rho_px = projection.argmax() - cx
    rho_mm = rho_px * MM_PER_PX

    confidence = max(energy_x, energy_y) / (min(energy_x, energy_y) + 1e-6)

    return {
        'rho_mm': rho_mm,
        'ridge_orient_deg': ridge_orient_deg,
        'energy_ratio': confidence,
        'confidence': confidence,
    }


def evaluate_dataset(collect_dir, method='hough', max_samples=None, show=0):
    """Evaluate Hough/Radon on collected dataset against ground-truth labels."""
    labels = pd.read_csv(os.path.join(collect_dir, 'labels.csv'))
    img_dir = os.path.join(collect_dir, 'images')

    if max_samples:
        labels = labels.iloc[:max_samples]

    results = []
    n_fail = 0

    for idx, row in labels.iterrows():
        img_path = os.path.join(img_dir, row['sensor_image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        gt_signed_d = row['signed_d_nearest_mm']
        gt_orient = row.get('ridge_orient_deg',
                           0.0 if row['nearest_ridge'] in [0, 2] else 90.0)
        gt_depth = row['depth_mm']

        if method == 'hough':
            det, edges = detect_ridge_hough(img, return_debug=True)
        else:
            det = detect_ridge_radon(img)
            edges = None

        if det is None:
            n_fail += 1
            continue

        results.append({
            'gt_signed_d': gt_signed_d,
            'gt_orient': gt_orient,
            'gt_depth': gt_depth,
            'pred_rho_mm': det['rho_mm'],
            'pred_orient': det['ridge_orient_deg'],
            'is_corner': row['is_corner'],
            'side': row['side'],
        })

        if show > 0 and idx < show:
            visualize(img, edges, det, gt_signed_d, gt_orient, idx)

    if not results:
        print("No detections!")
        return

    df = pd.DataFrame(results)

    # Orientation accuracy (binary: horizontal vs vertical)
    gt_is_horiz = (df['gt_orient'] < 45).astype(int)
    pred_is_horiz = (df['pred_orient'] < 45).astype(int)
    orient_acc = (gt_is_horiz == pred_is_horiz).mean()

    # signed_d error (only for correct orientation predictions)
    correct_orient = gt_is_horiz == pred_is_horiz
    d_err_all = (df['pred_rho_mm'] - df['gt_signed_d']).abs()
    d_err_correct = d_err_all[correct_orient]

    # Breakdown: straight vs corner
    straight = df['is_corner'] == 0
    corner = df['is_corner'] == 1

    # Absolute distance error (sign depends on approach direction, not detectable from image)
    abs_d_err = (df['pred_rho_mm'].abs() - df['gt_signed_d'].abs()).abs()

    print(f"\n{'='*60}")
    print(f"  {method.upper()} Ridge Detection — {len(df)} samples ({n_fail} failed)")
    print(f"{'='*60}")
    print(f"\n  Orientation accuracy:  {orient_acc*100:.1f}%")
    print(f"    Straight:            {(gt_is_horiz[straight] == pred_is_horiz[straight]).mean()*100:.1f}%")
    print(f"    Corner:              {(gt_is_horiz[corner] == pred_is_horiz[corner]).mean()*100:.1f}%")
    print(f"\n  |distance| MAE (all):            {abs_d_err.mean():.2f} mm")
    print(f"  |distance| MAE (straight):       {abs_d_err[straight].mean():.2f} mm")
    print(f"  |distance| MAE (corner):         {abs_d_err[corner].mean():.2f} mm")
    print(f"  |distance| MAE (correct orient): {abs_d_err[correct_orient].mean():.2f} mm")
    print(f"\n  signed_d MAE (all):              {d_err_all.mean():.2f} mm")
    print(f"  signed_d MAE (correct orient):   {d_err_correct.mean():.2f} mm")
    print(f"\n  Note: sign of distance depends on approach direction (known from drone pose)")
    print()

    return df


def visualize(img, edges, det, gt_d, gt_orient, idx):
    """Show detection result on a single image."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    cx, cy = w // 2, h // 2

    if det is not None:
        # Draw detected line
        theta = np.radians(det.get('hough_theta_deg', 90 - det['ridge_orient_deg']))
        rho = det['rho_mm'] / MM_PER_PX + cx * np.cos(theta) + cy * np.sin(theta)
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

    label = f"gt_d={gt_d:.1f} pred={det['rho_mm']:.1f}" if det else "FAIL"
    cv2.putText(vis, label, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    orient_label = f"gt_or={gt_orient:.0f} pred={det['ridge_orient_deg']:.0f}" if det else ""
    cv2.putText(vis, orient_label, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    cv2.imshow(f"Sample {idx}", vis)
    if edges is not None:
        cv2.imshow(f"Edges {idx}", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def single_image(path, method='hough'):
    """Run on a single image and print result."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot read: {path}")
        return

    if method == 'hough':
        det, edges = detect_ridge_hough(img, return_debug=True)
    else:
        det = detect_ridge_radon(img)

    if det is None:
        print("No ridge detected!")
        return

    print(f"rho = {det['rho_mm']:.2f} mm (distance from center)")
    print(f"ridge orientation = {det['ridge_orient_deg']:.1f}°  (0°=horiz, 90°=vert)")


def main():
    parser = argparse.ArgumentParser(description='Hough/Radon ridge detection on tactile images')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--method', choices=['hough', 'radon'], default='hough')
    parser.add_argument('--max', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--show', type=int, default=0, help='Visualize N samples')
    args = parser.parse_args()

    if args.image:
        single_image(args.image, args.method)
        return

    # Find latest dataset
    pattern = os.path.join(DATASET_DIR, 'collect_*')
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"No dataset found in {DATASET_DIR}")
        return
    collect_dir = dirs[-1]
    print(f"Dataset: {collect_dir}")

    evaluate_dataset(collect_dir, method=args.method, max_samples=args.max, show=args.show)


if __name__ == '__main__':
    main()
