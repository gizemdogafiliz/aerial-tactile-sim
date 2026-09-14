"""
Phase 3A: Collect tactile dataset for ridge detection.

TacTip sensor presses against a wall with a 120x120mm rectangular ridge frame
(4 ridges, 4 corners) — matching the Gazebo drone scenario geometry.

Samples are drawn uniformly along the frame perimeter:
  ~70% land on straight ridge sections (single edge visible)
  ~30% land near corners (two ridges visible)

Each sample: (tactile_image, labels) where labels encode the sensor's
position relative to the nearest ridge(s).

Usage:
  conda activate tactile
  python 06_collect_ridge_data.py                      # 20 samples, GUI
  python 06_collect_ridge_data.py --n 5000 --no-gui    # full dataset
  python 06_collect_ridge_data.py --n 60 --record       # GUI + video
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd
import json
import time

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
STIMULI_DIR = os.path.join(PROJECT_DIR, 'servo_control', 'tactile_gym_servo_control', 'stimuli')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'ridge_dataset')

RIDGE_W = 5.0       # ridge cross-section width in mm
FRAME_HALF = 60.0   # rectangle half-size in mm (120x120mm frame)
DOME_R = 16.0        # TacTip dome radius in mm
CORNER_THRESH = DOME_R + RIDGE_W / 2  # 18.5mm — dome can see both ridges


def sample_perimeter_poses(num_samples):
    """Sample poses sequentially along the rectangular ridge frame perimeter.

    Sensor walks the full perimeter (bottom -> right -> top -> left) with
    evenly spaced positions. At each position, pose variation (cross-offset,
    depth, angles) is randomized. Order is sequential for intuitive video;
    the DataLoader shuffles during CNN training.

    Rectangle: 120x120mm, ridges at wf_x=+-60mm and wf_y=+-60mm.
    Workframe mapping: wf_x -> world_y, wf_y -> world_x (yaw=pi/2).

    Returns (poses (N,6), sides (N,)) where side: 0=bottom,1=right,2=top,3=left.
    """
    side_len = 2 * FRAME_HALF
    perimeter = 4 * side_len

    t = np.linspace(0, perimeter, num_samples, endpoint=False)
    sides = np.clip((t // side_len).astype(int), 0, 3)
    s = (t % side_len) - FRAME_HALF

    cross = np.random.uniform(-14, 14, num_samples)

    wf_x = np.zeros(num_samples)
    wf_y = np.zeros(num_samples)

    # CW from bottom-left corner: left(B→T) → top(L→R) → right(T→B) → bottom(R→L)

    # Side 0: left — ridge at wf_y=-60, wf_x from -60→60
    m = sides == 0
    wf_x[m] = s[m]
    wf_y[m] = -FRAME_HALF + cross[m]

    # Side 1: top — ridge at wf_x=+60, wf_y from -60→60
    m = sides == 1
    wf_x[m] = FRAME_HALF + cross[m]
    wf_y[m] = s[m]

    # Side 2: right — ridge at wf_y=+60, wf_x from 60→-60
    m = sides == 2
    wf_x[m] = -s[m]
    wf_y[m] = FRAME_HALF + cross[m]

    # Side 3: bottom — ridge at wf_x=-60, wf_y from 60→-60
    m = sides == 3
    wf_x[m] = -FRAME_HALF + cross[m]
    wf_y[m] = -s[m]

    depth = np.random.uniform(-2.0, 5.5, num_samples)
    roll = np.clip(np.random.normal(0, 5, num_samples), -15, 15)
    pitch = np.clip(np.random.normal(0, 5, num_samples), -15, 15)
    yaw = np.clip(np.random.normal(0, 8, num_samples), -20, 20)

    poses = np.column_stack([wf_x, wf_y, depth, roll, pitch, yaw])

    return poses, sides


def compute_labels(poses_mm, sides):
    """Compute ground-truth labels for rectangular frame.

    Distances to all 4 ridges, nearest ridge, signed distance, corner flag.
    """
    wf_x = poses_mm[:, 0]
    wf_y = poses_mm[:, 1]
    depth = poses_mm[:, 2]

    d_bottom = np.abs(wf_x - (-FRAME_HALF))
    d_top = np.abs(wf_x - FRAME_HALF)
    d_left = np.abs(wf_y - (-FRAME_HALF))
    d_right = np.abs(wf_y - FRAME_HALF)

    d_all = np.stack([d_bottom, d_right, d_top, d_left], axis=1)
    nearest_id = np.argmin(d_all, axis=1)
    d_nearest = np.min(d_all, axis=1)

    d_sorted = np.sort(d_all, axis=1)
    d_second = d_sorted[:, 1]

    # Signed distance to nearest ridge (positive = inside frame)
    signed_d = np.where(
        nearest_id == 0, wf_x + FRAME_HALF,       # bottom: inside = wf_x > -60
        np.where(nearest_id == 1, FRAME_HALF - wf_y,   # right: inside = wf_y < 60
                 np.where(nearest_id == 2, FRAME_HALF - wf_x,  # top: inside = wf_x < 60
                          wf_y + FRAME_HALF)))            # left: inside = wf_y > -60

    is_corner = (d_second < CORNER_THRESH).astype(int)

    # Ridge orientation: 0° = horizontal (x-ridges at wf_x=±60, run along wf_y)
    #                   90° = vertical   (y-ridges at wf_y=±60, run along wf_x)
    ridge_orient = np.where(
        (nearest_id == 0) | (nearest_id == 2), 0.0, 90.0
    )

    return {
        'wf_x_mm': wf_x, 'wf_y_mm': wf_y, 'depth_mm': depth,
        'roll_deg': poses_mm[:, 3], 'pitch_deg': poses_mm[:, 4], 'yaw_deg': poses_mm[:, 5],
        'd_bottom_mm': d_bottom, 'd_top_mm': d_top,
        'd_left_mm': d_left, 'd_right_mm': d_right,
        'd_nearest_mm': d_nearest,
        'signed_d_nearest_mm': signed_d,
        'nearest_ridge': nearest_id,
        'd_second_nearest_mm': d_second,
        'is_corner': is_corner,
        'side': sides.astype(int),
        'ridge_orient_deg': ridge_orient,
    }


def make_target_df(labels):
    """Create targets.csv with CNN-relevant values in servo_control format.

    Mapping (what CNN will predict):
      pose_1 (x)  = 0 (unused)
      pose_2 (y)  = signed_d_nearest_mm  (cross-ridge distance, ±14mm)
      pose_3 (z)  = depth_mm             (contact depth)
      pose_4 (Rx) = ridge_orient_deg     (0°=horizontal, 90°=vertical)
      pose_5 (Ry) = 0 (unused)
      pose_6 (Rz) = yaw_deg             (sensor yaw)
    """
    num = len(labels['depth_mm'])
    target_df = pd.DataFrame(
        columns=[
            "sensor_image", "obj_id", "obj_pose", "pose_id",
            "pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6",
            "move_1", "move_2", "move_3", "move_4", "move_5", "move_6",
        ]
    )
    for i in range(num):
        image_file = f"image_{i+1:05d}.png"
        target_df.loc[i] = [
            image_file, 1, [0, 0, 0, 0, 0, 0], i + 1,
            0.0,
            float(labels['signed_d_nearest_mm'][i]),
            float(labels['depth_mm'][i]),
            float(labels['ridge_orient_deg'][i]),
            0.0,
            float(labels['yaw_deg'][i]),
            *np.zeros(6)
        ]
    return target_df


def collect_ridge_data(
    embodiment, poses_mm, image_dir,
    show_gui=True, show_tactile=True, quick_mode=True,
    record_video=False, video_path=None,
    labels=None,
):
    hover_dist = 0.0075
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    n = len(poses_mm)
    pb = embodiment._pb

    cam_width, cam_height = 640, 480
    side_names = {0: 'TOP', 1: 'RIGHT', 2: 'BOTTOM', 3: 'LEFT'}
    side_colors = {0: (0, 255, 100), 1: (255, 100, 0), 2: (0, 165, 255), 3: (200, 50, 200)}

    writer = None
    if record_video and video_path:
        cam_target = [0.6, 0.0, 0.035]
        cam_dist = 0.30
        cam_yaw, cam_pitch = 90, -35
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (cam_width, cam_height))
        print(f"  Recording ALL {n} frames to {video_path}")

    for index in range(n):
        pose = poses_mm[index]
        sensor_image = f"image_{index+1:05d}.png"

        final_pos = pose[:3] * 0.001
        final_rpy = pose[3:] * np.pi / 180

        if index % max(1, n // 20) == 0:
            print(f"  [{index+1}/{n}] wfx={pose[0]:.1f} wfy={pose[1]:.1f}mm "
                  f"d={pose[2]:.1f}mm r={pose[3]:.0f} p={pose[4]:.0f} w={pose[5]:.0f}")

        embodiment.move_linear(final_pos - [0, 0, hover_dist], final_rpy, quick_mode)
        embodiment.move_linear(final_pos, final_rpy, quick_mode)
        img = embodiment.process_sensor()

        if writer is not None:
            view_mat = pb.computeViewMatrixFromYawPitchRoll(
                cam_target, cam_dist, cam_yaw, cam_pitch, 0, 2)
            proj_mat = pb.computeProjectionMatrixFOV(60, cam_width / cam_height, 0.01, 1.0)
            _, _, rgb, _, _ = pb.getCameraImage(
                cam_width, cam_height, view_mat, proj_mat,
                lightDirection=[0, 0, 1],
                shadow=0,
                lightAmbientCoeff=0.6,
                lightDiffuseCoeff=0.4,
                lightSpecularCoeff=0.1)
            frame = np.ascontiguousarray(
                np.array(rgb, dtype=np.uint8).reshape(cam_height, cam_width, 4)[:, :, :3])
            tactile_resized = cv2.resize(img, (160, 160))
            if len(tactile_resized.shape) == 2:
                tactile_resized = cv2.cvtColor(tactile_resized, cv2.COLOR_GRAY2BGR)
            elif tactile_resized.shape[2] == 4:
                tactile_resized = tactile_resized[:, :, :3]
            frame[10:170, cam_width - 170:cam_width - 10] = tactile_resized
            cv2.putText(frame, f"[{index+1}/{n}]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if labels is not None:
                sid = int(labels['side'][index])
                color = side_colors[sid]
                name = side_names[sid]
                d = labels['d_nearest_mm'][index]
                sd = labels['signed_d_nearest_mm'][index]
                corner = labels['is_corner'][index]
                tag = f"{name} {'+ CORNER' if corner else ''}"
                cv2.putText(frame, tag, (10, cam_height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"d={d:.1f}mm  signed={sd:.1f}mm",
                            (10, cam_height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(frame)

        embodiment.move_linear(final_pos - [0, 0, hover_dist], final_rpy, quick_mode)
        cv2.imwrite(os.path.join(image_dir, sensor_image), img)

    embodiment.close()

    if writer is not None:
        writer.release()
        print(f"  Video done: {n} frames written")


def main():
    parser = argparse.ArgumentParser(description='Collect ridge tactile dataset')
    parser.add_argument('--n', type=int, default=20, help='Number of samples')
    parser.add_argument('--no-gui', action='store_true', help='Run headless')
    parser.add_argument('--quick', action='store_true', help='Skip settle time')
    parser.add_argument('--record', action='store_true', help='Record ALL frames to video')
    args = parser.parse_args()

    show_gui = not args.no_gui
    show_tactile = show_gui
    quick_mode = args.quick if args.quick else (not show_gui)

    poses_mm, sides = sample_perimeter_poses(args.n)
    labels = compute_labels(poses_mm, sides)
    target_df = make_target_df(labels)

    if os.path.exists(OUTPUT_DIR):
        import shutil
        for entry in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, entry)
            if os.path.isdir(path) and entry.startswith('collect_'):
                print(f"  Removing old dataset: {entry}")
                shutil.rmtree(path)

    collect_dir = os.path.join(OUTPUT_DIR, f"collect_{time.strftime('%m%d_%H%M')}")
    image_dir = os.path.join(collect_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    target_df.to_csv(os.path.join(collect_dir, 'targets.csv'), index=False)

    labels_df = pd.DataFrame(labels)
    labels_df.insert(0, 'sensor_image', [f"image_{i+1:05d}.png" for i in range(len(poses_mm))])
    labels_df.to_csv(os.path.join(collect_dir, 'labels.csv'), index=False)

    n_corner = int(np.sum(labels['is_corner']))
    n_straight = args.n - n_corner
    params = {
        'num_samples': args.n,
        'stimulus': 'wall_ridge_frame.urdf',
        'frame_size_mm': 2 * FRAME_HALF,
        'ridge_width_mm': RIDGE_W,
        'sampling': 'sequential_perimeter',
        'sides': ['bottom', 'right', 'top', 'left'],
        'n_straight': int(n_straight),
        'n_corner': int(n_corner),
        'corner_threshold_mm': CORNER_THRESH,
        'sensor': 'tactip_standard',
        'image_size': [128, 128],
    }
    with open(os.path.join(collect_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    # collection_params.json — training pipeline reads poses_rng for normalization
    # pose order: [x, y, z, Rx, Ry, Rz] mapped to [0, signed_d, depth, ridge_orient, 0, yaw]
    # Rx/Ry/Rz use sin/cos encoding (limits not used for normalization)
    collection_params = {
        'poses_rng': [
            [0.0, -14.0, -2.0, 0.0, 0.0, -20.0],   # lower limits
            [0.0,  14.0,  5.5, 90.0, 0.0, 20.0],    # upper limits
        ],
        'moves_rng': [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    }
    with open(os.path.join(collect_dir, 'collection_params.json'), 'w') as f:
        json.dump(collection_params, f, indent=2)

    per_side = [int(np.sum(labels['side'] == s)) for s in range(4)]
    print(f"Collecting {args.n} samples -> {collect_dir}")
    print(f"  Straight: {n_straight} ({100*n_straight/args.n:.0f}%)  "
          f"Corner: {n_corner} ({100*n_corner/args.n:.0f}%)")
    print(f"  Per side: B={per_side[0]} R={per_side[1]} T={per_side[2]} L={per_side[3]}")

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
    stim_path = os.path.join(STIMULI_DIR, 'wall_ridge', 'wall_ridge_frame.urdf')

    workframe_pos = [0.6, 0.0, 0.0175]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    embodiment, _ = setup_pybullet_env(
        stim_path, tactip_params,
        stimulus_pos, stimulus_rpy,
        workframe_pos, workframe_rpy,
        show_gui, show_tactile,
    )

    assets_dir = os.path.join(PROJECT_DIR, 'assets')
    video_path = os.path.join(assets_dir, 'ridge_data_collection.mp4') if args.record else None

    collect_ridge_data(
        embodiment, poses_mm, image_dir,
        show_gui=show_gui,
        show_tactile=show_tactile,
        quick_mode=quick_mode,
        record_video=args.record,
        video_path=video_path,
        labels=labels,
    )

    print(f"\nDone! {args.n} samples saved to {collect_dir}")
    print(f"  Images: {image_dir}/")
    print(f"  Labels: {os.path.join(collect_dir, 'labels.csv')}")
    print(f"  Straight ridge: {n_straight}  Corner: {n_corner}")
    for s, name in enumerate(['top', 'right', 'bottom', 'left']):
        count = int(np.sum(labels['side'] == s))
        print(f"    {name}: {count}")


if __name__ == '__main__':
    main()
