"""
Phase 3B validation: CNN-based tactile servo on PyBullet ridge frame.

Uses the trained ridge_4d CNN for perception + drone's servo logic from
tactile_servo_bridge.py for control:
  CNN (ridge_4d) → ServoController (v = -λ·e) → CornerDetector

Demonstrates the full pipeline (CNN + servo) working on the rectangular
ridge frame before deploying on the aerial platform with TeleKyb3.

Usage:
  conda activate tactile
  python 10_servo_on_ridge.py              # GUI + video
  python 10_servo_on_ridge.py --no-video   # GUI only
"""

import os
import sys
import argparse
import numpy as np
import torch
import imageio

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))

from torch.autograd import Variable
from tactile_gym.utils.general_utils import load_json_obj
from tactile_gym_servo_control.learning.learning_utils import import_task, POSE_LABEL_NAMES
from tactile_gym_servo_control.learning.networks import create_model
from tactile_gym_servo_control.utils.image_transforms import process_image
from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
STIMULI_DIR = os.path.join(PROJECT_DIR, 'servo_control',
                           'tactile_gym_servo_control', 'stimuli')

MODEL_DIR = os.path.join(PROJECT_DIR, 'servo_control',
                         'tactile_gym_servo_control', 'learned_models',
                         'ridge_4d', 'nature_cnn')

# Servo parameters (same as drone bridge)
SERVO_LAMBDA = 0.25
MAX_SERVO_VEL = 0.02       # m/s
FORWARD_SPEED = 0.020      # m/s along ridge
DT = 0.05                  # position increment timestep (s)
CONTACT_DEPTH = 0.004      # 4mm contact depth (wf_z)

# Corner detection (tuned for CNN noise — orient smoothing protects against false positives)
CORNER_ORIENT_THRESHOLD = 2
CORNER_COOLDOWN = 8
MIN_CORNER_STEPS = 20

# Rectangle traversal: same turn table as drone
TURN_AT_CORNER = {
    (+1, 0): (0, +1),
    (0, +1): (-1, 0),
    (-1, 0): (0, -1),
    (0, -1): (+1, 0),
}

# CNN signed_d convention: positive = inside frame.
# Cross-track correction sign depends on which ridge we're on.
SERVO_SIGN = {
    (+1, 0): -1,   # left ridge: push toward -y
    (0, +1): +1,   # top ridge: push toward +x
    (-1, 0): +1,   # right ridge: push toward +y
    (0, -1): -1,   # bottom ridge: push toward -x
}

# Ridge frame geometry (workframe coords, mm)
FRAME_HALF = 60.0



class CNNTactileSensor:
    """Wraps the trained ridge_4d CNN for real-time inference."""

    def __init__(self, model_dir, device='cpu'):
        self.device = device

        task = 'ridge_4d'
        self.out_dim, self.label_names = import_task(task)

        self.pose_limits_dict = load_json_obj(os.path.join(model_dir, 'pose_limits'))
        self.pose_limits = [self.pose_limits_dict['pose_llims'],
                            self.pose_limits_dict['pose_ulims']]
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        self.image_processing_params = load_json_obj(
            os.path.join(model_dir, 'image_processing_params'))

        self.model = create_model(
            self.image_processing_params['dims'],
            self.out_dim,
            model_params,
            saved_model_dir=model_dir,
            device=device
        )
        self.model.eval()
        self._orient_history = []
        self._smooth_window = 2
        print(f"CNN loaded from {model_dir}")

    def predict(self, tactile_image):
        """Run CNN inference on a tactile image.

        Returns dict with signed_d_mm, depth_mm, orient_deg, yaw_deg
        or None if prediction is out of range.
        """
        processed = process_image(
            tactile_image,
            gray=False,
            bbox=self.image_processing_params['bbox'],
            dims=self.image_processing_params['dims'],
            stdiz=self.image_processing_params['stdiz'],
            normlz=self.image_processing_params['normlz'],
            thresh=self.image_processing_params['thresh'],
        )

        processed = np.rollaxis(processed, 2, 0)
        processed = processed[np.newaxis, ...]

        with torch.no_grad():
            model_input = torch.from_numpy(processed).float().to(self.device)
            raw_output = self.model(model_input)

        predictions = decode_pose(raw_output, self.label_names, self.pose_limits)

        # y → signed_d_mm, z → depth_mm, Rx → orient_deg, Rz → yaw_deg
        signed_d_mm = predictions['y'].item()
        depth_mm = predictions['z'].item()
        orient_deg = predictions['Rx'].item()
        yaw_deg = predictions['Rz'].item()

        # Snap orient to 0 or 90 (CNN outputs continuous value)
        orient_snap = 90.0 if abs(orient_deg - 90) < abs(orient_deg - 0) else 0.0

        return {
            'signed_d_mm': signed_d_mm,
            'depth_mm': depth_mm,
            'orient': orient_snap,
            'orient_raw': orient_deg,
            'yaw_deg': yaw_deg,
        }


class ServoController:
    """v = -λ·e — same as drone bridge."""

    def compute(self, pred, forward):
        e_m = pred['signed_d_mm'] * 0.001
        sign = SERVO_SIGN.get(forward, -1)
        v = np.clip(sign * SERVO_LAMBDA * e_m, -MAX_SERVO_VEL, MAX_SERVO_VEL)
        # CNN convention: orient=90 → left/right ridge (wf_y=±60) → cross-track is y
        #                 orient=0  → bottom/top ridge (wf_x=±60) → cross-track is x
        vx = v if pred['orient'] == 0 else 0.0
        vy = v if pred['orient'] == 90 else 0.0
        return vx, vy


class CornerDetector:
    """Same logic as drone bridge."""

    def __init__(self, init_forward):
        self.forward = init_forward
        self.expected_orient = self._orient_for_forward()
        self.change_count = 0
        self.corner_count = 0
        self.cooldown = 0
        self.steps_since_corner = 0

    def _orient_for_forward(self):
        dx, dy = self.forward
        # CNN convention: moving along x = on left/right ridge = orient 90
        return 90 if dx != 0 else 0

    def is_expected(self, orient):
        return orient == self.expected_orient

    def update(self, orient):
        self.steps_since_corner += 1

        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        if self.steps_since_corner < MIN_CORNER_STEPS:
            return False

        if orient != self.expected_orient:
            self.change_count += 1
        else:
            self.change_count = 0

        if self.change_count >= CORNER_ORIENT_THRESHOLD:
            old_fwd = self.forward
            self.forward = TURN_AT_CORNER.get(self.forward, self.forward)
            self.expected_orient = self._orient_for_forward()
            self.change_count = 0
            self.corner_count += 1
            self.cooldown = CORNER_COOLDOWN
            self.steps_since_corner = 0
            print(f"  CORNER #{self.corner_count}: {old_fwd} -> {self.forward}")
            return True
        return False

    def get_forward_velocity(self):
        return (self.forward[0] * FORWARD_SPEED,
                self.forward[1] * FORWARD_SPEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--model-dir', default=MODEL_DIR,
                        help='Path to trained ridge_4d model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup PyBullet with ridge frame
    stim_path = os.path.join(STIMULI_DIR, 'wall_ridge', 'wall_ridge_frame.urdf')
    tactip_params = {
        "name": "tactip", "type": "standard", "core": "no_core",
        "dynamics": {}, "image_size": [128, 128], "turn_off_border": False,
    }
    stimulus_pos = [0.6, 0.0, 0.0125]
    stimulus_rpy = [0, 0, 0]
    workframe_pos = [0.6, 0.0, 0.0175]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    embodiment, _ = setup_pybullet_env(
        stim_path, tactip_params,
        stimulus_pos, stimulus_rpy,
        workframe_pos, workframe_rpy,
        show_gui=True, show_tactile=True,
    )

    # Start ON the left ridge: wf_x=-40mm (partway along), wf_y=-58mm (2mm offset)
    start_x, start_y = -0.040, -0.058
    embodiment.move_linear([start_x, start_y, CONTACT_DEPTH], [0, 0, 0], quick_mode=False)

    # Load CNN
    cnn_sensor = CNNTactileSensor(args.model_dir, device=device)
    servo = ServoController()
    corner_det = CornerDetector(init_forward=(+1, 0))

    print(f"\nPhase 3B validation: CNN servo on ridge frame")
    print(f"  lambda={SERVO_LAMBDA}, fwd={FORWARD_SPEED*1000:.0f}mm/s")
    print(f"  Start: wf=({start_x*1000:.0f}, {start_y*1000:.0f})mm")
    print(f"  Press 'q' to quit\n")

    render_frames = []
    log = []
    pos = np.array([start_x, start_y, CONTACT_DEPTH])

    for step in range(args.steps):
        wf_x_mm = pos[0] * 1000
        wf_y_mm = pos[1] * 1000

        # Get tactile image from TacTip sensor
        tactile_image = embodiment.get_tactile_observation()

        # CNN prediction
        pred = cnn_sensor.predict(tactile_image)

        vx_servo, vy_servo = 0.0, 0.0

        if pred is not None:
            corner_det.update(pred['orient'])
            if corner_det.is_expected(pred['orient']):
                vx_servo, vy_servo = servo.compute(pred, corner_det.forward)

        fwd_vx, fwd_vy = corner_det.get_forward_velocity()
        vx_total = vx_servo + fwd_vx
        vy_total = vy_servo + fwd_vy

        pos[0] += vx_total * DT
        pos[1] += vy_total * DT

        embodiment.move_linear(pos.copy(), [0, 0, 0], quick_mode=True)

        if step % 30 == 0:
            print(f"  [{step:3d}/{args.steps}] "
                  f"wf=({wf_x_mm:+6.1f}, {wf_y_mm:+6.1f})mm  "
                  f"d={pred['signed_d_mm']:+5.1f}mm  "
                  f"orient={pred['orient_raw']:+5.1f}→{pred['orient']:.0f}°  "
                  f"fwd={corner_det.forward}  "
                  f"corners={corner_det.corner_count}")

        log.append([step, wf_x_mm, wf_y_mm,
                    pred['signed_d_mm'],
                    pred['orient'],
                    corner_det.corner_count])

        if not args.no_video:
            render_img = embodiment.render()
            render_frames.append(render_img)

        if corner_det.corner_count >= 4:
            print(f"\n  Full rectangle complete at step {step}!")
            break

        q_key = ord("q")
        keys = embodiment._pb.getKeyboardEvents()
        if q_key in keys and keys[q_key] & embodiment._pb.KEY_WAS_TRIGGERED:
            break

    embodiment.close()

    if render_frames:
        video_path = os.path.join(PROJECT_DIR, 'assets', 'ridge_servo_demo.mp4')
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimwrite(video_path, np.stack(render_frames), fps=24)
        print(f"\nVideo: {video_path}")

    log_path = os.path.join(PROJECT_DIR, 'output', 'ridge_servo_log.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    np.savetxt(log_path, log, delimiter=',',
               header='step,wf_x_mm,wf_y_mm,signed_d_mm,orient,corners', comments='')
    print(f"Log: {log_path}")
    print(f"Corners detected: {corner_det.corner_count}")


if __name__ == '__main__':
    main()
