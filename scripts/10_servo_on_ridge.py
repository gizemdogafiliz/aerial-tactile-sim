"""
Phase 3B validation: Velocity-based tactile servo on PyBullet ridge frame.

Same algorithm as the drone's tactile_servo_bridge.py:
  GeometricTactileSensor → ServoController (v = -λ·e) → CornerDetector

Demonstrates the servo logic works on the rectangular ridge frame
before deploying it on the aerial platform with TeleKyb3.

Usage:
  conda activate tactile
  python 10_servo_on_ridge.py              # GUI + video
  python 10_servo_on_ridge.py --no-video   # GUI only
"""

import os
import sys
import argparse
import numpy as np
import imageio

sys.path.insert(0, os.path.expanduser('~/aerial-tactile-sim/servo_control'))
from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
STIMULI_DIR = os.path.join(PROJECT_DIR, 'servo_control',
                           'tactile_gym_servo_control', 'stimuli')

# Ridge frame geometry (workframe coords, mm)
FRAME_HALF = 60.0   # 120x120mm frame → ridges at ±60mm
RIDGE_W = 5.0       # ridge cross-section width
DOME_R = 16.0        # TacTip dome radius
RIDGES_WF = [
    {'name': 'left',   'orient': 0,  'axis': 'y', 'coord': -FRAME_HALF,
     'span_axis': 'x', 'span': (-FRAME_HALF, FRAME_HALF)},
    {'name': 'right',  'orient': 0,  'axis': 'y', 'coord':  FRAME_HALF,
     'span_axis': 'x', 'span': (-FRAME_HALF, FRAME_HALF)},
    {'name': 'bottom', 'orient': 90, 'axis': 'x', 'coord': -FRAME_HALF,
     'span_axis': 'y', 'span': (-FRAME_HALF, FRAME_HALF)},
    {'name': 'top',    'orient': 90, 'axis': 'x', 'coord':  FRAME_HALF,
     'span_axis': 'y', 'span': (-FRAME_HALF, FRAME_HALF)},
]

# Servo parameters (same as drone bridge)
SERVO_LAMBDA = 0.15
MAX_SERVO_VEL = 0.02       # m/s
FORWARD_SPEED = 0.035      # m/s along ridge
DT = 0.05                  # position increment timestep (s)
CONTACT_DEPTH = 0.004      # 4mm contact depth (wf_z)

# Corner detection (same as drone bridge)
CORNER_ORIENT_THRESHOLD = 2
CORNER_COOLDOWN = 5
MIN_CORNER_STEPS = 15

# Rectangle traversal: same turn table as drone
# wf coords: x-forward along ridge, y is cross-track for orient=0
TURN_AT_CORNER = {
    (+1, 0): (0, +1),   # right along x → up along y
    (0, +1): (-1, 0),   # up along y → left along x
    (-1, 0): (0, -1),   # left along x → down along y
    (0, -1): (+1, 0),   # down along y → right along x
}


class GeometricTactileSensor:
    """Same interface as drone's sensor, adapted for PyBullet workframe coords."""

    def __init__(self, ridges, noise_std=0.0):
        self.ridges = ridges
        self.noise_std = noise_std

    def sense(self, wf_x_mm, wf_y_mm, wf_z_mm):
        best = None
        best_dist = float('inf')

        for ridge in self.ridges:
            if ridge['axis'] == 'y':
                perp_dist = wf_y_mm - ridge['coord']
                along_val = wf_x_mm
            else:
                perp_dist = wf_x_mm - ridge['coord']
                along_val = wf_y_mm

            span_lo, span_hi = ridge['span']
            if along_val < span_lo - 5 or along_val > span_hi + 5:
                continue

            if abs(perp_dist) < best_dist:
                best_dist = abs(perp_dist)
                best = {
                    'signed_d_mm': perp_dist,
                    'depth_mm': wf_z_mm,
                    'orient': ridge['orient'],
                    'ridge_name': ridge['name'],
                    'along_val': along_val,
                }

        if best is None or best_dist > DOME_R:
            return None

        if self.noise_std > 0:
            best['signed_d_mm'] += np.random.normal(0, self.noise_std)

        return best


class ServoController:
    """v = -λ·e — same as drone bridge."""

    def compute(self, pred):
        e_m = pred['signed_d_mm'] * 0.001
        v = np.clip(-SERVO_LAMBDA * e_m, -MAX_SERVO_VEL, MAX_SERVO_VEL)
        # Cross-track correction perpendicular to ridge
        vx = v if pred['orient'] == 90 else 0.0
        vy = v if pred['orient'] == 0 else 0.0
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
        return 0 if dx != 0 else 90

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
    parser.add_argument('--noise', type=float, default=0.0, help='Sensor noise std (mm)')
    args = parser.parse_args()

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

    sensor = GeometricTactileSensor(RIDGES_WF, noise_std=args.noise)
    servo = ServoController()
    corner_det = CornerDetector(init_forward=(+1, 0))  # start going right along x

    print(f"\nPhase 3B validation: servo on ridge frame")
    print(f"  lambda={SERVO_LAMBDA}, fwd={FORWARD_SPEED*1000:.0f}mm/s")
    print(f"  Start: wf=({start_x*1000:.0f}, {start_y*1000:.0f})mm")
    print(f"  Press 'q' to quit\n")

    render_frames = []
    log = []
    pos = np.array([start_x, start_y, CONTACT_DEPTH])

    for step in range(args.steps):
        wf_x_mm = pos[0] * 1000
        wf_y_mm = pos[1] * 1000
        wf_z_mm = pos[2] * 1000

        pred = sensor.sense(wf_x_mm, wf_y_mm, wf_z_mm)

        vx_servo, vy_servo = 0.0, 0.0

        if pred is not None:
            corner_det.update(pred['orient'])
            if corner_det.is_expected(pred['orient']):
                vx_servo, vy_servo = servo.compute(pred)

        fwd_vx, fwd_vy = corner_det.get_forward_velocity()
        vx_total = vx_servo + fwd_vx
        vy_total = vy_servo + fwd_vy

        pos[0] += vx_total * DT
        pos[1] += vy_total * DT

        embodiment.move_linear(pos.copy(), [0, 0, 0], quick_mode=True)

        if step % 30 == 0:
            if pred:
                print(f"  [{step:3d}/{args.steps}] "
                      f"wf=({wf_x_mm:+6.1f}, {wf_y_mm:+6.1f})mm  "
                      f"ridge={pred['ridge_name']:6s}  "
                      f"d={pred['signed_d_mm']:+5.1f}mm  "
                      f"fwd={corner_det.forward}  "
                      f"corners={corner_det.corner_count}")
            else:
                print(f"  [{step:3d}/{args.steps}] "
                      f"wf=({wf_x_mm:+6.1f}, {wf_y_mm:+6.1f})mm  NO RIDGE")

        log.append([step, wf_x_mm, wf_y_mm,
                    pred['signed_d_mm'] if pred else np.nan,
                    pred['orient'] if pred else np.nan,
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
