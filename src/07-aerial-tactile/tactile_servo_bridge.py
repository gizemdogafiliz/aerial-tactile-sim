"""
Phase 3B: Tactile Servo Bridge — Ridge Following on Rectangular Frame

Drone flies to wall, makes contact near a ridge, then follows the entire
rectangular frame using tactile servo control with corner detection.

Architecture:
  Inner loop (untouched): maneuver → phynt(AF+WO) → uavpos(1000Hz) → uavatt → motors
  Outer loop (this script, ~0.59 Hz):
    pom.frame('robot') → body_to_ee_tip(FK) → GeometricTactileSensor
      → ServoController(v = -λ·e) → maneuver.velocity()

Control:
  Cross-track (y-z): velocity-based servo, v = -λ·e (position-independent)
  Depth (x): wrench reference or conservative position bias (no accumulation)
  When CNN replaces geometric sensor, servo code stays unchanged (same interface)

Usage:
  Terminal 1: sh simulation.sh
  Terminal 2: python3 -i tactile_servo_bridge.py
              >>> simulation()
"""

import genomix
import math
import numpy as np
import os
import time
import shutil

# ############################################
#  PARAMETERS
# ############################################

MASS = 2.72
J = [0.0115, 0, 0, 0, 0.0114, 0, 0, 0, 0.0194]
ARMLEN = 0.38998
CF = 9.9016e-4
CT = 1.9e-5

X_WALL = 2.0
L_BAR = 0.6
TIP_RADIUS = 0.02
L_EFF = L_BAR  # SDF sphere center distance from body origin
EE_Z_OFFSET = -0.125
DOME_PROXIMITY = 0.005  # 5mm — soft dome detects contact before rigid sphere touches

LOG_DIR = '/shared-workspace/logs/07-aerial-tactile-servo'
os.makedirs(LOG_DIR, exist_ok=True)

# Ridge positions from hexa-fa-wall-tactile.world
# Ridges protrude 5mm from wall face (x=1.9975)
RIDGE_X = 1.9975
RIDGES = [
    {'name': 'bottom', 'orient': 0,  'coord': 0.875, 'axis': 'z',
     'span_axis': 'y', 'span': (0.0, 1.0)},
    {'name': 'top',    'orient': 0,  'coord': 1.875, 'axis': 'z',
     'span_axis': 'y', 'span': (0.0, 1.0)},
    {'name': 'right',  'orient': 90, 'coord': 0.0,   'axis': 'y',
     'span_axis': 'z', 'span': (0.875, 1.875)},
    {'name': 'left',   'orient': 90, 'coord': 1.0,   'axis': 'y',
     'span_axis': 'z', 'span': (0.875, 1.875)},
]

# Velocity-based tactile servo: v = -λ·e
# Direct velocity commands (maneuver.velocity) — no trajectory planner snap/jerk.
# Smooth continuous motion: no impulsive pitch transients from goto.
# With goto: 3.6mm correction → 7.2N force < 8.5N friction → EE stuck.
# With velocity: continuous motion, AF force accumulates → EE slides.
GOTO_DURATION = 1.5         # for Phase 0/1/3 positioning (not servo)
VEL_DURATION = 2.0          # velocity command duration (overlaps next step)
STEP_SLEEP = 1.0            # sensor loop rate (~1 Hz, was 1.7 with goto)
FORWARD_SPEED = 0.035       # m/s along ridge
SERVO_LAMBDA = 0.15         # convergence rate (1/s)
MAX_SERVO_VEL = 0.02        # max cross-track velocity (m/s, safety clamp)
DEPTH_VEL_K = 0.5           # depth velocity gain: vx = -K * depth_error
MAX_DEPTH_VEL = 0.005       # max depth velocity (m/s)
LOW_DEPTH_THRESHOLD = 0.001 # 1mm — stop servo corrections (forward continues)
RAMP_STEPS = 5              # ramp forward velocity over N steps (avoid roll transients)

# Corner detection
CORNER_ORIENT_THRESHOLD = 1   # consecutive samples with new orient = corner
CORNER_COOLDOWN = 3           # resume tracking sooner after corner turn
MIN_CORNER_STEPS = 10         # minimum steps between corners (~0.3m travel)

# Depth servo via wrench reference (approach 1 from literature).
# CNN/geometric sensor gives depth → compute force correction → inject into AF.
# This avoids pitch coupling because AF works in force space, not position space.
# Fallback: conservative position bias (no accumulation, resets each step).
DEPTH_REF = 0.002             # target dome compression (2mm)
DEPTH_K_WRENCH = 50.0         # N/m — wrench correction gain: F = K * depth_error
DEPTH_K_POS = 0.3             # position fallback gain (conservative)
MAX_DEPTH_POS_BIAS = 0.003    # max position correction (3mm, no accumulation)
DEPTH_SERVO_MODE = 'position' # 'wrench' or 'position' — set to wrench after API test





# Rectangle traversal: left → up → right → down (counterclockwise, looking at wall)
TURN_AT_CORNER = {
    (+1, 0): (0, +1),   # left → up
    (0, +1): (-1, 0),   # up → right
    (-1, 0): (0, -1),   # right → down
    (0, -1): (+1, 0),   # down → left
}

# ############################################
#  GEOMETRIC TACTILE SENSOR
# ############################################

class GeometricTactileSensor:
    """Perfect tactile sensor — computes ground truth from ridge geometry."""

    def __init__(self, ridges, noise_std=0.0):
        self.ridges = ridges
        self.noise_std = noise_std

    def in_contact(self, ee_x):
        """True if soft dome is touching wall (within DOME_PROXIMITY of surface)."""
        wall_gap = X_WALL - (ee_x + TIP_RADIUS)
        return wall_gap < DOME_PROXIMITY

    def sense(self, ee_y, ee_z, ee_x):
        """Soft dome tactile sensor model.

        Detects contact when ball surface is within DOME_PROXIMITY of wall,
        simulating TacTip silicone dome deformation. CNN-compatible interface.
        """
        wall_gap = X_WALL - (ee_x + TIP_RADIUS)
        if wall_gap > DOME_PROXIMITY:
            return None
        depth = DOME_PROXIMITY - wall_gap

        best = None
        best_dist = float('inf')

        for ridge in self.ridges:
            if ridge['axis'] == 'z':
                perp_dist = ee_z - ridge['coord']
                along_val = ee_y
            else:
                perp_dist = ee_y - ridge['coord']
                along_val = ee_z

            span_lo, span_hi = ridge['span']
            if along_val < span_lo - 0.02 or along_val > span_hi + 0.02:
                continue

            if abs(perp_dist) < best_dist:
                best_dist = abs(perp_dist)
                best = {
                    'signed_d_m': perp_dist,
                    'depth_m': depth,
                    'orient': ridge['orient'],
                    'yaw_deg': 0.0,
                    'ridge_name': ridge['name'],
                    'along_val': along_val,
                }

        if best is None or best_dist > TIP_RADIUS:
            return None

        if self.noise_std > 0:
            best['signed_d_m'] += np.random.normal(0, self.noise_std)
            best['depth_m'] += np.random.normal(0, self.noise_std * 0.3)

        return best


# ############################################
#  SERVO CONTROLLER
# ############################################

class ServoController:
    """Velocity-based tactile servo: v = -λ·e.
    Returns velocity (m/s) — fed directly to maneuver.velocity()."""

    def compute(self, pred):
        e = pred['signed_d_m']
        v = np.clip(-SERVO_LAMBDA * e, -MAX_SERVO_VEL, MAX_SERVO_VEL)
        vy = v if pred['orient'] == 90 else 0.0
        vz = v if pred['orient'] == 0 else 0.0
        return vy, vz


# ############################################
#  DEPTH SERVO
# ############################################

class DepthServo:
    """Depth control via wrench reference or conservative position bias.

    Wrench mode (preferred): computes F_x correction from depth error,
    injects into phynt's AF as wrench reference. AF absorbs it smoothly.

    Position mode (fallback): small position bias on target_x, NO accumulation.
    Resets each step → no drift. Much safer than old integrating depth servo.
    """

    def __init__(self, mode='position'):
        self.mode = mode
        self.wrench_available = False

    def try_wrench_injection(self, phynt_handle):
        """Test if phynt supports wrench reference injection.
        Call once during setup, inside Docker container."""
        if self.mode != 'wrench':
            return
        try:
            services = dir(phynt_handle)
            wrench_services = [s for s in services
                               if 'wrench' in s.lower() or 'force' in s.lower()]
            print(f"[DepthServo] phynt wrench-related services: {wrench_services}")
            self.wrench_available = len(wrench_services) > 0
            if not self.wrench_available:
                print("[DepthServo] No wrench injection API found — falling back to position mode")
                self.mode = 'position'
        except Exception as e:
            print(f"[DepthServo] API check failed: {e} — using position mode")
            self.mode = 'position'

    def compute(self, depth_m):
        """Compute depth correction from sensor depth.

        Returns:
            dx_position: body_x bias (meters), for position mode
            fx_wrench: force correction (N), for wrench mode
        """
        if depth_m is None:
            return 0.0, 0.0

        depth_error = depth_m - DEPTH_REF

        fx_wrench = -DEPTH_K_WRENCH * depth_error

        dx_pos = -DEPTH_K_POS * depth_error * GOTO_DURATION
        dx_pos = np.clip(dx_pos, -MAX_DEPTH_POS_BIAS, MAX_DEPTH_POS_BIAS)

        return dx_pos, fx_wrench

    def apply(self, pred, phynt_handle=None):
        """Apply depth correction. Returns position bias for target_x."""
        if pred is None:
            return 0.0

        dx_pos, fx_wrench = self.compute(pred['depth_m'])

        if self.mode == 'wrench' and self.wrench_available and phynt_handle:
            try:
                phynt_handle.set_af_wrench(
                    wrench={'fx': fx_wrench, 'fy': 0, 'fz': 0,
                            'tx': 0, 'ty': 0, 'tz': 0})
                return 0.0
            except Exception:
                pass

        return dx_pos

    def compute_vx(self, depth_m):
        """Compute depth velocity for velocity-based servo."""
        if depth_m is None:
            return 0.0
        depth_error = depth_m - DEPTH_REF
        return np.clip(-DEPTH_VEL_K * depth_error, -MAX_DEPTH_VEL, MAX_DEPTH_VEL)


# ############################################
#  CORNER DETECTOR
# ############################################

class CornerDetector:
    """Detects corners by orient change with expected-ridge filtering."""

    def __init__(self, init_forward):
        self.forward = init_forward
        self.expected_orient = self._orient_for_forward()
        self.change_count = 0
        self.corner_count = 0
        self.cooldown = 0
        self.steps_since_corner = 0  # must follow ridge MIN_CORNER_STEPS before first corner

    def _orient_for_forward(self):
        dy, dz = self.forward
        return 0 if dy != 0 else 90

    def is_expected(self, orient):
        """True if this reading is from the ridge we're supposed to follow."""
        return orient == self.expected_orient

    def update(self, orient):
        """Returns True if corner detected."""
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
            old_forward = self.forward
            self.forward = TURN_AT_CORNER.get(self.forward, self.forward)
            self.expected_orient = self._orient_for_forward()
            self.change_count = 0
            self.corner_count += 1
            self.cooldown = CORNER_COOLDOWN
            self.steps_since_corner = 0
            print(f"  CORNER #{self.corner_count}: {old_forward}→{self.forward} "
                  f"(expect orient={self.expected_orient})")
            return True

        return False

    def get_forward_velocity(self):
        return (self.forward[0] * FORWARD_SPEED,
                self.forward[1] * FORWARD_SPEED)


# ############################################
#  EE FORWARD KINEMATICS
# ############################################

def body_to_ee_tip(pos, att_quat):
    """Compute EE tip position from body pose.

    pos: [x, y, z] body center
    att_quat: [qw, qx, qy, qz] body orientation
    Returns: [x, y, z] of EE tip in world frame
    """
    qw, qx, qy, qz = att_quat

    r11 = 1 - 2*(qy*qy + qz*qz)
    r12 = 2*(qx*qy - qz*qw)
    r13 = 2*(qx*qz + qy*qw)
    r21 = 2*(qx*qy + qz*qw)
    r22 = 1 - 2*(qx*qx + qz*qz)
    r23 = 2*(qy*qz - qx*qw)
    r31 = 2*(qx*qz - qy*qw)
    r32 = 2*(qy*qz + qx*qw)
    r33 = 1 - 2*(qx*qx + qy*qy)

    ee_x = pos[0] + L_EFF * r11 + EE_Z_OFFSET * r13
    ee_y = pos[1] + L_EFF * r21 + EE_Z_OFFSET * r23
    ee_z = pos[2] + L_EFF * r31 + EE_Z_OFFSET * r33

    return np.array([ee_x, ee_y, ee_z])


# ############################################
#  GENOMIX SETUP (same as model_hexa_fa_tactile.py)
# ############################################

g = genomix.connect()
g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom = g.load('pom')
uavpos = g.load('uavpos')
uavatt = g.load('uavatt')
maneuver = g.load('maneuver')
phynt = g.load('phynt')


def setup():
    print("Waiting 5s for Gazebo...")
    time.sleep(5)

    optitrack.connect({
        'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
    })

    rotorcraft.connect({'serial': '/tmp/pty-hr6', 'baud': 0})
    rotorcraft.set_sensor_rate({'rate': {
        'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
    }})
    rotorcraft.set_imu_filter({
        'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
    })

    pom.set_prediction_model('::pom::constant_acceleration')
    pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
    pom.set_history_length({'history_length': 0.25})
    pom.set_mag_field({'magdir': {
        'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
    }})
    pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
    pom.add_measurement('imu')
    pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
    pom.add_measurement('mag')
    pom.connect_port({
        'local': 'measure/mocap', 'remote': 'optitrack/bodies/HR_6'
    })
    pom.add_measurement('mocap')

    uavpos.set_mass({'mass': MASS})
    uavpos.set_xyradius({'rxy': 5.0})
    uavpos.set_saturation({'sat': {'x': 3.0, 'v': 3.0, 'ix': 0.5}})
    uavpos.set_servo_gain({'gain': {
        'Kpxy': 5.0,  'Kpz': 25.0,
        'Kvxy': 7.0,  'Kvz': 18.0,
        'Kixy': 0.0,  'Kiz': 1.0
    }})
    uavpos.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 50.0, 'dv': 50.0
    }})
    uavpos.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})
    uavpos.connect_port({'local': 'reference', 'remote': 'phynt/desired'})

    uavatt.set_gtmrp_geom({
        'rotors': 6, 'cx': 0, 'cy': 0, 'cz': 0,
        'armlen': ARMLEN, 'mass': MASS,
        'rx': -21.2, 'ry': -18.7, 'rz': -1, 'cf': CF, 'ct': CT
    })
    uavatt.set_wlimit({'wmin': 16.0, 'wmax': 100.0})
    uavatt.set_servo_gain({'gain': {
        'Kqxy': 4.0, 'Kqz': 3.0,
        'Kwxy': 0.8, 'Kwz': 0.8
    }})
    uavatt.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})
    uavatt.connect_port({'local': 'uav_input', 'remote': 'uavpos/uav_input'})

    rotorcraft.connect_port({
        'local': 'rotor_input', 'remote': 'uavatt/rotor_input'
    })

    maneuver.set_bounds({
        'xmin': -100, 'xmax': 100,
        'ymin': -100, 'ymax': 100,
        'zmin': -100, 'zmax': 100,
        'yawmin': -2*math.pi, 'yawmax': 2*math.pi
    })
    maneuver.set_velocity_limit({'v': 1, 'w': 0.5})
    maneuver.set_acceleration_limit({'a': 0.8, 'dw': 0.5})
    maneuver.set_jerk_limit({'j': 5, 'ddw': 3})
    maneuver.set_snap_limit({'s': 25, 'dddw': 15})
    maneuver.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})

    phynt.set_mass(mass=MASS)
    phynt.set_geom(J=J)

    af_mass = 5.0
    af_K = [100.0, 2000.0, 2000.0, 50.0, 50.0, 50.0]
    af_B = [2*math.sqrt(af_mass*af_K[i]) for i in range(6)]
    af_J = [0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05]
    phynt.set_af_parameters(mass=af_mass, B=af_B, K=af_K, J=af_J)

    phynt.set_wo_gains(K=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    phynt.set_wo_fc(fc=[20.0, 20.0, 20.0, 1.0, 1.0, 1.0])

    phynt.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})
    phynt.connect_port({
        'local': 'wrench_measure', 'remote': 'uavatt/wrench_measure'
    })
    phynt.connect_port({
        'local': 'reference', 'remote': 'maneuver/desired'
    })

    print("All components connected.")


def start():
    pom.log_state('/tmp/pom.log')
    optitrack.set_logfile('/tmp/opti.log')
    rotorcraft.log('/tmp/rotorcraft.log')
    uavpos.log('/tmp/uavpos.log')
    uavatt.log('/tmp/uavatt.log')
    maneuver.log('/tmp/maneuver.log')

    rotorcraft.start()
    rotorcraft.servo(ack=True)

    uavpos.set_current_position()
    maneuver.set_current_state()

    phynt.enable(enable={'wo': True, 'af': True})
    phynt.set_current_position()
    phynt.servo(ack=True)

    uavatt.servo(ack=True)
    uavpos.servo(ack=True)

    time.sleep(0.5)
    phynt.log('/tmp/phynt.log')
    print("Started.")


def stop():
    rotorcraft.stop()
    uavpos.stop()
    uavatt.stop()
    phynt.stop()

    rotorcraft.log_stop()
    uavpos.log_stop()
    uavatt.log_stop()
    maneuver.log_stop()
    phynt.log_stop()
    pom.log_stop()
    optitrack.unset_logfile()

    for f in ['rotorcraft.log', 'pom.log', 'uavpos.log',
              'uavatt.log', 'maneuver.log', 'phynt.log', 'opti.log']:
        src = '/tmp/' + f
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(LOG_DIR, f))
    print(f"Logs saved to: {LOG_DIR}")


def read_state():
    try:
        s = pom.frame('robot')['frame']
        pos = np.array([s['pos']['x'], s['pos']['y'], s['pos']['z']])
        att = np.array([s['att']['qw'], s['att']['qx'],
                        s['att']['qy'], s['att']['qz']])
        return pos, att
    except Exception as e:
        print(f"[read_state ERROR] {e}")
        return np.zeros(3), np.array([1, 0, 0, 0])


# ############################################
#  APPROACH WAYPOINTS
# ############################################

# Approximate body target for bottom ridge (z=0.875).
# Exact EE position depends on pitch under contact — Phase 1 scan corrects.
HOVER_POS = (0, 0, 1.0)
APPROACH_POS = (1.2, 0.05, 1.0)
CONTACT_BODY_X = 1.45
CONTACT_START = (CONTACT_BODY_X, 0.05, 1.0)
RETRACT_POS = (1.0, 0, 1.0)
LAND_POS = (0, 0, 0)


# ############################################
#  MAIN SIMULATION
# ############################################

def simulation(sensor_mode='geometric', noise_std=0.0005, max_servo_time=700.0, tag=None):
    """Run tactile servo ridge following.

    Args:
        sensor_mode: 'geometric' (perfect) or 'noisy' (geometric + noise)
        tag: if given, also saves a copy to run_<tag>/ (for comparison later)
        noise_std: noise std for 'noisy' mode (meters), default 0.5mm
        max_servo_time: max seconds in servo loop before retract
    """
    print("=" * 60)
    print("  Phase 3B: Tactile Servo Ridge Following")
    print(f"  Sensor: {sensor_mode}, noise: {noise_std*1000:.1f}mm")
    print("=" * 60)

    setup()
    start()

    sensor = GeometricTactileSensor(
        RIDGES,
        noise_std=noise_std if sensor_mode == 'noisy' else 0.0
    )
    servo = ServoController()
    depth_servo = DepthServo(mode=DEPTH_SERVO_MODE)
    depth_servo.try_wrench_injection(phynt)
    # Start going right along bottom ridge
    corner_det = CornerDetector(init_forward=(+1, 0))

    # --- Logging arrays ---
    max_steps = int(max_servo_time / STEP_SLEEP) + 100
    log = {
        't': np.zeros(max_steps),
        'body_pos': np.zeros((max_steps, 3)),
        'ee_pos': np.zeros((max_steps, 3)),
        'signed_d': np.zeros(max_steps),
        'depth': np.zeros(max_steps),
        'orient': np.zeros(max_steps),
        'correction': np.zeros((max_steps, 3)),
        'target_pos': np.zeros((max_steps, 3)),
        'phase': np.zeros(max_steps, dtype=int),
    }
    step = 0

    def log_step(t, body_pos, ee_pos, pred, corr, target, phase):
        nonlocal step
        if step >= max_steps:
            return
        log['t'][step] = t
        log['body_pos'][step] = body_pos
        log['ee_pos'][step] = ee_pos
        if pred:
            log['signed_d'][step] = pred['signed_d_m'] * 1000
            log['depth'][step] = pred['depth_m'] * 1000
            log['orient'][step] = pred['orient']
        log['correction'][step] = corr
        log['target_pos'][step] = target
        log['phase'][step] = phase
        step += 1

    try:
        # ===== PHASE 0: Hover =====
        print(f"\n[Phase 0] Hover at {HOVER_POS}")
        maneuver.goto(*HOVER_POS, 0, 5)
        time.sleep(8)

        # Calibrate wrench observer in free flight
        print("[Phase 0] Calibrating WO...")
        phynt.set_wo_zero(duration=2.0)
        time.sleep(3)

        # ===== PHASE 1: Approach =====
        print(f"\n[Phase 1] Approach wall → {APPROACH_POS}")
        maneuver.goto(*APPROACH_POS, 0, 4)
        time.sleep(12)

        print(f"[Phase 1] Move to contact start → body_x={CONTACT_BODY_X}")
        maneuver.goto(*CONTACT_START, 0, 3)
        time.sleep(10)

        # Read current state, verify we're near wall
        pos, att = read_state()
        ee = body_to_ee_tip(pos, att)
        print(f"[Phase 1] Body: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"[Phase 1] EE tip: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")

        scan_x = CONTACT_BODY_X
        expected_orient = corner_det.expected_orient
        pred = sensor.sense(ee[1], ee[2], ee[0])
        if pred and pred['orient'] != expected_orient:
            print(f"[Phase 1] Found {pred['ridge_name']} (orient={pred['orient']}°) "
                  f"but need orient={expected_orient}° — scanning for correct ridge")
            pred = None
        if pred:
            print(f"[Phase 1] Ridge found: {pred['ridge_name']}, "
                  f"signed_d={pred['signed_d_m']*1000:.1f}mm, "
                  f"depth={pred['depth_m']*1000:.1f}mm")
        else:
            print(f"[Phase 1] Scanning for orient={expected_orient}° ridge...")
            maneuver.set_current_state()
            time.sleep(0.3)
            scan_z = pos[2]
            scan_y = pos[1]
            scan_x = CONTACT_BODY_X
            for i in range(30):
                scan_z += 0.012
                if not sensor.in_contact(ee[0]):
                    scan_x += 0.01
                    print(f"  scan #{i}: no contact → push x={scan_x:.2f}")
                try:
                    maneuver.goto(scan_x, scan_y, scan_z, 0,
                                  GOTO_DURATION)
                except Exception as exc:
                    print(f"  scan #{i}: goto fail ({exc})")
                    continue
                time.sleep(STEP_SLEEP)
                pos, att = read_state()
                ee = body_to_ee_tip(pos, att)
                pred = sensor.sense(ee[1], ee[2], ee[0])
                if pred is not None and pred['orient'] == expected_orient:
                    print(f"  scan #{i}: ee_z={ee[2]:.3f} → "
                          f"{pred['ridge_name']} d={pred['signed_d_m']*1000:+.1f}mm "
                          f"depth={pred['depth_m']*1000:.1f}mm ✓")
                    break
                elif pred is not None:
                    print(f"  scan #{i}: ee_z={ee[2]:.3f} → "
                          f"{pred['ridge_name']} (orient={pred['orient']}°, skip)")
                    pred = None
                else:
                    contact_str = "contact" if sensor.in_contact(ee[0]) else "no contact"
                    print(f"  scan #{i}: ee_z={ee[2]:.3f} ({contact_str})")

            if pred is None:
                print("[Phase 1] Ridge not found after scan — aborting.")
                raise KeyboardInterrupt

        # Setup goto: center on ridge + set x_ref=CONTACT_BODY_X
        # NOT calling set_current_state after — keeps AF pushing with
        # F = K_x * (1.55 - body_x) ≈ 17N toward wall.
        d_off = pred['signed_d_m']
        center_y, center_z = pos[1], pos[2]
        if abs(d_off) > 0.005:
            print(f"[Phase 1] Centering: d={d_off*1000:+.1f}mm")
            if pred['orient'] == 0:
                center_z -= d_off
            else:
                center_y -= d_off
        try:
            maneuver.goto(CONTACT_BODY_X, center_y, center_z, 0, GOTO_DURATION)
            time.sleep(GOTO_DURATION + 1.0)
            pos, att = read_state()
            ee = body_to_ee_tip(pos, att)
            pred_c = sensor.sense(ee[1], ee[2], ee[0])
            if pred_c:
                print(f"[Phase 1] Ready: d={pred_c['signed_d_m']*1000:+.1f}mm, "
                      f"depth={pred_c['depth_m']*1000:.1f}mm")
        except Exception as e:
            print(f"[Phase 1] Setup goto failed: {e}")

        # ===== PHASE 2: Velocity-based Tactile Servo =====
        # No set_current_state — x_ref stays at CONTACT_BODY_X.
        # AF spring pushes drone toward wall, maintaining contact.
        time.sleep(0.5)

        print(f"\n[Phase 2] Velocity servo — v = -λe → maneuver.velocity()")
        print(f"  lambda={SERVO_LAMBDA}, max_vel={MAX_SERVO_VEL*1000:.0f}mm/s")
        print(f"  Forward: {FORWARD_SPEED*1000:.0f}mm/s, loop: {STEP_SLEEP:.1f}s")
        print(f"  Max time: {max_servo_time:.0f}s")

        t0 = time.time()
        servo_step = 0
        ramp_step = 0
        no_ridge_count = 0
        backtrack_count = 0
        last_ridge_body_pos = pos.copy()

        while True:
            t_elapsed = time.time() - t0

            if t_elapsed > max_servo_time:
                print(f"\n[Phase 2] Timeout ({max_servo_time:.0f}s)")
                break
            if corner_det.corner_count >= 4:
                print(f"\n[Phase 2] Full rectangle complete!")
                break

            # 1. Read state
            pos, att = read_state()
            ee = body_to_ee_tip(pos, att)

            if (pos[0] > 2.5 or pos[0] < -0.5 or
                    pos[2] < 0.3 or pos[2] > 3.0 or
                    abs(pos[1]) > 5.0):
                print(f"\n[Phase 2] PHYSICS ANOMALY — stopping.")
                break

            # 2. Sense
            pred = sensor.sense(ee[1], ee[2], ee[0])

            # 3. Compute servo velocities
            vy_servo, vz_servo = 0.0, 0.0

            if pred is not None:
                no_ridge_count = 0
                last_ridge_body_pos = pos.copy()
                if corner_det.update(pred['orient']):
                    ramp_step = 0
                if corner_det.is_expected(pred['orient']):
                    vy_servo, vz_servo = servo.compute(pred)
            else:
                no_ridge_count += 1
                if no_ridge_count >= 8:
                    backtrack_count += 1
                    if backtrack_count >= 3:
                        print(f"\n[Phase 2] Lost ridge 3 times, stopping")
                        break
                    recovery_y = float(np.clip(last_ridge_body_pos[1], 0.05, 0.95))
                    recovery_z = last_ridge_body_pos[2]
                    print(f"  RECOVERY #{backtrack_count}: scanning from "
                          f"[{recovery_y:.3f},{recovery_z:.3f}]")
                    try:
                        maneuver.set_current_state()
                        time.sleep(0.2)
                        maneuver.goto(CONTACT_BODY_X, recovery_y,
                                      recovery_z, 0, 2.0)
                    except Exception:
                        pass
                    time.sleep(3.0)
                    reacquired = False
                    scan_base_z = recovery_z
                    scan_steps = [i * 0.012 for i in range(-15, 16)]
                    scan_steps.sort(key=lambda x: abs(x))
                    for scan_off in scan_steps:
                        scan_z = scan_base_z + scan_off
                        if scan_z < 0.8 or scan_z > 2.2:
                            continue
                        try:
                            maneuver.goto(CONTACT_BODY_X, recovery_y,
                                          scan_z, 0, GOTO_DURATION)
                        except Exception:
                            continue
                        time.sleep(1.5)
                        pos, att = read_state()
                        ee = body_to_ee_tip(pos, att)
                        pred_scan = sensor.sense(ee[1], ee[2], ee[0])
                        if pred_scan and corner_det.is_expected(pred_scan['orient']):
                            print(f"    scan {scan_off*1000:+.0f}mm: "
                                  f"d={pred_scan['signed_d_m']*1000:+.1f}mm ✓")
                            last_ridge_body_pos = pos.copy()
                            reacquired = True
                            break
                        else:
                            c = "contact" if sensor.in_contact(ee[0]) else "air"
                            print(f"    scan {scan_off*1000:+.0f}mm: ({c})")
                    no_ridge_count = 0
                    ramp_step = 0
                    servo_step += 1
                    time.sleep(0.5)
                    continue

            # 4. Forward velocity with ramp (avoid roll transients)
            fwd_vy, fwd_vz = corner_det.get_forward_velocity()
            ramp = min(ramp_step / max(RAMP_STEPS, 1), 1.0)
            fwd_vy *= ramp
            fwd_vz *= ramp
            if pred and pred['depth_m'] < LOW_DEPTH_THRESHOLD:
                vy_servo, vz_servo = 0.0, 0.0

            vy_total = vy_servo + fwd_vy
            vz_total = vz_servo + fwd_vz

            # 5. Send velocity command (vx=0: AF handles depth via compliance)
            try:
                maneuver.velocity(0, vy_total, vz_total, 0,
                                  0, 0, 0, VEL_DURATION)
            except Exception as e:
                print(f"  [velocity FAIL → resync]")
                try:
                    maneuver.set_current_state()
                    time.sleep(0.2)
                    maneuver.goto(CONTACT_BODY_X, pos[1], pos[2], 0,
                                  GOTO_DURATION)
                    time.sleep(GOTO_DURATION + 0.5)
                except Exception:
                    pass
                ramp_step = 0

            time.sleep(STEP_SLEEP)
            ramp_step += 1

            # 6. Log & print
            corr = np.array([0, vy_servo, vz_servo])
            log_step(t_elapsed, pos, ee, pred, corr,
                     np.array([pos[0], pos[1], pos[2]]), 2)

            if pred:
                v_cross = vz_servo if pred['orient'] == 0 else vy_servo
                print(f"  #{servo_step:3d} t={t_elapsed:5.1f}s  "
                      f"ee=[{ee[1]:.3f},{ee[2]:.3f}]  "
                      f"ridge={pred['ridge_name']:6s}  "
                      f"d={pred['signed_d_m']*1000:+6.1f}mm  "
                      f"depth={pred['depth_m']*1000:4.1f}mm  "
                      f"v={v_cross*1000:+5.1f}mm/s  "
                      f"fwd={corner_det.forward} ramp={ramp:.0%}")
            else:
                contact = "wall" if sensor.in_contact(ee[0]) else "air"
                print(f"  #{servo_step:3d} t={t_elapsed:5.1f}s  "
                      f"ee=[{ee[1]:.3f},{ee[2]:.3f}]  NO RIDGE ({contact})")

            servo_step += 1

        # ===== PHASE 3: Retract =====
        maneuver.set_current_state()
        time.sleep(0.5)
        print(f"\n[Phase 3] Retract → {RETRACT_POS}")
        maneuver.goto(*RETRACT_POS, 0, 4)
        time.sleep(10)

        print(f"[Phase 3] Land → {LAND_POS}")
        maneuver.goto(*LAND_POS, 0, 5)
        time.sleep(8)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("\nStopping...")
        stop()

        n = step
        save_dict = {k: v[:n] if isinstance(v, np.ndarray) else v
                     for k, v in log.items()}
        save_dict['servo_params'] = {
            'forward_speed': FORWARD_SPEED,
            'vel_duration': VEL_DURATION,
            'servo_lambda': SERVO_LAMBDA,
            'max_servo_vel': MAX_SERVO_VEL,
            'contact_body_x': CONTACT_BODY_X,
            'sensor_mode': sensor_mode,
            'noise_std': noise_std,
            'corners_detected': corner_det.corner_count,
            'depth_ref': DEPTH_REF,
            'depth_servo_mode': depth_servo.mode,
            'depth_vel_k': DEPTH_VEL_K,
        }

        # Always save to LOG_DIR (overwrites previous)
        np.savez(os.path.join(LOG_DIR, 'servo_data.npz'), **save_dict)
        print(f"\nLogs saved to: {LOG_DIR}")
        print(f"  servo_data.npz ({n} steps), corners: {corner_det.corner_count}")
        print(f"  Plot: python3 plot_servo.py")

        # If tag given, also copy everything to run_<tag>/
        if tag:
            tag_dir = os.path.join(LOG_DIR, f'run_{tag}')
            os.makedirs(tag_dir, exist_ok=True)
            np.savez(os.path.join(tag_dir, 'servo_data.npz'), **save_dict)
            for f in ['rotorcraft.log', 'pom.log', 'uavpos.log',
                       'uavatt.log', 'maneuver.log', 'phynt.log', 'opti.log']:
                src = os.path.join(LOG_DIR, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(tag_dir, f))
            print(f"  Tagged copy: {tag_dir}")
            print(f"  Plot: python3 plot_servo.py {tag}")

        print("=== Done ===")
