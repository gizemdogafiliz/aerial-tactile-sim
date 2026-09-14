"""
Phase 2: Aerial Tactile — Gazebo Simulation with Contact Sensor
Based on Assignment 06a, extended with contact sensor data logging.

Same architecture as 06a (phynt AF + WO), plus:
  - Gazebo contact sensor on ee-tip-col publishes to gz topic
  - This script reads contact data via subprocess (gz topic -e)
  - After simulation, parses component log files for flight data
    (phynt genomix port reads broken in this Docker — phynt-1.2.1)

Usage:
  Terminal 1: sh simulation.sh
  Terminal 2: python3 -i model_hexa_fa_tactile.py
              >>> simulation()
"""

import genomix
import math
import numpy as np
import os
import time
import shutil
import subprocess
import re

# ############################################
#  PARAMETERS
# ############################################

MASS = 2.72  # base 2.3 + 6 rotors × 0.07 (from mrsim-rotor SDF)
J = [0.0115, 0, 0, 0, 0.0114, 0, 0, 0, 0.0194]
ARMLEN = 0.38998
CF = 9.9016e-4
CT = 1.9e-5

X_WALL = 2.0
L_BAR = 0.6
TIP_RADIUS = 0.02
L_EFF = L_BAR + TIP_RADIUS
EE_Z_OFFSET = -0.125
K_WALL = 500.0

BODY_X_CONTACT = X_WALL - L_BAR + 0.30

LOG_DIR = '/shared-workspace/logs/07-aerial-tactile'
os.makedirs(LOG_DIR, exist_ok=True)

# ############################################
#  GENOMIX SETUP
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
    print("Waiting 5s for Gazebo plugins to load...")
    time.sleep(5)
    print("Connecting components...")

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
    af_K = [25.0, 100.0, 100.0, 50.0, 50.0, 50.0]
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
    pom.log_measurements('/tmp/pom-measurements.log')
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
    print("Started: rotors armed, controllers running, AF/WO enabled.")


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

    for f in ['rotorcraft.log', 'pom.log', 'pom-measurements.log',
              'uavpos.log', 'uavatt.log', 'maneuver.log', 'phynt.log',
              'opti.log']:
        src = '/tmp/' + f
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(LOG_DIR, f))
    print(f"Logs saved to: {LOG_DIR}")


# ############################################
#  REAL-TIME DATA READERS
# ############################################

def read_state():
    """Read current state from pom (works in this Docker)."""
    try:
        s = pom.frame('robot')['frame']
        pos = np.array([s['pos']['x'], s['pos']['y'], s['pos']['z']])
        att = np.array([s['att']['qw'], s['att']['qx'],
                        s['att']['qy'], s['att']['qz']])
        vel = np.array([s['vel']['vx'], s['vel']['vy'], s['vel']['vz']])
        avel = np.array([s['avel']['wx'], s['avel']['wy'], s['avel']['wz']])
        return pos, att, vel, avel
    except Exception as e:
        print(f"[read_state ERROR] {e}")
        return np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)


# ############################################
#  LOG FILE PARSER (same approach as plot_06a.py)
# ############################################

def parse_log(filepath):
    """Parse column-format GenoM3 log file → dict of arrays."""
    header = None
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = line.split()
                continue
            parts = line.split()
            if len(parts) == len(header):
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue
    if not rows:
        return None
    arr = np.array(rows)
    return {col: arr[:, i] for i, col in enumerate(header)}


def post_process_logs():
    """Parse component logs after simulation. Returns dict with all flight data."""
    result = {}

    # pom.log → position, attitude, velocity
    pom_path = os.path.join(LOG_DIR, 'pom.log')
    if os.path.exists(pom_path):
        pom_data = parse_log(pom_path)
        if pom_data:
            ts = pom_data['ts']
            t = ts - ts[0]
            result['t_pom'] = t
            result['pos_pom'] = np.column_stack([pom_data['x'], pom_data['y'], pom_data['z']])
            result['vel_pom'] = np.column_stack([pom_data['vx'], pom_data['vy'], pom_data['vz']])
            result['euler_pom'] = np.column_stack([pom_data['roll'], pom_data['pitch'], pom_data['yaw']])
            print(f"  pom.log: {len(t)} samples, {t[-1]:.1f}s")

    # uavpos.log → AF-filtered desired (xd, yd, zd), controller force
    uavpos_path = os.path.join(LOG_DIR, 'uavpos.log')
    if os.path.exists(uavpos_path):
        up = parse_log(uavpos_path)
        if up and 'xd' in up:
            result['pos_d_log'] = np.column_stack([up['xd'], up['yd'], up['zd']])
            result['t_uavpos'] = up['ts'] - up['ts'][0]
            if 'fx' in up:
                result['f_ctrl'] = np.column_stack([up['fx'], up['fy'], up['fz']])
            print(f"  uavpos.log: {len(up['ts'])} samples")

    # maneuver.log → nominal trajectory
    man_path = os.path.join(LOG_DIR, 'maneuver.log')
    if os.path.exists(man_path):
        mv = parse_log(man_path)
        if mv and 'x' in mv:
            result['p_nom_log'] = np.column_stack([mv['x'], mv['y'], mv['z']])
            result['t_maneuver'] = mv['ts'] - mv['ts'][0]
            print(f"  maneuver.log: {len(mv['ts'])} samples")

    # Compute EE position and idealized contact force from pom data
    if 'pos_pom' in result and 'euler_pom' in result:
        pos = result['pos_pom']
        roll, pitch, yaw = result['euler_pom'][:, 0], result['euler_pom'][:, 1], result['euler_pom'][:, 2]
        cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
        cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
        cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        r11 = 1 - 2*(qy*qy + qz*qz)
        r13 = 2*(qx*qz + qy*qw)
        r23 = 2*(qy*qz - qx*qw)
        r33 = 1 - 2*(qx*qx + qy*qy)
        p_ee_x = pos[:, 0] + L_EFF * r11 + EE_Z_OFFSET * r13
        result['p_ee'] = np.column_stack([
            p_ee_x,
            pos[:, 1] + L_EFF * 2*(qx*qy + qz*qw) + EE_Z_OFFSET * r23,
            pos[:, 2] + L_EFF * 2*(qx*qz - qy*qw) + EE_Z_OFFSET * r33,
        ])
        penetration = np.maximum(p_ee_x - X_WALL, 0.0)
        f_normal = K_WALL * penetration
        result['f_spring'] = np.column_stack([-f_normal, np.zeros_like(f_normal), np.zeros_like(f_normal)])
        print(f"  EE penetration: max {penetration.max()*1000:.1f}mm, max force {f_normal.max():.1f}N")

    return result


# ############################################
#  CONTACT SENSOR READER
# ############################################

CONTACT_TOPIC = '/world/mrsim/model/hr6/link/base/sensor/ee_contact_sensor/contact'


class ContactReader:
    """Background reader for gz contact topic. Starts once, reads continuously."""

    def __init__(self, topic):
        self.topic = topic
        self.proc = None
        self.latest_pos = np.zeros(3)
        self.latest_normal = np.zeros(3)
        self.latest_force = np.zeros(3)
        self.in_contact = False
        self._buf = ''

    def start(self):
        self.proc = subprocess.Popen(
            ['gz', 'topic', '-e', '-t', self.topic],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        import fcntl
        fd = self.proc.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None

    @staticmethod
    def _safe_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def update(self):
        if not self.proc:
            return
        got_new = False
        try:
            chunk = self.proc.stdout.read(8192)
            if chunk:
                self._buf += chunk
                got_new = True
        except (BlockingIOError, IOError):
            pass

        if not got_new:
            self.in_contact = False
            return

        if 'position' not in self._buf:
            self.in_contact = False
            self._buf = ''
            return

        last_pos = self._buf.rfind('position')
        if last_pos < 0:
            return

        start = self._buf.rfind('contact {', 0, last_pos)
        if start < 0:
            start = max(0, last_pos - 500)
        msg = self._buf[start:]
        self._buf = self._buf[last_pos:]

        try:
            pos_match = re.search(
                r'position\s*\{[^}]*x:\s*([-\d.e+]+)[^}]*y:\s*([-\d.e+]+)[^}]*z:\s*([-\d.e+]+)',
                msg, re.DOTALL)
            if pos_match:
                self.latest_pos = np.array([self._safe_float(pos_match.group(i)) for i in (1, 2, 3)])
                self.in_contact = True

            normal_match = re.search(r'normal\s*\{[^}]*x:\s*([-\d.e+]+)', msg, re.DOTALL)
            if normal_match:
                self.latest_normal[0] = self._safe_float(normal_match.group(1))

            force_match = re.search(
                r'body_1_wrench\s*\{[^}]*force\s*\{[^}]*x:\s*([-\d.e+]+)',
                msg, re.DOTALL)
            if force_match:
                self.latest_force[0] = self._safe_float(force_match.group(1))
                fy = re.search(r'body_1_wrench\s*\{[^}]*force\s*\{[^}]*y:\s*([-\d.e+]+)', msg, re.DOTALL)
                fz = re.search(r'body_1_wrench\s*\{[^}]*force\s*\{[^}]*z:\s*([-\d.e+]+)', msg, re.DOTALL)
                if fy: self.latest_force[1] = self._safe_float(fy.group(1))
                if fz: self.latest_force[2] = self._safe_float(fz.group(1))
        except Exception:
            pass

    def read(self):
        self.update()
        return self.in_contact, self.latest_pos.copy(), self.latest_normal.copy(), self.latest_force.copy()


# ############################################
#  WAYPOINT SEQUENCE — square pattern on wall
# ############################################

waypoints = [
    (2.0,   (0, 0, 1.0, 0, 5)),                      # hover
    (10.0,  (1.2, 0.025, 1.025, 0, 4)),              # approach
    (22.0,  (BODY_X_CONTACT, 0.025, 1.025, 0, 2)),   # contact bottom-left
    (30.0,  (BODY_X_CONTACT, 0.975, 1.025, 0, 5)),   # bottom-right
    (38.0,  (BODY_X_CONTACT, 0.975, 1.975, 0, 5)),   # top-right
    (46.0,  (BODY_X_CONTACT, 0.025, 1.975, 0, 5)),   # top-left
    (54.0,  (BODY_X_CONTACT, 0.025, 1.025, 0, 5)),   # close square
    (62.0,  (1.0, 0.025, 1.025, 0, 4)),              # retract
    (69.0,  (0, 0, 0, 0, 5)),                        # land
]


def simulation():
    print("=== 07 Aerial Tactile — Gazebo simulation with contact sensor ===")
    setup()
    start()

    tf = 80.0
    dt_log = 0.02   # 50Hz logging
    N = int(tf / dt_log)

    t_log = np.zeros(N)
    pos_log = np.zeros((N, 3))
    att_log = np.zeros((N, 4))
    vel_log = np.zeros((N, 3))
    avel_log = np.zeros((N, 3))

    contact_flag_log = np.zeros(N, dtype=bool)
    contact_pos_log = np.zeros((N, 3))
    contact_normal_log = np.zeros((N, 3))
    contact_force_log = np.zeros((N, 3))

    contact_reader = ContactReader(CONTACT_TOPIC)
    contact_reader.start()
    print("Contact sensor reader started.")

    wo_calibrated = False
    wp_idx = 0
    t0 = time.time()

    for i in range(N):
        ts = i * dt_log

        if not wo_calibrated and ts >= 20.0:
            print(f"[t={ts:.1f}s] calibrating WO...")
            phynt.set_wo_zero(duration=2.0)
            wo_calibrated = True

        if wp_idx < len(waypoints) and ts >= waypoints[wp_idx][0]:
            t_wp, args = waypoints[wp_idx]
            print(f"[t={ts:.1f}s] maneuver.goto{args}")
            maneuver.goto(*args)
            wp_idx += 1

        pos, att, vel, avel = read_state()

        t_log[i] = ts
        pos_log[i] = pos
        att_log[i] = att
        vel_log[i] = vel
        avel_log[i] = avel

        in_contact, c_pos, c_normal, c_force = contact_reader.read()
        if in_contact:
            contact_flag_log[i] = True
            contact_pos_log[i] = c_pos
            contact_normal_log[i] = c_normal
            contact_force_log[i] = c_force

        if i % 50 == 0:
            c_str = f"  contact_f: [{c_force[0]:.2f}, {c_force[1]:.2f}, {c_force[2]:.2f}]" if contact_flag_log[i] else "  no contact"
            print(f"t: {ts:5.1f}  pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]{c_str}")

        elapsed = time.time() - t0
        target = (i + 1) * dt_log
        if target > elapsed:
            time.sleep(target - elapsed)

    print("Stopping...")
    contact_reader.stop()
    stop()

    n_contacts = contact_flag_log.sum()
    print(f"Contact samples logged: {n_contacts} / {N}")

    # Post-process: parse component log files for AF/WO/nominal data
    print("\nParsing component log files...")
    log_data = post_process_logs()

    # Save everything
    save_dict = dict(
        t=t_log, pos=pos_log, att=att_log, vel=vel_log, avel=avel_log,
        contact_flag=contact_flag_log,
        contact_pos=contact_pos_log,
        contact_normal=contact_normal_log,
        contact_force=contact_force_log,
    )
    for key, val in log_data.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val

    np.savez(os.path.join(LOG_DIR, 'simulation_data.npz'), **save_dict)
    print(f"Data saved: {LOG_DIR}/simulation_data.npz")
    print("=== Simulation complete ===")


# To run:
#   python3 -i model_hexa_fa_tactile.py
#   >>> simulation()
