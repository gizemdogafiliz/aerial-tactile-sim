"""
Phase 3B: Tactile Servo Ridge Following — Plot & Analysis

Reads both:
  1. servo_data.npz — our servo loop log (EE pos, sensor, corrections, targets)
  2. TK3 component logs — pom.log, uavpos.log, uavatt.log, maneuver.log, phynt.log

Produces 5 figures:
  Fig 1: Servo performance (signed_d, depth, depth servo bias, orient, corrections)
  Fig 2: EE trajectory on wall plane (Y-Z) with ridge overlay
  Fig 3: Body trajectory tracking (actual vs desired, 3 axes + 3D)
  Fig 4: Contact forces (spring model from EE penetration)
  Fig 5: Wrench estimate and admittance filter effect

Usage:
  python3 plot_servo.py                         # latest run
  python3 plot_servo.py --run-dir /path/to/logs # specific run
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.expanduser('~/tk3lab-ws/logs/07-aerial-tactile-servo')

RAD2DEG = 180.0 / math.pi
X_WALL = 2.0
L_BAR = 0.6
TIP_RADIUS = 0.02
L_EFF = L_BAR  # SDF sphere center distance from body origin
EE_Z_OFFSET = -0.125
K_WALL = 100.0  # matches SDF kp (reduced for soft dome simulation)

RIDGE_X = 1.9975
RIDGES = [
    {'name': 'bottom', 'orient': 0,  'y0': 0.0, 'y1': 1.0, 'z': 0.875},
    {'name': 'top',    'orient': 0,  'y0': 0.0, 'y1': 1.0, 'z': 1.875},
    {'name': 'right',  'orient': 90, 'y': 0.0,  'z0': 0.875, 'z1': 1.875},
    {'name': 'left',   'orient': 90, 'y': 1.0,  'z0': 0.875, 'z1': 1.875},
]


def parse_pom_log(filepath):
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
    arr = np.array(rows)
    return {col: arr[:, i] for i, col in enumerate(header)}


def load_tk3_logs(log_dir):
    logs = {}
    for name in ['pom', 'uavpos', 'uavatt', 'maneuver', 'phynt']:
        path = os.path.join(log_dir, f'{name}.log')
        if os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                logs[name] = parse_pom_log(path)
                print(f"  Loaded {name}.log ({len(list(logs[name].values())[0])} samples)")
            except Exception as e:
                print(f"  Failed to parse {name}.log: {e}")
    return logs


def interp_to_time(log_dict, t_ref, t0, cols):
    log_t = log_dict['ts'] - t0
    return np.column_stack([np.interp(t_ref, log_t, log_dict[c]) for c in cols])


def body_to_ee(pos, att):
    qw, qx, qy, qz = att[:, 0], att[:, 1], att[:, 2], att[:, 3]
    r11 = 1 - 2*(qy*qy + qz*qz)
    r13 = 2*(qx*qz + qy*qw)
    r21 = 2*(qx*qy + qz*qw)
    r23 = 2*(qy*qz - qx*qw)
    r31 = 2*(qx*qz - qy*qw)
    r33 = 1 - 2*(qx*qx + qy*qy)
    ee = np.column_stack([
        pos[:, 0] + L_EFF * r11 + EE_Z_OFFSET * r13,
        pos[:, 1] + L_EFF * r21 + EE_Z_OFFSET * r23,
        pos[:, 2] + L_EFF * r31 + EE_Z_OFFSET * r33,
    ])
    return ee


def draw_ridges(ax, color='orange', lw=2, labels=True):
    for r in RIDGES:
        if r['orient'] == 0:
            ax.plot([r['y0'], r['y1']], [r['z'], r['z']],
                    color=color, lw=lw, label='ridges' if r['name'] == 'bottom' else None)
            if labels:
                ax.text((r['y0']+r['y1'])/2, r['z'], f"  {r['name']}",
                        fontsize=7, color=color, va='bottom', ha='center')
        else:
            ax.plot([r['y'], r['y']], [r['z0'], r['z1']],
                    color=color, lw=lw)
            if labels:
                ha = 'right' if r['y'] < 0.5 else 'left'
                ax.text(r['y'], (r['z0']+r['z1'])/2, f" {r['name']} ",
                        fontsize=7, color=color, va='center', ha=ha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', nargs='?', default=None, help='Run tag (e.g. "geometric_v1")')
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()

    if args.tag:
        log_dir = os.path.join(LOG_DIR, f'run_{args.tag}')
    else:
        log_dir = LOG_DIR

    # Load servo data
    servo_path = os.path.join(log_dir, 'servo_data.npz')
    if not os.path.exists(servo_path):
        print(f"No servo_data.npz in {log_dir}")
        return

    print(f"Loading servo data from: {servo_path}")
    sd = np.load(servo_path, allow_pickle=True)

    t_servo = sd['t']
    body_pos = sd['body_pos']
    ee_pos = sd['ee_pos']
    signed_d = sd['signed_d']
    depth = sd['depth']
    orient = sd['orient']
    correction = sd['correction']
    target_pos = sd['target_pos']
    phase = sd['phase']

    n_servo = len(t_servo)
    params = sd['servo_params'].item() if 'servo_params' in sd else {}
    print(f"  {n_servo} servo steps, duration {t_servo[-1]:.1f}s")
    if params:
        print(f"  Params: {params}")

    # Load TK3 logs
    print("\nLoading TK3 component logs...")
    tk3 = load_tk3_logs(log_dir)

    # Parse pom for full flight trajectory
    pom = tk3.get('pom')
    has_pom = pom is not None and 'ts' in pom
    if has_pom:
        t0_pom = pom['ts'][0]
        t_full = pom['ts'] - t0_pom
        pos_full = np.column_stack([pom['x'], pom['y'], pom['z']])
        roll_f = pom['roll']
        pitch_f = pom['pitch']
        yaw_f = pom['yaw']
        cy = np.cos(yaw_f*0.5);   sy = np.sin(yaw_f*0.5)
        cp = np.cos(pitch_f*0.5); sp = np.sin(pitch_f*0.5)
        cr = np.cos(roll_f*0.5);  sr = np.sin(roll_f*0.5)
        att_full = np.column_stack([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
        ee_full = body_to_ee(pos_full, att_full)
        vel_full = np.column_stack([pom['vx'], pom['vy'], pom['vz']])

    up = tk3.get('uavpos')
    mv = tk3.get('maneuver')

    # Plot directory
    if args.tag:
        plot_dir = os.path.join(SCRIPT_DIR, f'plots_{args.tag}')
    else:
        plot_dir = os.path.join(SCRIPT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ============================================================
    #  FIGURE 1: Servo Performance (5x1)
    # ============================================================
    fig1, axes1 = plt.subplots(5, 1, figsize=(14, 14))
    fig1.suptitle('Tactile Servo — Sensor & Control Performance', fontsize=14)

    # 1a: signed_d (lateral distance to ridge)
    ax = axes1[0]
    ax.plot(t_servo, signed_d, 'b-', lw=1, marker='.', ms=3, label='signed_d [mm]')
    ax.axhline(0, color='k', ls='--', lw=0.8, label='target (centered)')
    ax.fill_between(t_servo, -2, 2, alpha=0.1, color='green', label='±2mm band')
    ax.set_ylabel('Lateral Distance [mm]')
    ax.set_title('Signed Distance to Ridge Center (cross-track error)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b: depth + depth reference
    ax = axes1[1]
    ax.plot(t_servo, depth, 'r-', lw=1, marker='.', ms=3, label='depth [mm]')
    depth_ref_mm = params.get('depth_ref', 0.002) * 1000
    ax.axhline(depth_ref_mm, color='k', ls='--', lw=0.8, label=f'DEPTH_REF ({depth_ref_mm:.1f}mm)')
    ax.fill_between(t_servo, depth_ref_mm - 1, depth_ref_mm + 1,
                     alpha=0.1, color='red', label='±1mm band')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('Contact Depth (dome compression)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1c: depth servo — target_x bias
    ax = axes1[2]
    depth_servo_mode = params.get('depth_servo_mode', 'position')
    contact_body_x = params.get('contact_body_x', 1.55)
    tx_bias = (target_pos[:, 0] - contact_body_x) * 1000
    ax.plot(t_servo, tx_bias, 'm-', lw=1, marker='.', ms=3,
            label=f'target_x bias [mm] (mode={depth_servo_mode})')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    max_bias = params.get('depth_k_pos', 0.003) * 1000 if depth_servo_mode == 'position' else 3.0
    ax.axhline(max_bias, color='gray', ls=':', lw=0.5, label=f'±{max_bias:.0f}mm clamp')
    ax.axhline(-max_bias, color='gray', ls=':', lw=0.5)
    ax.set_ylabel('Bias [mm]')
    ax.set_title('Depth Servo — Body X Correction (resets each step, no accumulation)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1d: orientation
    ax = axes1[3]
    ax.plot(t_servo, orient, 'g-', lw=1, marker='.', ms=3, label='orient (0°=horiz, 90°=vert)')
    ax.set_ylabel('Orientation [deg]')
    ax.set_title('Detected Ridge Orientation')
    ax.set_yticks([0, 90])
    ax.set_yticklabels(['0° (horiz)', '90° (vert)'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1e: corrections (y-z cross-track)
    ax = axes1[4]
    ax.plot(t_servo, correction[:, 1]*1000, 'g-', lw=0.8, label='dy (lateral) [mm]')
    ax.plot(t_servo, correction[:, 2]*1000, 'b-', lw=0.8, label='dz (lateral) [mm]')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Correction [mm]')
    ax.set_title('Cross-Track Servo Corrections per Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, 'servo_performance.png'), dpi=150, bbox_inches='tight')

    # ============================================================
    #  FIGURE 2: EE Trajectory on Wall Plane (Y-Z)
    # ============================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    fig2.suptitle('Tactile Servo — End-Effector Trajectory on Wall', fontsize=14)

    # 2a: servo loop EE path
    ax = axes2[0]
    draw_ridges(ax, color='orange', lw=3)
    ax.plot(ee_pos[:, 1], ee_pos[:, 2], 'b-', lw=1.5, label='EE path (servo)')
    ax.plot(ee_pos[0, 1], ee_pos[0, 2], 'go', ms=8, label='start')
    ax.plot(ee_pos[-1, 1], ee_pos[-1, 2], 'rs', ms=8, label='end')
    # target_pos is body command, not EE — skip to avoid confusion on EE plot
    ax.set_xlabel('Y [m]  (← left | right →)')
    ax.set_ylabel('Z [m]')
    ax.set_title('Servo Loop — EE on Wall Plane (drone viewpoint)')
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2b: full flight EE path (from pom)
    ax = axes2[1]
    draw_ridges(ax, color='orange', lw=3)
    if has_pom:
        ax.plot(ee_full[:, 1], ee_full[:, 2], 'b-', lw=0.8, alpha=0.6, label='full flight EE')
        ax.plot(ee_full[0, 1], ee_full[0, 2], 'go', ms=8, label='takeoff')
        ax.plot(ee_full[-1, 1], ee_full[-1, 2], 'rs', ms=8, label='land')
    ax.set_xlabel('Y [m]  (← left | right →)')
    ax.set_ylabel('Z [m]')
    ax.set_title('Full Flight — EE on Wall Plane (drone viewpoint)')
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(os.path.join(plot_dir, 'ee_trajectory.png'), dpi=150, bbox_inches='tight')

    # ============================================================
    #  FIGURE 3: Body Trajectory Tracking (3+1)
    # ============================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Tactile Servo — Body Trajectory & Tracking', fontsize=14)

    if has_pom:
        # Desired from maneuver log
        has_mv = mv is not None and 'x' in mv
        if has_mv:
            p_nom = interp_to_time(mv, t_full, t0_pom, ['x', 'y', 'z'])

        # Desired from uavpos log (admittance-filtered)
        has_up = up is not None and 'xd' in up
        if has_up:
            p_des = interp_to_time(up, t_full, t0_pom, ['xd', 'yd', 'zd'])

        labels_xyz = [('X', 'red'), ('Y', 'green'), ('Z', 'blue')]
        for i, (lbl, clr) in enumerate(labels_xyz):
            ax = axes3[i // 2, i % 2] if i < 2 else axes3[1, 0]
            ax.plot(t_full, pos_full[:, i], color=clr, lw=0.8, label=f'{lbl} actual')
            if has_mv:
                ax.plot(t_full, p_nom[:, i], color=clr, ls='--', lw=0.5, alpha=0.6, label=f'{lbl} nominal')
            if has_up:
                ax.plot(t_full, p_des[:, i], color=clr, ls=':', lw=0.5, alpha=0.5, label=f'{lbl} desired (AF)')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{lbl} [m]')
            ax.set_title(f'{lbl} Position')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 3D trajectory
        ax = axes3[1, 1]
        ax.plot(pos_full[:, 1], pos_full[:, 2], 'k-', lw=0.8, label='body path')
        ax.plot(body_pos[:, 1], body_pos[:, 2], 'r-', lw=1.5, label='servo phase')
        ax.plot(pos_full[0, 1], pos_full[0, 2], 'go', ms=8, label='start')
        ax.set_xlabel('Y [m]  (← left | right →)')
        ax.set_ylabel('Z [m]')
        ax.set_title('Body Path (Y-Z plane, drone viewpoint)')
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        # Fallback: servo-only body trajectory
        for i, (lbl, clr) in enumerate([('X', 'red'), ('Y', 'green'), ('Z', 'blue')]):
            ax = axes3[i // 2, i % 2] if i < 2 else axes3[1, 0]
            ax.plot(t_servo, body_pos[:, i], color=clr, lw=1, label=f'{lbl} actual')
            ax.plot(t_servo, target_pos[:, i], color=clr, ls='--', lw=0.5, label=f'{lbl} target')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{lbl} [m]')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        ax = axes3[1, 1]
        ax.plot(body_pos[:, 1], body_pos[:, 2], 'k-', lw=1, label='body path')
        ax.set_xlabel('Y [m]  (← left | right →)'); ax.set_ylabel('Z [m]')
        ax.invert_xaxis()
        ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(plot_dir, 'body_trajectory.png'), dpi=150, bbox_inches='tight')

    # ============================================================
    #  FIGURE 4: Contact Force & Penetration
    # ============================================================
    fig4, axes4 = plt.subplots(3, 1, figsize=(14, 10))
    fig4.suptitle('Tactile Servo — Contact Force & Penetration', fontsize=14)

    if has_pom:
        pen = np.maximum(ee_full[:, 0] + TIP_RADIUS - X_WALL, 0.0)
        f_normal = K_WALL * pen
        pen_servo = np.maximum(ee_pos[:, 0] + TIP_RADIUS - X_WALL, 0.0)
        f_servo = K_WALL * pen_servo

        ax = axes4[0]
        ax.plot(t_full, f_normal, 'r-', lw=0.8, label='|F_contact| (spring model)')
        ax.set_ylabel('Force [N]')
        ax.set_title('Contact Normal Force (full flight)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes4[1]
        ax.plot(t_servo, f_servo, 'r-', lw=1, marker='.', ms=3, label='|F_contact| (servo phase)')
        ax.set_ylabel('Force [N]')
        ax.set_title('Contact Force During Servo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes4[2]
        ax.plot(t_full, ee_full[:, 0], 'r-', lw=0.8, label='EE_x')
        ax.axhline(X_WALL, color='k', ls='--', lw=0.8, label=f'wall x={X_WALL}')
        ax.axhline(RIDGE_X, color='orange', ls='--', lw=0.8, label=f'ridge x={RIDGE_X}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('EE_x [m]')
        ax.set_title('EE X Position vs Wall')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax = axes4[0]
        pen_servo = np.maximum(ee_pos[:, 0] + TIP_RADIUS - X_WALL, 0.0)
        f_servo = K_WALL * pen_servo
        ax.plot(t_servo, f_servo, 'r-', lw=1, label='|F_contact|')
        ax.set_ylabel('Force [N]'); ax.set_title('Contact Force'); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes4[1]
        ax.plot(t_servo, ee_pos[:, 0], 'r-', lw=1, label='EE_x')
        ax.axhline(X_WALL, color='k', ls='--', label='wall')
        ax.set_ylabel('[m]'); ax.legend(); ax.grid(True, alpha=0.3)
        axes4[2].axis('off')

    fig4.tight_layout()
    fig4.savefig(os.path.join(plot_dir, 'contact_force.png'), dpi=150, bbox_inches='tight')

    # ============================================================
    #  FIGURE 5: Nominal vs Admittance-Filtered + Velocity
    # ============================================================
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle('Tactile Servo — Admittance Filter Effect & Velocity', fontsize=14)

    if has_pom:
        has_mv = mv is not None and 'x' in mv
        has_up = up is not None and 'xd' in up

        # X: nominal vs actual (admittance effect most visible here)
        ax = axes5[0, 0]
        ax.plot(t_full, pos_full[:, 0], 'r-', lw=0.8, label='actual')
        if has_mv:
            ax.plot(t_full, p_nom[:, 0], 'r--', lw=0.5, alpha=0.6, label='nominal (maneuver)')
        if has_up:
            ax.plot(t_full, p_des[:, 0], 'r:', lw=0.5, alpha=0.5, label='desired (AF filtered)')
        ax.axhline(X_WALL - L_EFF, color='k', ls=':', lw=0.5, label=f'body at wall')
        ax.set_title('X: Nominal vs AF-Filtered')
        ax.set_ylabel('[m]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Admittance offset (nominal - actual)
        ax = axes5[0, 1]
        if has_mv:
            af_offset = (p_nom - pos_full) * 1000
            ax.plot(t_full, af_offset[:, 0], 'r-', lw=0.8, label='AF offset x [mm]')
            ax.plot(t_full, af_offset[:, 1], 'g-', lw=0.8, label='AF offset y [mm]')
            ax.plot(t_full, af_offset[:, 2], 'b-', lw=0.8, label='AF offset z [mm]')
        ax.set_title('Admittance Filter Offset (nominal - actual)')
        ax.set_ylabel('[mm]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Velocity
        ax = axes5[1, 0]
        ax.plot(t_full, vel_full[:, 0], 'r-', lw=0.8, label='vx')
        ax.plot(t_full, vel_full[:, 1], 'g-', lw=0.8, label='vy')
        ax.plot(t_full, vel_full[:, 2], 'b-', lw=0.8, label='vz')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[m/s]')
        ax.set_title('Body Velocity')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Speed magnitude
        ax = axes5[1, 1]
        speed = np.linalg.norm(vel_full, axis=1)
        ax.plot(t_full, speed, 'k-', lw=0.8, label='|v|')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[m/s]')
        ax.set_title('Speed Magnitude')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes5.flat:
            ax.text(0.5, 0.5, 'No pom.log', ha='center', va='center', transform=ax.transAxes)

    fig5.tight_layout()
    fig5.savefig(os.path.join(plot_dir, 'admittance_velocity.png'), dpi=150, bbox_inches='tight')

    print(f"\n5 figures saved to: {plot_dir}/")
    print(f"  servo_performance.png")
    print(f"  ee_trajectory.png")
    print(f"  body_trajectory.png")
    print(f"  contact_force.png")
    print(f"  admittance_velocity.png")

    if not args.save_only:
        plt.show()


if __name__ == '__main__':
    main()
