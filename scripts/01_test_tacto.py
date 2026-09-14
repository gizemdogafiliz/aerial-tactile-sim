"""
Phase 1, Step 1: Test TACTO — render a DIGIT tactile image.

Pushes a small sphere into a DIGIT sensor and saves the resulting
tactile image. Runs headless (no GUI). Uses raw PyBullet (no pybulletX).

Usage:
    cd ~/aerial-tactile-sim
    python scripts/01_test_tacto.py
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pybullet as p
import cv2

TACTO_ROOT = os.path.join(os.path.dirname(__file__), "..", "tacto")
sys.path.insert(0, TACTO_ROOT)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Monkey-patch for Python 3.12 + NumPy 2.x compat with old packages
import collections
import collections.abc
for attr in ("Mapping", "MutableMapping", "Sequence", "MutableSequence"):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import numpy as _np
for old_alias in ("float", "int", "bool", "complex", "object", "str"):
    if not hasattr(_np, old_alias) or isinstance(getattr(_np, old_alias), type):
        continue
    # np.float, np.int etc. removed in NumPy 2.0 — restore as builtins
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "bool"):
    _np.bool = bool

import tacto

# Start PyBullet headless
physics_client = p.connect(p.DIRECT)
p.setGravity(0, 0, 0)

# Load DIGIT sensor URDF
digit_urdf = os.path.join(TACTO_ROOT, "meshes", "digit.urdf")
digit_id = p.loadURDF(
    digit_urdf,
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, -np.pi/2, 0]),
    useFixedBase=True
)

# Load sphere object
sphere_urdf = os.path.join(TACTO_ROOT, "examples", "objects", "sphere_small.urdf")
sphere_id = p.loadURDF(
    sphere_urdf,
    basePosition=[-0.015, 0, 0.035],
    globalScaling=0.15
)

# Create TACTO sensor (use background image for realistic rendering)
bg_path = os.path.join(TACTO_ROOT, "examples", "conf", "bg_digit_240_320.jpg")
bg = cv2.imread(bg_path)

digit_sensor = tacto.Sensor(width=120, height=160, visualize_gui=False, background=bg)
digit_sensor.add_camera(digit_id, [-1])

# TACTO needs a pybulletX.Body-like object for add_body.
# Create a minimal wrapper.
class FakeBody:
    def __init__(self, body_id, urdf_path, global_scaling=1.0):
        self.id = body_id
        self.urdf_path = urdf_path
        self.global_scaling = global_scaling

fake_sphere = FakeBody(sphere_id, sphere_urdf, 0.15)
digit_sensor.add_body(fake_sphere)

print("Rendering tactile images at different penetration depths...")
print(f"Output: {OUTPUT_DIR}\n")

depths = [0.035, 0.030, 0.025, 0.020, 0.015]
for i, z in enumerate(depths):
    p.resetBasePositionAndOrientation(sphere_id, [-0.015, 0, z], [0, 0, 0, 1])
    for _ in range(5):
        p.stepSimulation()

    color, depth = digit_sensor.render()
    tactile_img = color[0]

    filename = f"tactile_depth_{i}_{z:.3f}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), tactile_img)

    contacts = p.getContactPoints(digit_id, sphere_id)
    print(f"  z={z:.3f}m | contacts={len(contacts)} | shape={tactile_img.shape} | {filename}")

# No-contact reference
p.resetBasePositionAndOrientation(sphere_id, [-0.015, 0, 0.1], [0, 0, 0, 1])
p.stepSimulation()
color_ref, _ = digit_sensor.render()
cv2.imwrite(os.path.join(OUTPUT_DIR, "tactile_no_contact.png"), color_ref[0])
print(f"  No-contact reference saved: tactile_no_contact.png")

p.disconnect()
print("\nDone! Compare no_contact vs contact images to see gel deformation.")
