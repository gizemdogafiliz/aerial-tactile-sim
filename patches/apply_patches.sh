#!/bin/bash
# Apply ridge_4d modifications to tactile_gym_servo_control.
# Run from the aerial-tactile-sim project root after cloning servo_control.
#
# Usage:
#   cd ~/aerial-tactile-sim
#   git clone https://github.com/ac-93/tactile_gym_servo_control servo_control
#   cd servo_control && git checkout e02cd60 && cd ..
#   bash patches/apply_patches.sh

set -e

SERVO="servo_control/tactile_gym_servo_control"

if [ ! -d "$SERVO" ]; then
    echo "ERROR: $SERVO not found. Clone servo_control first (see README)."
    exit 1
fi

echo "=== 1. Adding ridge_4d task to learning_utils.py ==="
# Insert ridge_4d task after ridge_3d block (line 47)
UTILS="$SERVO/learning/learning_utils.py"
if grep -q "ridge_4d" "$UTILS"; then
    echo "  Already present, skipping."
else
    sed -i "/elif task_name == 'ridge_3d':/,/label_names = \['y', 'z', 'Rz'\]/ {
        /label_names = \['y', 'z', 'Rz'\]/a\\
\\
    elif task_name == 'ridge_4d':\\
        out_dim = 6\\
        label_names = ['y', 'z', 'Rx', 'Rz']
    }" "$UTILS"
    echo "  Done."
fi

echo "=== 2. Compatibility fix: collect_data.py (NumPy 2.x) ==="
COLLECT="$SERVO/data_collection/collect_data.py"
if grep -q "np.warnings" "$COLLECT"; then
    sed -i 's/import numpy as np/import numpy as np\nimport warnings/' "$COLLECT"
    sed -i 's/np\.warnings\.filterwarnings/warnings.filterwarnings/' "$COLLECT"
    echo "  Fixed np.warnings -> warnings."
else
    echo "  Already fixed or not needed."
fi

echo "=== 3. Compatibility fix: data_collection_utils.py (NumPy 2.x) ==="
DCU="$SERVO/data_collection/data_collection_utils.py"
if grep -q "np.hstack" "$DCU"; then
    sed -i 's/np\.hstack(\[pose, move\])/[*pose, *move]/' "$DCU"
    echo "  Fixed np.hstack -> list unpacking."
else
    echo "  Already fixed or not needed."
fi

echo "=== 4. Compatibility fix: servo_control.py (device parameter) ==="
SC="$SERVO/servo_control/servo_control.py"
if ! grep -q "device='cpu'" "$SC"; then
    sed -i "s/def run_servo_control(/def run_servo_control(\n    device='cpu',/" "$SC"
    echo "  Added device parameter."
else
    echo "  Already fixed."
fi

echo "=== 5. Copying wall_ridge URDF stimuli ==="
mkdir -p "$SERVO/stimuli/wall_ridge"
cp patches/stimuli/wall_ridge/*.urdf "$SERVO/stimuli/wall_ridge/"
echo "  Copied 3 URDF files."

echo "=== 6. Copying ridge_4d dataset metadata ==="
mkdir -p "$SERVO/data/ridge_4d/tap/train" "$SERVO/data/ridge_4d/tap/val"
cp patches/data/ridge_4d/tap/train/targets.csv "$SERVO/data/ridge_4d/tap/train/"
cp patches/data/ridge_4d/tap/train/collection_params.json "$SERVO/data/ridge_4d/tap/train/"
cp patches/data/ridge_4d/tap/val/targets.csv "$SERVO/data/ridge_4d/tap/val/"
cp patches/data/ridge_4d/tap/val/collection_params.json "$SERVO/data/ridge_4d/tap/val/"
echo "  Copied targets.csv + collection_params.json for train and val."

echo "=== 7. Copying trained ridge_4d models ==="
MODELS_SRC="output/trained_models/ridge_4d"
MODELS_DST="$SERVO/learned_models/ridge_4d"
if [ -d "$MODELS_SRC" ]; then
    cp -r "$MODELS_SRC"/* "$MODELS_DST/" 2>/dev/null || mkdir -p "$MODELS_DST"
    cp -r "$MODELS_SRC"/* "$MODELS_DST/"
    echo "  Copied trained models (nature_cnn, resnet, tap variants)."
else
    echo "  WARNING: $MODELS_SRC not found. Train models with scripts/07_train_ridge_cnn.py"
fi

echo ""
echo "All patches applied. Install with: cd servo_control && pip install --no-deps -e ."
