#!/bin/bash
# Download required models and data for the avatar pipeline.
#
# Some downloads require manual steps (FLAME model needs registration).
# This script handles what can be automated.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
VENDOR_DIR="$PROJECT_ROOT/vendor"

echo "=== Avatar Pipeline: Model & Data Download ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# 1. MediaPipe FaceLandmarker model
echo "--- 1. MediaPipe FaceLandmarker Model ---"
MEDIAPIPE_MODEL="$DATA_DIR/face_landmarker_v2_with_blendshapes.task"
if [ -f "$MEDIAPIPE_MODEL" ]; then
    echo "  Already exists: $MEDIAPIPE_MODEL"
else
    echo "  Downloading..."
    curl -L -o "$MEDIAPIPE_MODEL" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    echo "  Done: $MEDIAPIPE_MODEL"
fi

# 2. Clone DECA
echo ""
echo "--- 2. DECA Repository ---"
DECA_DIR="$VENDOR_DIR/deca"
if [ -d "$DECA_DIR" ]; then
    echo "  Already exists: $DECA_DIR"
else
    echo "  Cloning DECA..."
    git clone https://github.com/yfeng95/DECA "$DECA_DIR"
    echo "  Done"
fi

# 3. Clone Deformation Transfer ARKit
echo ""
echo "--- 3. Deformation Transfer ARKit Repository ---"
DT_DIR="$VENDOR_DIR/deformation_transfer"
if [ -d "$DT_DIR" ]; then
    echo "  Already exists: $DT_DIR"
else
    echo "  Cloning..."
    git clone https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes "$DT_DIR"
    echo "  Done"
fi

# 4. Clone FLAME_PyTorch
echo ""
echo "--- 4. FLAME_PyTorch Repository ---"
FLAME_PT_DIR="$VENDOR_DIR/flame_pytorch"
if [ -d "$FLAME_PT_DIR" ]; then
    echo "  Already exists: $FLAME_PT_DIR"
else
    echo "  Cloning..."
    git clone https://github.com/soubhiksanyal/FLAME_PyTorch "$FLAME_PT_DIR"
    echo "  Done"
fi

# 5. FLAME model (manual)
echo ""
echo "--- 5. FLAME Model (MANUAL DOWNLOAD REQUIRED) ---"
FLAME_MODEL="$DATA_DIR/flame/generic_model.pkl"
if [ -f "$FLAME_MODEL" ]; then
    echo "  Found: $FLAME_MODEL"
else
    echo "  *** MANUAL STEP REQUIRED ***"
    echo "  1. Go to: https://flame.is.tue.mpg.de/"
    echo "  2. Register and sign the license agreement"
    echo "  3. Download 'FLAME 2020' model"
    echo "  4. Extract generic_model.pkl to: $DATA_DIR/flame/"
    echo "  5. Also download FLAME_texture.npz to: $DATA_DIR/flame/"
fi

# 6. DECA pretrained models (manual)
echo ""
echo "--- 6. DECA Pretrained Models (MANUAL DOWNLOAD REQUIRED) ---"
DECA_MODEL="$DATA_DIR/deca/deca_model.tar"
if [ -f "$DECA_MODEL" ]; then
    echo "  Found: $DECA_MODEL"
else
    echo "  *** MANUAL STEP REQUIRED ***"
    echo "  1. Follow instructions in $DECA_DIR/README.md"
    echo "  2. Run: cd $DECA_DIR && bash fetch_data.sh"
    echo "  3. Copy deca_model.tar to: $DATA_DIR/deca/"
fi

# 7. Extract ARKit reference blendshapes
echo ""
echo "--- 7. ARKit Reference Blendshapes ---"
ARKIT_REF="$DATA_DIR/arkit_reference"
if [ "$(ls -A $ARKIT_REF 2>/dev/null)" ]; then
    echo "  Already populated: $ARKIT_REF"
else
    if [ -d "$DT_DIR/data/ARKit_blendShapes" ]; then
        echo "  Copying from DT repo..."
        cp -r "$DT_DIR/data/ARKit_blendShapes/"* "$ARKIT_REF/"
        echo "  Done"
    else
        echo "  ARKit reference meshes not found in DT repo."
        echo "  They will be available after cloning the DT repo."
    fi
fi

echo ""
echo "=== Download Summary ==="
echo "  MediaPipe model:  $([ -f "$MEDIAPIPE_MODEL" ] && echo 'OK' || echo 'MISSING')"
echo "  DECA repo:        $([ -d "$DECA_DIR" ] && echo 'OK' || echo 'MISSING')"
echo "  DT ARKit repo:    $([ -d "$DT_DIR" ] && echo 'OK' || echo 'MISSING')"
echo "  FLAME_PyTorch:    $([ -d "$FLAME_PT_DIR" ] && echo 'OK' || echo 'MISSING')"
echo "  FLAME model:      $([ -f "$FLAME_MODEL" ] && echo 'OK' || echo 'MANUAL DOWNLOAD NEEDED')"
echo "  DECA pretrained:  $([ -f "$DECA_MODEL" ] && echo 'OK' || echo 'MANUAL DOWNLOAD NEEDED')"
echo "  ARKit references: $([ "$(ls -A $ARKIT_REF 2>/dev/null)" ] && echo 'OK' || echo 'PENDING')"
echo ""
