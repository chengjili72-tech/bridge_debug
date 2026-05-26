#!/bin/bash

# MindSpeed-Bridge Installation Script
# Usage:
#   bash install_auto.sh                  # Execute registration only (requires existing Megatron-Bridge)
#   bash install_auto.sh --download       # Download dependencies first, then execute registration

set -e

DOWNLOAD_MODE=false

if [ "$1" = "--download" ]; then
    DOWNLOAD_MODE=true
fi

MINDSPEED_BRIDGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(cd "${MINDSPEED_BRIDGE_DIR}/.." && pwd)"

# ================================================================
# Download Dependencies (--download mode only)
# Clone Megatron-LM, Megatron-Bridge, MindSpeed and install
# ================================================================
if [ "${DOWNLOAD_MODE}" = true ]; then
    echo ""
    echo "[Start] Download Dependencies"
    cd "${PARENT_DIR}"
    echo "Working directory: $(pwd)"
    echo ""

    # Clone and install Megatron-LM
    if [ -d "Megatron-LM" ]; then
        echo "[Skip] Megatron-LM directory already exists"
    else
        echo "[Clone] git clone https://github.com/NVIDIA/Megatron-LM.git"
        git clone https://github.com/NVIDIA/Megatron-LM.git
        cd Megatron-LM/
        echo "[Checkout] git checkout core_v0.16.1"
        git checkout core_v0.16.1
        echo "[Install] pip install -e ."
        pip install -e .
        cd "${PARENT_DIR}"
        echo ""
    fi

    # Clone Megatron-Bridge
    if [ -d "Megatron-Bridge" ]; then
        echo "[Skip] Megatron-Bridge directory already exists"
    else
        echo "[Clone] git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
        git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
        cd Megatron-Bridge/
        echo "[Checkout] git checkout v0.3.1"
        git checkout v0.3.1
        cd "${PARENT_DIR}"
        echo ""
    fi

    # Clone and install MindSpeed
    if [ -d "MindSpeed" ]; then
        echo "[Skip] MindSpeed directory already exists"
    else
        echo "[Clone] git clone https://gitcode.com/ascend/MindSpeed.git"
        git clone https://gitcode.com/ascend/MindSpeed.git
        cd MindSpeed/
        echo "[Checkout] git checkout core_r0.16.0"
        git checkout core_r0.16.0
        echo "[Install] pip install -r requirements.txt"
        pip install -r requirements.txt
        echo "[Install] pip install -e ."
        pip install -e .
        cd "${PARENT_DIR}"
        echo ""
    fi

    echo ""
    echo "[End] Dependencies Download Complete"
    echo ""
fi

# ================================================================
# Directory Validation
# ================================================================
MEGATRON_BRIDGE_DIR="${PARENT_DIR}/Megatron-Bridge"
MEGATRON_BRIDGE_DIR="$(cd "${MEGATRON_BRIDGE_DIR}" 2>/dev/null && pwd)" || true

echo ""
echo "Directory Structure"
echo "  Parent directory:    ${PARENT_DIR}"
echo "  MindSpeed-Bridge:    ${MINDSPEED_BRIDGE_DIR}"
echo "  Megatron-Bridge:     ${MEGATRON_BRIDGE_DIR}"
echo ""

if [ ! -d "${MEGATRON_BRIDGE_DIR}" ] || [ "${PARENT_DIR}" != "$(cd "${MEGATRON_BRIDGE_DIR}/.." && pwd)" ]; then
echo "[Error] Invalid directory structure"
    echo ""
    echo "Current structure:"
    echo "  MindSpeed-Bridge: ${MINDSPEED_BRIDGE_DIR}"
    if [ -d "${MEGATRON_BRIDGE_DIR}" ]; then
        echo "  Megatron-Bridge:  ${MEGATRON_BRIDGE_DIR}"
        echo "  MindSpeed-Bridge parent: ${PARENT_DIR}"
        echo "  Megatron-Bridge  parent: $(cd "${MEGATRON_BRIDGE_DIR}/.." && pwd)"
    else
        echo "  Megatron-Bridge:  Not found at ${MEGATRON_BRIDGE_DIR}"
    fi
    echo ""
    echo "Required directory structure:"
    echo "  parent_dir/"
    echo "  ├── Megatron-Bridge/"
    echo "  └── MindSpeed-Bridge/"
    echo ""
    echo "Tip: Use --download flag to automatically download dependencies"
    echo "  bash tools/install_auto.sh --download"
    exit 1
fi

cd "${MEGATRON_BRIDGE_DIR}"
echo "Current Working directory: $(pwd)"
echo ""

# ================================================================
# Transactional Installation: All steps must succeed, otherwise rollback
# ================================================================

# Rollback function
rollback_all() {
    echo ""
    echo "========================================"
    echo "Executing Rollback"
    echo "========================================"

    # Restore all backup files
    if [ -f "${MODELS_INIT}.backup" ]; then
        mv "${MODELS_INIT}.backup" "${MODELS_INIT}"
        echo "[Rollback] Restored ${MODELS_INIT}"
    fi

    if [ -f "${RECIPES_INIT}.backup" ]; then
        mv "${RECIPES_INIT}.backup" "${RECIPES_INIT}"
        echo "[Rollback] Restored ${RECIPES_INIT}"
    fi

    if [ -f "${RUN_RECIPE}.backup" ]; then
        mv "${RUN_RECIPE}.backup" "${RUN_RECIPE}"
        echo "  [Rollback] Restored ${RUN_RECIPE}"
    fi

    # Remove copied directory
    if [ -d "mindspeed_bridge" ]; then
        rm -rf mindspeed_bridge
        echo "[Rollback] Removed mindspeed_bridge directory"
    fi

    echo "========================================"
    echo "Rollback Complete"
    echo "========================================"
}

# Cleanup backup function
cleanup_backups() {
    rm -f "${MODELS_INIT}.backup" "${RECIPES_INIT}.backup" "${RUN_RECIPE}.backup"
}

# ================================================================
# Step 1: Copy Directory
# ================================================================
# Target: Megatron-Bridge/mindspeed_bridge/
# Description: Copy MindSpeed-Bridge/mindspeed_bridge folder to Megatron-Bridge root
# Content:
#   - Copy entire mindspeed_bridge directory with all subdirectories and files
#   - Includes models/, recipes/, examples/ subdirectories
# Purpose:
#   - Add Qwen3-VL model support to Megatron-Bridge
#   - Provide custom model implementations and training configurations

echo ""
echo "Step 1: Copy Directory"
echo "[Target] $(pwd)/mindspeed_bridge/"
echo ""

if [ -d "mindspeed_bridge" ]; then
    echo "[Skip] mindspeed_bridge directory already exists"
else
    if ! cp -r "${MINDSPEED_BRIDGE_DIR}/mindspeed_bridge" ./; then
echo "[Error] Failed to copy mindspeed_bridge directory"
        rollback_all
        exit 1
    fi
    echo "[Success] Copied ${MINDSPEED_BRIDGE_DIR}/mindspeed_bridge -> $(pwd)/mindspeed_bridge"
fi

# ================================================================
# Step 2: Register Models
# ================================================================
# Target: src/megatron/bridge/models/__init__.py
# Description: Add mindspeed_bridge.models import at the end of import section
# Content:
#   Append after all "from ... import ..." statements:
#     from mindspeed_bridge.models import *
# Purpose:
#   - Register all models defined in mindspeed_bridge
#   - Enable Megatron-Bridge to recognize and use Qwen3-VL and other custom models

echo ""
echo "Step 2: Register Models"

MODELS_INIT="src/megatron/bridge/models/__init__.py"
    echo "[Target] ${MODELS_INIT}"
echo ""

if [ ! -f "${MODELS_INIT}" ]; then
echo "[Error] File not found: ${MODELS_INIT}"
    rollback_all
    exit 1
fi

if grep -q 'from mindspeed_bridge.models import' "${MODELS_INIT}"; then
    echo "[Skip] mindspeed_bridge.models import already exists"
else
    # Backup file
    cp "${MODELS_INIT}" "${MODELS_INIT}.backup"

    # Append import statement at the end of file
    echo "" >> "${MODELS_INIT}"
    echo "from mindspeed_bridge.models import *" >> "${MODELS_INIT}"

    # Verify insertion
    if ! grep -q 'from mindspeed_bridge.models import' "${MODELS_INIT}"; then
echo "[Error] Verification failed - mindspeed_bridge.models import not found"
        rollback_all
        exit 1
    fi

    LINE=$(wc -l < "${MODELS_INIT}")
    echo "[Success] Added at line ${LINE}: from mindspeed_bridge.models import *"
fi


# ================================================================
# Step 3: Register Recipes
# ================================================================
# Target: src/megatron/bridge/recipes/__init__.py
# Description: Add mindspeed_bridge.recipes import at the end of import section
# Content:
#   Append after all "from ... import ..." statements:
#     from mindspeed_bridge.recipes import *
# Purpose:
#   - Register all training recipes defined in mindspeed_bridge
#   - Enable Megatron-Bridge to use Qwen3-VL training configurations

echo ""
echo "Step 3: Register Recipes"

RECIPES_INIT="src/megatron/bridge/recipes/__init__.py"
    echo "[Target] ${RECIPES_INIT}"
echo ""

if [ ! -f "${RECIPES_INIT}" ]; then
echo "[Error] File not found: ${RECIPES_INIT}"
    rollback_all
    exit 1
fi

if grep -q 'from mindspeed_bridge.recipes import' "${RECIPES_INIT}"; then
    echo "[Skip] mindspeed_bridge.recipes import already exists"
else
    # Backup file
    cp "${RECIPES_INIT}" "${RECIPES_INIT}.backup"

    # Append import statement at the end of file
    echo "" >> "${RECIPES_INIT}"
    echo "from mindspeed_bridge.recipes import *" >> "${RECIPES_INIT}"

    # Verify insertion
    if ! grep -q 'from mindspeed_bridge.recipes import' "${RECIPES_INIT}"; then
echo "[Error] Verification failed - mindspeed_bridge.recipes import not found"
        rollback_all
        exit 1
    fi

    LINE=$(wc -l < "${RECIPES_INIT}")
    echo "[Success] Added at line ${LINE}: from mindspeed_bridge.recipes import *"
fi

# ================================================================
# Step 4: Adapt Training Entry Point
# ================================================================
# Add MindSpeed adapter and Qwen3-VL forward function imports and registration.
# Note: Megatron-Bridge 0.3.1 does not natively support qwen35_vl, manual registration required.

echo ""
echo "Step 4: Adapt Training Entry Point"

RUN_RECIPE="scripts/training/run_recipe.py"
    echo "[Target] ${RUN_RECIPE}"
echo ""

if [ ! -f "${RUN_RECIPE}" ]; then
echo "[Error] File not found: ${RUN_RECIPE}"
    rollback_all
    exit 1
fi

# Backup file (only once)
if ! grep -q 'import mindspeed.megatron_adaptor' "${RUN_RECIPE}" || \
   ! grep -q 'from mindspeed_bridge.models.qwen_vl import' "${RUN_RECIPE}" || \
   ! grep -q '"qwen3_vl_step"' "${RUN_RECIPE}"; then
    cp "${RUN_RECIPE}" "${RUN_RECIPE}.backup"
fi

# ==============Mod 1: Add mindspeed.megatron_adaptor import==============
# Insert after "from typing import Callable" statement
if grep -q 'import mindspeed.megatron_adaptor' "${RUN_RECIPE}"; then
    echo "[Skip] mindspeed.megatron_adaptor import already exists"
else
    # Find line number of "from typing import Callable"
    TYPING_LINE=$(grep -n '^from typing import Callable' "${RUN_RECIPE}" | head -1 | cut -d: -f1)

    if [ -z "${TYPING_LINE}" ]; then
        echo "[Error] 'from typing import Callable' import statement not found"
        rollback_all
        exit 1
    fi

    # Insert after this line
    if ! sed -i "${TYPING_LINE}a\import mindspeed.megatron_adaptor" "${RUN_RECIPE}"; then
        echo "[Error] Failed to insert mindspeed.megatron_adaptor import"
        rollback_all
        exit 1
    fi

    # Verify insertion
    if ! grep -q 'import mindspeed.megatron_adaptor' "${RUN_RECIPE}"; then
        echo "[Error] Verification failed - mindspeed.megatron_adaptor import not found"
        rollback_all
        exit 1
    fi

    NEXT_LINE=$((TYPING_LINE + 1))
    echo "[Success] Added at line ${NEXT_LINE}: import mindspeed.megatron_adaptor"
fi

# ==============Mod 2: Add qwen3_vl_forward_step import==============
# Insert after vlm_forward_step import statement
if grep -q 'from mindspeed_bridge.models.qwen_vl import' "${RUN_RECIPE}"; then
    echo "[Skip] qwen3_vl_forward_step import already exists"
else
    # Find line number of vlm_forward_step import
    VLM_LINE=$(grep -n 'from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step' "${RUN_RECIPE}" | head -1 | cut -d: -f1)

    if [ -z "${VLM_LINE}" ]; then
echo "[Error] vlm_forward_step import statement not found"
        rollback_all
        exit 1
    fi

    # Insert after this line
    if ! sed -i "${VLM_LINE}a\from mindspeed_bridge.models.qwen_vl import qwen3_vl_forward_step" "${RUN_RECIPE}"; then
echo "[Error] Failed to insert qwen3_vl_forward_step import"
        rollback_all
        exit 1
    fi

    # Verify insertion
    if ! grep -q 'from mindspeed_bridge.models.qwen_vl import' "${RUN_RECIPE}"; then
echo "[Error] Verification failed - qwen3_vl_forward_step import not found"
        rollback_all
        exit 1
    fi

    NEXT_LINE=$((VLM_LINE + 1))
    echo "[Success] Added at line ${NEXT_LINE}: from mindspeed_bridge.models.qwen_vl import qwen3_vl_forward_step"
fi

# ===========Mod 3: Register forward function==============
# Append "qwen3_vl_step" entry after "llava_step" in STEP_FUNCTIONS dictionary
if grep -q '"qwen3_vl_step"' "${RUN_RECIPE}"; then
    echo "[Skip] qwen3_vl_step already registered in STEP_FUNCTIONS"
else
    # Find line number of llava_step
    LLAVA_LINE=$(grep -n '"llava_step"' "${RUN_RECIPE}" | head -1 | cut -d: -f1)

    if [ -z "${LLAVA_LINE}" ]; then
echo "[Error] llava_step entry not found"
        rollback_all
        exit 1
    fi

    # Insert after this line
    if ! sed -i "${LLAVA_LINE}a\    \"qwen3_vl_step\": qwen3_vl_forward_step," "${RUN_RECIPE}"; then
echo "[Error] Failed to insert qwen3_vl_step registration"
        rollback_all
        exit 1
    fi

    # Verify insertion
    if ! grep -q '"qwen3_vl_step"' "${RUN_RECIPE}"; then
echo "[Error] Verification failed - qwen3_vl_step registration not found"
        rollback_all
        exit 1
    fi

    NEXT_LINE=$((LLAVA_LINE + 1))
    echo "[Success] Added at line ${NEXT_LINE}: \"qwen3_vl_step\": qwen3_vl_forward_step"
fi

# ===========Mod 4: Insert mindspeed repatch initialization code==============
# Insert repatch code after process_config_with_overrides
if grep -q 'from mindspeed.megatron_adaptor import repatch' "${RUN_RECIPE}"; then
    echo "[Skip] repatch code already exists"
else
    CONFIG_LINE=$(grep -n 'process_config_with_overrides' "${RUN_RECIPE}" | tail -1 | cut -d: -f1)
    if [ -z "${CONFIG_LINE}" ]; then
        echo "[Error] process_config_with_overrides line not found"
        rollback_all
        exit 1
    fi

    sed -i "${CONFIG_LINE}a\from mindspeed.megatron_adaptor import repatch" "${RUN_RECIPE}"
    sed -i "$((CONFIG_LINE+1))a\from dataclasses import asdict" "${RUN_RECIPE}"
    sed -i "$((CONFIG_LINE+2))a\repatch(asdict(config.model))" "${RUN_RECIPE}"

    if ! grep -q 'from mindspeed.megatron_adaptor import repatch' "${RUN_RECIPE}"; then
        echo "[Error] Verification failed - repatch code not found"
        rollback_all
        exit 1
    fi
    echo "[Success] Added repatch initialization code"
fi

# All Steps Completed Successfully - Cleanup Backup Files
cleanup_backups

# Installation Complete
echo ""
echo "========================================"
echo "MindSpeed-Bridge Installation Complete"
echo "========================================"
echo ""
echo "Next Steps:"
echo "  1. switch to ${MEGATRON_BRIDGE_DIR}"
echo "  2. Configure dataset and model weights in mindspeed_bridge/examples/ scripts"
echo "  3. Start training with your configured script"
echo ""
echo "Example:"
echo "  bash mindspeed_bridge/examples/models/vlm/qwen35_vl/qwen35_vl_35b_sft.sh"
echo ""
