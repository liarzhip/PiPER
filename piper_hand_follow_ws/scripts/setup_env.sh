#!/usr/bin/env bash
set -e

export ORBBEC_PYSDK_ROOT="${ORBBEC_PYSDK_ROOT:-$HOME/PiPER_X/orbbec/pyorbbecsdk}"
export PYTHONPATH="$ORBBEC_PYSDK_ROOT/install/lib:${PYTHONPATH:-}"

source /opt/ros/jazzy/setup.bash
source "$HOME/agx_arm_ws/install/setup.bash"

if [ -f "$HOME/PiPER_X/piper_hand_follow_ws/install/setup.bash" ]; then
    source "$HOME/PiPER_X/piper_hand_follow_ws/install/setup.bash"
fi

echo "ROS_DISTRO=$ROS_DISTRO"
echo "ORBBEC_PYSDK_ROOT=$ORBBEC_PYSDK_ROOT"

python3 - <<'PY'
import pyorbbecsdk
print("pyorbbecsdk:", pyorbbecsdk.__file__)
PY