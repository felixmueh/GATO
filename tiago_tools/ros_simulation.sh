#!/usr/bin/env bash
# Source this file to select the same-host PAL simulator in the current shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source tiago_tools/ros_simulation.sh' >&2
    exit 1
fi

# Use the same PAL interface overlay as the real robot, with different DDS
# settings. Source ROS first so the selected profile remains authoritative.
if [[ -n "${GATO_ROS_SETUP:-}" ]]; then
    if [[ ! -r "${GATO_ROS_SETUP}" ]]; then
        echo "Missing ROS setup: ${GATO_ROS_SETUP}; rebuild the TIAGo image." >&2
        return 1
    fi
    source "${GATO_ROS_SETUP}" || return 1
fi

_gato_ros_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
if [[ ! -r "${_gato_ros_config_dir}/cyclone_pal_loopback.xml" ]]; then
    echo 'Cannot read the simulation Cyclone DDS configuration.' >&2
    unset _gato_ros_config_dir
    return 1
fi

export GATO_ROS_PROFILE=simulation
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${_gato_ros_config_dir}/cyclone_pal_loopback.xml"
export ROS_LOCALHOST_ONLY=0
unset _gato_ros_config_dir
printf 'ROS profile: simulation (domain 1, loopback only)\n'
