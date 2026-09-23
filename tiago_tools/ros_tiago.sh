#!/usr/bin/env bash
# Source this file to select the university TIAGo Pro 12 over Jetson Ethernet.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source tiago_tools/ros_tiago.sh' >&2
    exit 1
fi

# The image selects the PAL interface overlay; load it before applying DDS
# settings so both new shells and interactive profile changes use its types.
if [[ -n "${GATO_ROS_SETUP:-}" ]]; then
    if [[ ! -r "${GATO_ROS_SETUP}" ]]; then
        echo "Missing ROS setup: ${GATO_ROS_SETUP}; rebuild the TIAGo image." >&2
        return 1
    fi
    source "${GATO_ROS_SETUP}" || return 1
fi

_gato_ros_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
if [[ ! -r "${_gato_ros_config_dir}/cyclone_tiago_ethernet.xml" ]]; then
    echo 'Cannot read the TIAGo Ethernet Cyclone DDS configuration.' >&2
    unset _gato_ros_config_dir
    return 1
fi

export GATO_ROS_PROFILE=tiago
export ROS_DOMAIN_ID=2
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${_gato_ros_config_dir}/cyclone_tiago_ethernet.xml"
# Remove even an inherited Docker image value, as required by PAL's guidance.
unset ROS_LOCALHOST_ONLY
unset _gato_ros_config_dir
printf 'ROS profile: tiago (domain 2, enP4p1s0 -> 10.68.0.1)\n'
