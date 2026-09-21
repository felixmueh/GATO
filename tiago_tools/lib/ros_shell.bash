# Bash rcfile used by the launcher. Keep normal user/venv initialization, then
# apply the requested connection profile so stale .bashrc exports cannot win.
_gato_initialize_ros_shell() {
    local requested_profile="${GATO_ROS_PROFILE:-simulation}"
    local config_dir
    config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    case "${requested_profile}" in
        simulation|tiago) ;;
        *) printf 'Unknown ROS profile: %s\n' "${requested_profile}" >&2; return 1 ;;
    esac
    if [[ -r "${HOME}/.bashrc" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.bashrc"
    fi
    # Both permitted profile names are checked above; the path is runtime-local.
    # shellcheck disable=SC1090
    source "${config_dir}/ros_${requested_profile}.sh"
}

if ! _gato_initialize_ros_shell; then
    exit 1
fi
unset -f _gato_initialize_ros_shell
