#!/usr/bin/env bash
# Shared by the Tiago launcher and optional development overlays.
# The caller sets REPO_ROOT and may override these defaults before docker_main.
source "${REPO_ROOT}/tools/common.sh"

IMAGE_NAME=gato-tiago
CONTAINER_NAME=gato-tiago-container
CONTAINER_REPO_ROOT=/workspace
WORKSPACE_HOST="${REPO_ROOT}"
LOCAL_UID="$(id -u)"
LOCAL_GID="$(id -g)"
LOCAL_USER="$(id -un)"
EXTRA_RUN_ARGS=()
EXTRA_BUILD_ARGS=()
OVERLAY_DOCKERFILE=""

prepare_container() {
    :
}

build_images() {
    if [[ "${REBUILD_IMAGE}" -eq 1 ]] || ! docker image inspect "${BASE_IMAGE_NAME}" >/dev/null 2>&1; then
        docker build --build-arg "BASE_IMAGE=${CUDA_IMAGE}" -t "${BASE_IMAGE_NAME}" "${REPO_ROOT}"
    fi
    if [[ "${REBUILD_IMAGE}" -eq 1 ]] || ! docker image inspect "${TIAGO_IMAGE_NAME}" >/dev/null 2>&1; then
        docker build \
            --build-arg "BASE_IMAGE=${BASE_IMAGE_NAME}" \
            --build-arg "LOCAL_UID=${LOCAL_UID}" \
            --build-arg "LOCAL_GID=${LOCAL_GID}" \
            --build-arg "LOCAL_USER=${LOCAL_USER}" \
            -t "${TIAGO_IMAGE_NAME}" -f "${REPO_ROOT}/tiago_tools/Dockerfile" "${REPO_ROOT}"
    fi
    if [[ -n "${OVERLAY_DOCKERFILE}" ]]; then
        docker build --build-arg "BASE_IMAGE=${TIAGO_IMAGE_NAME}" \
            --build-arg "LOCAL_USER=${LOCAL_USER}" "${EXTRA_BUILD_ARGS[@]}" \
            -t "${IMAGE_NAME}" -f "${OVERLAY_DOCKERFILE}" "${REPO_ROOT}"
    fi
}

docker_main() {
    local target=desktop ros_profile=simulation
    REBUILD_IMAGE=0
    local recreate_container=0 restart_container=0 attach_shell=1
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --target)
                if [[ $# -lt 2 ]]; then echo '--target requires desktop or jetson' >&2; return 1; fi
                target="$2"; shift 2 ;;
            --ros-profile)
                if [[ $# -lt 2 ]]; then echo '--ros-profile requires simulation or tiago' >&2; return 1; fi
                ros_profile="$2"; shift 2 ;;
            --rebuild-image) REBUILD_IMAGE=1; shift ;;
            --recreate-container) recreate_container=1; shift ;;
            --restart) restart_container=1; shift ;;
            --no-attach) attach_shell=0; shift ;;
            -h|--help)
                cat <<HELP
Usage: $0 [--target desktop|jetson] [--ros-profile simulation|tiago] [--rebuild-image] [--recreate-container] [--restart] [--no-attach]

  --target              Hardware target (default: desktop; Jetson uses L4T r36.4.0).
  --ros-profile         ROS connection for new shells (default: simulation).
                        tiago uses domain 2, enP4p1s0 and peer 10.68.0.1.
  --rebuild-image       Rebuild the image chain, then recreate this container.
  --recreate-container  Recreate this container without rebuilding existing images.
  --restart             Stop and start this container.
  --no-attach           Leave the container running without opening a shell.

Run on the target machine. Images are built with your host UID/GID;
rebuild and recreate existing containers to adopt a different user.
ROS profiles load the image's PAL interface overlay and apply ROS/DDS settings.
Changing a profile does not reconfigure existing shells or running nodes.
HELP
                return 0 ;;
            *) printf 'Unknown option: %s\n' "$1" >&2; return 1 ;;
        esac
    done

    case "${ros_profile}" in
        simulation|tiago) ;;
        *) printf 'Unknown ROS profile: %s\n' "${ros_profile}" >&2; return 1 ;;
    esac

    BASE_IMAGE_NAME=gato
    TIAGO_IMAGE_NAME=gato-tiago
    local gpu_args=(--gpus all)
    case "${target}" in
        desktop) CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04 ;;
        jetson)
            CUDA_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
            BASE_IMAGE_NAME=gato:jetson-r36.4.0
            TIAGO_IMAGE_NAME=gato-tiago:jetson-r36.4.0
            IMAGE_NAME="${IMAGE_NAME}:jetson-r36.4.0"
            CONTAINER_NAME="${CONTAINER_NAME}-jetson-r36.4.0"
            gpu_args=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all)
            ;;
        *) printf 'Unknown target: %s\n' "${target}" >&2; return 1 ;;
    esac

    if [[ "${REBUILD_IMAGE}" -eq 1 ]] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        build_images
    fi

    export DISPLAY="${DISPLAY:-:0}"
    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null 2>&1 || true
    fi
    local shell_args=(
        --user "${LOCAL_UID}:${LOCAL_GID}"
        -e "HOME=/home/${LOCAL_USER}" -e "USER=${LOCAL_USER}"
        -e "GATO_ROS_PROFILE=${ros_profile}"
        -e "PYTHONPATH=${CONTAINER_REPO_ROOT}/tiago_src:${CONTAINER_REPO_ROOT}/python"
        -w "${CONTAINER_REPO_ROOT}"
    )
    # The same rcfile applies profiles to the initial shell and later exec
    # shells, including containers created before profile support was added.
    local shell_command=(/bin/bash --rcfile "${CONTAINER_REPO_ROOT}/tiago_tools/lib/ros_shell.bash")
    local container_exists
    container_exists="$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")"
    if [[ -n "${container_exists}" && ( "${REBUILD_IMAGE}" -eq 1 || "${recreate_container}" -eq 1 ) ]]; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null
        container_exists=""
    fi
    if [[ -z "${container_exists}" ]]; then
        prepare_container
        docker create -it "${gpu_args[@]}" --network=host --ipc=host \
            -e "DISPLAY=${DISPLAY}" "${shell_args[@]}" \
            -v "${WORKSPACE_HOST}:/workspace:Z" \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            "${EXTRA_RUN_ARGS[@]}" --name "${CONTAINER_NAME}" "${IMAGE_NAME}" "${shell_command[@]}" >/dev/null
    elif [[ "${restart_container}" -eq 1 ]]; then
        docker stop "${CONTAINER_NAME}" >/dev/null
    fi
    docker start "${CONTAINER_NAME}" >/dev/null
    if [[ "${attach_shell}" -eq 1 ]]; then
        docker exec -it "${shell_args[@]}" "${CONTAINER_NAME}" "${shell_command[@]}"
    else
        printf "Container '%s' is ready.\n" "${CONTAINER_NAME}"
        printf 'Existing shells keep their settings; open a new shell with --ros-profile %s to select that profile.\n' "${ros_profile}"
    fi
}
