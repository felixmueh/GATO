#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source common definitions
source "${REPO_ROOT}/tools/common.sh"

BASE_IMAGE_NAME="gato"
IMAGE_NAME="gato-tiago"
CONTAINER_NAME="gato-tiago-container"
DEFAULT_ROS_DOMAIN_ID=1
DEFAULT_ROS_LOCALHOST_ONLY=0
DEFAULT_RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
DEFAULT_CYCLONEDDS_URI="/workspace/tiago_tools/cyclone_pal_loopback.xml"
REBUILD_IMAGE=0
ATTACH_SHELL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild-image)
            REBUILD_IMAGE=1
            shift
            ;;
        --no-attach)
            ATTACH_SHELL=0
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./tiago_tools/docker.sh [--rebuild-image] [--no-attach]

  --rebuild-image  Rebuild the Docker image and recreate the container.
  --no-attach      Ensure the container is running, but do not open a shell.
EOF
            exit 0
            ;;
        *)
            printf "\n${RED}${BOLD}${CROSS} Unknown option: %s${RESET}\n" "$1"
            exit 1
            ;;
    esac
done

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Setting up docker container.${RESET}\n"

if [[ "${REBUILD_IMAGE}" -eq 1 ]]; then
    printf "${YELLOW}${BOLD}${GEAR} Rebuilding image '${IMAGE_NAME}'.${RESET}\n\n"
    if docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
        docker stop "${CONTAINER_NAME}" >/dev/null
    fi
    if docker ps -aq -f name=^/${CONTAINER_NAME}$ | grep -q .; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi
    docker build -t "${BASE_IMAGE_NAME}" "${REPO_ROOT}"
    docker build -t "${IMAGE_NAME}" -f "${REPO_ROOT}/tiago_tools/Dockerfile" "${REPO_ROOT}"
    printf "\n${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' rebuilt successfully.${RESET}\n"
elif ! docker image inspect ${IMAGE_NAME} >/dev/null 2>&1; then
    printf "${YELLOW}${BOLD}${GEAR} Image '${IMAGE_NAME}' not found. Building...${RESET}\n\n"
    if ! docker image inspect "${BASE_IMAGE_NAME}" >/dev/null 2>&1; then
        docker build -t "${BASE_IMAGE_NAME}" "${REPO_ROOT}"
    fi
    docker build -t "${IMAGE_NAME}" -f "${REPO_ROOT}/tiago_tools/Dockerfile" "${REPO_ROOT}"
    printf "\n${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' built successfully.${RESET}\n"
else
    printf "${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' found.${RESET}\n"
fi

# Prepare for GUI forwarding if possible, but do not fail if X11 is unavailable.
export DISPLAY=${DISPLAY:-:0} # Default to :0 if not set
if command -v xhost >/dev/null 2>&1; then
    if ! xhost +local:docker >/dev/null 2>&1; then
        printf "${YELLOW}${BOLD}${GEAR} Skipping optional xhost setup; X11 forwarding is unavailable.${RESET}\n"
    fi
fi

EXIT_CODE=0

# Check if container exists and is running
if docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    if [[ "${ATTACH_SHELL}" -eq 1 ]]; then
        printf "${BOLD}${GREEN}${ARROW} Attaching to running container ${YELLOW}${CONTAINER_NAME}${RESET}...\n\n"
        docker exec -it \
            -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}" \
            -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-${DEFAULT_ROS_LOCALHOST_ONLY}}" \
            -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-${DEFAULT_RMW_IMPLEMENTATION}}" \
            -e "CYCLONEDDS_URI=${CYCLONEDDS_URI:-${DEFAULT_CYCLONEDDS_URI}}" \
            -e "PYTHONPATH=/workspace/tiago_src:/workspace/python" \
            "${CONTAINER_NAME}" /bin/bash
        EXIT_CODE=$?
    else
        printf "${GREEN}${BOLD}${CHECK} Container '${CONTAINER_NAME}' is already running.${RESET}\n"
    fi
# Check if container exists but is stopped
elif docker ps -aq -f status=exited -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    if [[ "${ATTACH_SHELL}" -eq 1 ]]; then
        printf "${YELLOW}${BOLD}${GEAR} Starting stopped container ${YELLOW}${CONTAINER_NAME}${RESET} and attaching...\n\n"
        docker start "${CONTAINER_NAME}" >/dev/null
        docker exec -it \
            -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}" \
            -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-${DEFAULT_ROS_LOCALHOST_ONLY}}" \
            -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-${DEFAULT_RMW_IMPLEMENTATION}}" \
            -e "CYCLONEDDS_URI=${CYCLONEDDS_URI:-${DEFAULT_CYCLONEDDS_URI}}" \
            -e "PYTHONPATH=/workspace/tiago_src:/workspace/python" \
            "${CONTAINER_NAME}" /bin/bash
        EXIT_CODE=$?
    else
        printf "${YELLOW}${BOLD}${GEAR} Starting stopped container ${YELLOW}${CONTAINER_NAME}${RESET}...\n\n"
        docker start "${CONTAINER_NAME}" >/dev/null
    fi
# Container does not exist, run a new one
else
    if [[ "${ATTACH_SHELL}" -eq 1 ]]; then
        printf "${BOLD}${GREEN}${ARROW} Running new container (name: ${YELLOW}${CONTAINER_NAME}${YELLOW})...${RESET}\n\n"
        docker run -it \
            --gpus all \
            --network=host \
            --ipc=host \
            -e DISPLAY="${DISPLAY}" \
            -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}" \
            -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-${DEFAULT_ROS_LOCALHOST_ONLY}}" \
            -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-${DEFAULT_RMW_IMPLEMENTATION}}" \
            -e "CYCLONEDDS_URI=${CYCLONEDDS_URI:-${DEFAULT_CYCLONEDDS_URI}}" \
            -e "PYTHONPATH=/workspace/tiago_src:/workspace/python" \
            -v "${REPO_ROOT}":/workspace:Z \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name "${CONTAINER_NAME}" \
            "${IMAGE_NAME}"
        EXIT_CODE=$?
    else
        printf "${BOLD}${GREEN}${ARROW} Creating new container (name: ${YELLOW}${CONTAINER_NAME}${YELLOW})...${RESET}\n\n"
        docker run -d -it \
            --gpus all \
            --network=host \
            --ipc=host \
            -e DISPLAY="${DISPLAY}" \
            -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}" \
            -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-${DEFAULT_ROS_LOCALHOST_ONLY}}" \
            -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-${DEFAULT_RMW_IMPLEMENTATION}}" \
            -e "CYCLONEDDS_URI=${CYCLONEDDS_URI:-${DEFAULT_CYCLONEDDS_URI}}" \
            -e "PYTHONPATH=/workspace/tiago_src:/workspace/python" \
            -v "${REPO_ROOT}":/workspace:Z \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name "${CONTAINER_NAME}" \
            "${IMAGE_NAME}" >/dev/null
    fi
fi

if [[ "${ATTACH_SHELL}" -eq 0 ]]; then
    printf "\n${GREEN}${BOLD}${CHECK} Container ${YELLOW}${CONTAINER_NAME}${RESET} is ready.${RESET}\n"
elif [ "$EXIT_CODE" -eq 0 ]; then
    printf "\n${GREEN}${BOLD}${CHECK} Exited container session for ${YELLOW}${CONTAINER_NAME}${RESET} successfully.${RESET}\n"
else
    printf "\n${RED}${BOLD}${CROSS} Command failed or container exited with error (Code: $EXIT_CODE).${RESET}\n"
fi
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"
