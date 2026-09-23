ARG BASE_IMAGE=gato
FROM ${BASE_IMAGE}

ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
ARG LOCAL_USER=felix

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ARG ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

# This GATO checkout is used against the PAL TIAGo simulator. PAL's Cyclone DDS
# setup is loopback-only and multicast-disabled, so bake matching ROS discovery
# defaults into the image. These can still be overridden at `docker run` time
# when attaching the image to a different ROS graph.
ENV ROS_DOMAIN_ID=1
ENV ROS_LOCALHOST_ONLY=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/etc/cyclonedds/pal_loopback.xml

# ROS 2 client/runtime dependencies plus native interface generation tools.
RUN apt-get update \
        && apt-get install -y ca-certificates curl gnupg locales \
        && locale-gen en_US en_US.UTF-8 \
        && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
        && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            | gpg --dearmor --yes -o /usr/share/keyrings/ros-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
            > /etc/apt/sources.list.d/ros2.list \
        && apt-get update \
        && apt-get install -y \
            "ros-${ROS_DISTRO}-ros-base" \
            "ros-${ROS_DISTRO}-rclpy" \
            "ros-${ROS_DISTRO}-control-msgs" \
            "ros-${ROS_DISTRO}-trajectory-msgs" \
            "ros-${ROS_DISTRO}-sensor-msgs" \
            "ros-${ROS_DISTRO}-geometry-msgs" \
            "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" \
            "ros-${ROS_DISTRO}-ros2controlcli" \
            "ros-${ROS_DISTRO}-rosidl-default-generators" \
            "ros-${ROS_DISTRO}-lifecycle-msgs" \
            python3-colcon-common-extensions build-essential cmake \
        && rm -rf /var/lib/apt/lists/*

# The robot and PAL simulator use different wire layouts from standard Humble.
# Build only their message package, natively on either amd64 or arm64.
# Humble's generators use PYTHON_EXECUTABLE as well as Python3_EXECUTABLE;
# keep both on system Python, which has the apt-installed generator modules.
COPY tiago_tools/pal_interfaces /tmp/gato-pal-interfaces
RUN /bin/bash -c 'set -e; cd /tmp/gato-pal-interfaces; \
        sha256sum -c SHA256SUMS; \
        source /opt/ros/${ROS_DISTRO}/setup.bash; \
        colcon build --merge-install --install-base /opt/gato-pal-interfaces \
            --allow-overriding controller_manager_msgs \
            --cmake-args -DBUILD_TESTING=OFF \
                -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3' \
        && rm -rf /tmp/gato-pal-interfaces
ENV GATO_ROS_SETUP=/opt/gato-pal-interfaces/setup.bash

COPY tiago_tools/cyclone_pal_loopback.xml /etc/cyclonedds/pal_loopback.xml

# Create the development user after installing system packages.
RUN groupadd --gid "${LOCAL_GID}" "${LOCAL_USER}" \
        && useradd --uid "${LOCAL_UID}" --gid "${LOCAL_GID}" --create-home --shell /bin/bash "${LOCAL_USER}" \
        && echo '[ -f /opt/gato-venv/bin/activate ] && source /opt/gato-venv/bin/activate' >> "/home/${LOCAL_USER}/.bashrc" \
        && echo '# ROS is loaded by tiago_tools/ros_simulation.sh or ros_tiago.sh.' >> "/home/${LOCAL_USER}/.bashrc" \
        && chown -R "${LOCAL_UID}:${LOCAL_GID}" "/home/${LOCAL_USER}"

ENV HOME=/home/${LOCAL_USER}
ENV USER=${LOCAL_USER}
ENV PYTHONPATH=/workspace/tiago_src:/workspace/python
USER ${LOCAL_USER}

CMD ["/bin/bash"]
