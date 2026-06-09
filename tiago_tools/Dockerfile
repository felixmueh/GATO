FROM gato

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

# ROS 2 client/runtime dependencies for lightweight command adapters. Keep ROS
# sourcing opt-in so the ROS Python path does not contaminate the GATO venv.
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
        && rm -rf /var/lib/apt/lists/*

COPY tiago_tools/cyclone_pal_loopback.xml /etc/cyclonedds/pal_loopback.xml

RUN echo 'export PYTHONPATH=/workspace/tiago_src:/workspace/python${PYTHONPATH:+:$PYTHONPATH}' >> ~/.bashrc
RUN echo '# Source ROS manually only in shells that run ROS tools: source /opt/ros/humble/setup.bash' >> ~/.bashrc

CMD ["/bin/bash"]
