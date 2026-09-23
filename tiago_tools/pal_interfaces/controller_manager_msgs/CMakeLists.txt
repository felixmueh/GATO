cmake_minimum_required(VERSION 3.16)
project(controller_manager_msgs)

# Generate native type support from the unchanged PAL definitions. No robot
# controller or hardware implementation is built into this client overlay.
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(std_msgs REQUIRED)
file(GLOB interfaces RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "msg/*.msg" "srv/*.srv")
rosidl_generate_interfaces(${PROJECT_NAME} ${interfaces}
  DEPENDENCIES builtin_interfaces lifecycle_msgs std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
