cmake_minimum_required(VERSION 3.24)
project(frozen_off_torque_n32 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT GATO_SOURCE_DIR OR NOT EXISTS "${GATO_SOURCE_DIR}/python/bindings.cu")
  message(FATAL_ERROR "Set GATO_SOURCE_DIR to the exact numerical85ef checkout")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(bsqpN32_tiago_right SHARED "${GATO_SOURCE_DIR}/python/bindings.cu")
target_include_directories(bsqpN32_tiago_right PRIVATE "${GATO_SOURCE_DIR}/gato")
target_compile_definitions(bsqpN32_tiago_right PRIVATE KNOT_POINTS=32 PLANT_TIAGO_RIGHT=1
  TIAGO_COST_JOINT_MARGIN_RAD=0.20 TIAGO_COST_VELOCITY_LIMIT_FRACTION=0.80
  TIAGO_COST_CONTROL_LIMIT_FRACTION=0.85)
target_compile_options(bsqpN32_tiago_right PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)
set_target_properties(bsqpN32_tiago_right PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/modules")
