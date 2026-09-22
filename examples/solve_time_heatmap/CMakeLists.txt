cmake_minimum_required(VERSION 3.18)
project(gato_solve_time_heatmap LANGUAGES CXX CUDA)

# Standalone build: keep these modules separate from existing local experiments.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# Prefer headers compatible with NumPy in the selected interpreter.
execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE PYBIND11_RESULT)
if(PYBIND11_RESULT EQUAL 0)
  list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)
get_filename_component(GATO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(KNOTS "8;16;32;64;128;256" CACHE STRING "Horizons to build")
set(PLANT "tiago_right" CACHE STRING "Plant to benchmark: tiago_right or indy7")
set_property(CACHE PLANT PROPERTY STRINGS tiago_right indy7)
if(PLANT STREQUAL "tiago_right")
  set(PLANT_DEFINITION PLANT_TIAGO_RIGHT=1)
elseif(PLANT STREQUAL "indy7")
  set(PLANT_DEFINITION PLANT_INDY7=1)
else()
  message(FATAL_ERROR "Unsupported PLANT '${PLANT}'; choose tiago_right or indy7")
endif()
# Resource executables have horizon-only names. Prevent old probes/modules from
# another plant remaining in a reused build directory after reconfiguration.
if(DEFINED HEATMAP_BUILD_PLANT AND NOT HEATMAP_BUILD_PLANT STREQUAL PLANT)
  message(FATAL_ERROR "Use a separate build directory for each PLANT (this one is ${HEATMAP_BUILD_PLANT})")
endif()
set(HEATMAP_BUILD_PLANT "${PLANT}" CACHE INTERNAL "Plant belonging to this build directory")

foreach(n IN LISTS KNOTS)
  set(module bsqpN${n}_${PLANT})
  pybind11_add_module(${module} SHARED "${GATO_ROOT}/python/bindings.cu")
  target_include_directories(${module} PRIVATE "${GATO_ROOT}/gato")
  target_compile_definitions(${module} PRIVATE KNOT_POINTS=${n} ${PLANT_DEFINITION})
  # Keep CUDA error checks enabled even in an optimized benchmark build.
  target_compile_options(${module} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-UNDEBUG>)
  set_target_properties(${module} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/modules")

  add_executable(resources_N${n} resources.cu)
  target_include_directories(resources_N${n} PRIVATE "${GATO_ROOT}/gato")
  target_compile_definitions(resources_N${n} PRIVATE KNOT_POINTS=${n} ${PLANT_DEFINITION})
  target_compile_options(resources_N${n} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-UNDEBUG>)
endforeach()
