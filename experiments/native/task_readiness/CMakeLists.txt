# Include from an experiment's native build: only the original CUDA access probe.
if(NOT TARGET cuda_access_smoke)
  add_executable(cuda_access_smoke "${GATO_SOURCE_DIR}/tests/cpp/cuda_access_smoke.cu")
  target_include_directories(cuda_access_smoke PRIVATE "${GATO_SOURCE_DIR}/gato")
  set_target_properties(cuda_access_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
