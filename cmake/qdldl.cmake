# Optional hybrid backend dependency. Shared with the CPU-only bridge test.
include(FetchContent)
enable_language(C)
set(QDLDL_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(QDLDL_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
set(QDLDL_UNITTESTS OFF CACHE BOOL "" FORCE)
set(QDLDL_FLOAT ON CACHE BOOL "" FORCE)
set(QDLDL_LONG OFF CACHE BOOL "" FORCE)
FetchContent_Declare(qdldl
  GIT_REPOSITORY https://github.com/osqp/qdldl.git
  GIT_TAG df481005111f2864770a3eccb05ed4120727eff6)
FetchContent_MakeAvailable(qdldl)
