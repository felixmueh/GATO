#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

source "$(dirname "$0")/common.sh"

# Define the build directory
BUILD_DIR="build"

# Clean the build directory
echo "Cleaning build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"

# Recreate the build directory and navigate into it
echo "Creating build directory: $BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake to configure the project
echo "Configuring project with CMake..."
cmake ..

# Build the project
echo "Building project..."
cmake --build . --parallel

cd ..

if [[ "${GATO_SKIP_CUDA_COMPATIBILITY_TEST:-0}" == "1" ]]; then
    printf "${YELLOW}${BOLD}${GEAR} Skipping CUDA compatibility validation because GATO_SKIP_CUDA_COMPATIBILITY_TEST=1.${RESET}\n"
else
    printf "${YELLOW}${BOLD}${GEAR} Validating CUDA compatibility for built artifacts...${RESET}\n"
    ./tools/test_cuda_compatibility.sh --strict
fi

echo "Build complete!"
