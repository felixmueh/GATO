# GATO
> GPU-Accelerated Trajectory Optimization

Numerical experiments and the open-source solver from  ["GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control"](https://arxiv.org/abs/2510.07625)

## Installation

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
```

### Requirements
- Docker
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Docker is used for containerization and is strongly advised.

### Setup

```sh
./tools/install.sh
```

This will:

- initialize git submodules
- rebuild the Docker image
- recreate the Docker container from that image
- build the project inside the container

To enter the container later:

```sh
./tools/docker.sh
```

To force a fresh image rebuild:

```sh
./tools/docker.sh --rebuild-image
```

### Build Options

You can control which Python extension modules are built by selecting plant models and horizon lengths at CMake configure time:

```sh
mkdir -p build && cd build
cmake -DPLANT="indy7;iiwa14" -DKNOTS="8;32;128" ..
cmake --build . --parallel
```

- `PLANT`: semicolon-separated list of plant targets (`indy7`, `iiwa14`).
- `KNOTS`: semicolon-separated list of horizon lengths.

Built Python modules are written to `python/bsqp/` as `bsqpN{N}_{plant}.so`.
After `./tools/build.sh`, the repo also runs `./tools/test_cuda_compatibility.sh`
to validate that the built artifacts are compatible with the currently visible
GPU. Set `GATO_SKIP_CUDA_COMPATIBILITY_TEST=1` to skip that validation when
needed.

### Reference Environment

- Ubuntu 22.04
- CUDA 12.9
- CMake 3.24+
- C++17
- gcc 11.4.0
- Python 3.10.12
- Docker 28.1.0

## Usage

See [batch_sqp.cu](examples/bsqp.cu) for a minimal example of a batched trajectory optimization solve in C++/CUDA. Example Jupyter notebooks and python benchmarks using GATO for MPC are in [examples/](examples/).

The container shell automatically picks up the image-backed Python environment
and exports `PYTHONPATH=/workspace/python`.

Run the Python benchmark example inside the container with:

```sh
python examples/benchmark_fig8.py
```

Run the C++ example with:

``` sh
./build/bsqp
```

### TIAGo ROS 2 connection profiles

From the repository root on the host:

```bash
./tiago_tools/docker.sh --target desktop --ros-profile simulation
./tiago_tools/docker.sh --target jetson --ros-profile tiago
```

| Profile | Domain | DDS interfaces | Peers |
| --- | --- | --- | --- |
| `simulation` (default) | 1 | `lo` | `localhost` |
| `tiago` | 2 | `enP4p1s0`, `lo` | `10.68.0.1`, `localhost` |

Both use Cyclone DDS with multicast disabled. The robot profile's interface and
address are hard-coded for the university Jetson/TIAGo setup in
[cyclone_tiago_ethernet.xml](tiago_tools/cyclone_tiago_ethernet.xml).

Inside a running container, switch the current Bash terminal without restarting:

```bash
source tiago_tools/ros_tiago.sh
# Or:
source tiago_tools/ros_simulation.sh
```

Source ROS separately when using ROS tools:

```bash
source /opt/ros/humble/setup.bash
```

Profiles override inherited ROS/DDS settings; `tiago` unsets
`ROS_LOCALHOST_ONLY`. They affect new launcher shells or the terminal where
sourced, leaving existing nodes and other terminals unchanged. No image rebuild
is needed. Plain `docker exec ... bash` requires sourcing a profile manually.

### TIAGo experiment sessions

Inside the container, from the repository root:

```bash
source tiago_tools/start_session.sh
set -o pipefail
python tiago_examples/validate_ros_assumptions.py 2>&1 | tee "$TIAGO_SESSION/preflight.log"
```

Each source call creates a fresh directory under `example_artifacts/real_tiago/`
and exports its absolute path as `TIAGO_SESSION` in the current shell. It saves
the commit, branch, worktree status and a patch of tracked changes. Untracked
files are listed but not copied. Use `$TIAGO_SESSION` for experiment output.

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients

## Cite

```bibtex
@misc{du2025gatogpuacceleratedbatchedtrajectory,
      title={GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control},
      author={Alexander Du and Emre Adabag and Gabriel Bravo and Brian Plancher},
      year={2025},
      eprint={2510.07625},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07625},
}
```
