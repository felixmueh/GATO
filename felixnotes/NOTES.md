# Session Log

## 2026-04-19

### Docker-first installation rewrite

- Reworked the repository setup flow so Docker is the primary environment.
- `tools/docker.sh` is now the authoritative container lifecycle script.
- Added `--rebuild-image` and `--no-attach` to `tools/docker.sh`.
- `tools/install.sh` no longer installs or activates a host-side Python
  environment.
- The Docker image now installs `uv` and creates the Python environment inside
  the image at `/opt/gato-venv`.
- Added `.dockerignore` so local artifacts like `.git`, `build/`, and `.venv`
  do not get copied into the Docker build context.
- Updated `../README.md` to document:
  - `./tools/install.sh`
  - `./tools/docker.sh`
  - Python example execution with `PYTHONPATH=python`
