# Runtime Environment

Codex usually runs inside the Felix development container defined by
`felixtools/Dockerfile` and `felixtools/docker.sh`.

The Docker launcher creates `gato-felix-container` from the `gato-felix` image,
mounts this repository at `/workspace`, mounts the local Codex and agent homes,
uses host networking, passes through GPUs, and disables the default Docker
seccomp/AppArmor restrictions.

Treat commands as having broad access to the container and mounted host paths.
Be careful with destructive commands because they can affect the real working
tree and local notes mirror.

Useful runtime checks:

```bash
test -f /.dockerenv
sed -n '1,80p' /proc/1/cgroup
pwd
id
findmnt -T /workspace
printf '%s\n' "${NOTES_MIRROR_REPO:-}"
```

Expected signals for the Felix container include:

- `/workspace` as the working tree
- user `felix`
- `NOTES_MIRROR_REPO=/GATO-notes`
- a mounted host path for `/workspace`
- `/.dockerenv` on typical Docker setups

These checks are signals, not a contract. If the environment does not match,
note the mismatch before relying on container-specific assumptions.
