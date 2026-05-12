# Git Style

This repository's commit messages are pragmatic and terse. They prioritize
scope, reason, and evidence over conventional formatting ceremony.

## Subject Lines

- Use a short imperative or summary-style subject.
- Prefix with a bracketed scope when the commit belongs to a known workflow.
- Use `[gato]` for upstreamable GATO changes.
- Use `[git]` for local repository hygiene and note-management changes.
- Keep the subject focused on what changed, not on the debugging story.
- A trailing period is uncommon, though one exists in older history.

Examples:

```text
[gato] Keep CUDA error checks in release
[gato] Match GRiD forward dynamics shared memory
[git] gitignore local tools
[git] gitignore markdown files except README.md
add time-varying disturbance example
remove TODO.md
cleanup
```

## Bodies

- Omit the body for small obvious commits.
- Add a body when the change fixes a confusing failure, changes setup
  behavior, or needs validation context.
- Start the body with the practical reason for the change.
- Use direct language; first person is acceptable for local setup notes.
- For multi-part setup changes, a flat bullet list is consistent with history.
- For bug fixes, include the evidence that made the fix convincing, such as
  failing behavior, generated-code references, or benchmark numbers.
- Keep the body technical and concrete rather than narrative.

Examples:

```text
Used for local notes only
```

```text
I fixed the setup problems I encountered on my machine.
- Moved the Docker Python environment into the image at /opt/gato-venv and stopped relying on a host-mounted .venv.
- Switched Python dependency installation in the image from ad hoc pip3 install ... to uv sync --frozen --no-install-project.
- Added .dockerignore.
```

```text
Release builds previously compiled gpuErrchk out under NDEBUG, which hid
CUDA failures until a later crash or copy surfaced them.

Benchmark: built two isolated Release sm_61 bsqpN64_indy7 variants...
Checks enabled averaged 0.554 ms GPU solve time; checks disabled averaged
0.568 ms.
```

```text
The GATO plain forwardDynamics wrappers used hard-coded XI offsets from
the larger gradient layout: 864 for Indy7 and 1008 for IIWA14.

72 * grid::NUM_JOINTS for the XI region matches the generated plain GRiD
layout:
- Indy7 load_update_XImats_helpers copies 432 entries in
gato/dynamics/indy7/indy7_grid.cuh:1599
- IIWA14 forward_dynamics_device splits s_XITemp at 504 in
gato/dynamics/iiwa14/iiwa14_grid.cuh:5389.
```

## Practical Template

For a small commit:

```text
[scope] Verb concise object
```

For a non-obvious bug fix:

```text
[scope] Verb concise object

State the bug or mismatch in one short paragraph.

State why the new behavior is correct. Include file/line references,
generated-code evidence, or benchmark results when they matter.
```
