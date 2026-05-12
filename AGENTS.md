# AGENTS Notes

## Runtime environment

- Codex usually runs inside the Felix development container defined by
  `felixtools/Dockerfile` and `felixtools/docker.sh`.
- Treat commands as having broad access to the container and mounted host paths,
  including the real `/workspace` working tree.
- If package, mount, device, or security settings need to change, ask the user
  to update the Docker files and rebuild or recreate the container.
- See `felixnotes/RUNTIME_ENVIRONMENT.md` for container-detection checks and
  details.

## Session log

- Maintain a local chronological work log in `felixnotes/SESSION.md`.
  - Note any design decisions, implementation problems and general implementation steps.
  - Do not note any small step. Especally during debugging.
- Use it to capture approach problems, decisions, debugging findings.
  The goal is to give an overview over all work done in this repository.

## Documentation tracking

- The upstream repository does not use agentic tooling conventions.
- Keep `README.md` aligned with the upstream repository. Modify it only for
  changes that should be committed as `[gato]` upstream-facing work.
- Because of that policy, `README.md` may not contain the most up-to-date
  local workflow information for this repository; prefer `AGENTS.md`,
  `felixnotes/`, and tool help output for local workflow notes.
- Do not commit documentation or metadata that exists only to support agentic
  coding workflows.
- To still preserve this information locally, use a second Git repository.
- A Git hook is in place to mirror and commit `AGENTS.md`, `felixnotes/`, and
  `felixtools/` to that second repository at every commit in the main working
  tree.
- Because of that split workflow, commit messages in the main repository should
  not mention agentic coding.

## Commit labeling

- Prefix commits with a category label.
- Use `[tiago]` for Tiago-specific integration, experiments, wrappers, URDF
  preparation, tuning, and related debugging.
- Use `[gato]` for changes that are of interest to the original upstream GATO
  repository.
- Prefer using `[gato]` whenever a change is plausibly upstreamable on its own,
  even if it was discovered during Tiago work.
- Follow `felixnotes/GIT_STYLE.md` for local commit-message tone, structure,
  and examples.

## Upstream-oriented structure

- Keep changes to original GATO sources confined to modifications that are
  reasonable candidates for contributing back upstream.
- If Tiago-specific or experimental logic can live in isolated files, wrappers,
  examples, scripts, or separate configuration paths, prefer that structure.
- It is acceptable to break this rule when strictly separating the changes would
  add too much complexity, but treat that as an explicit tradeoff rather than
  the default.
- When touching core solver/runtime code, try to keep the diff narrow and
  general-purpose so the upstreamable part is easy to identify later.

## Generated code

- Treat generated artifacts as immutable checked-in outputs.
- Do not hand-edit generated `grid.cuh` files.
- Adapt behavior in handwritten wrappers, build wiring, preprocessing, or by
  regenerating the artifact from its source inputs.

## Host workflow

- If a result depends on a non-default local environment, note that clearly so
  it is not mistaken for a general repository requirement.
- Multiple agents may be running on this machine in different GATO-family
  folders and sharing the same GPU. This is usually acceptable because GATO
  experiments often need only one SM, but account for shared GPU usage during
  performance-critical benchmarking or timing-sensitive debugging.
- CUDA builds, especially Python extension targets and multi-architecture
  builds, can be slow. Treat long compile times as expected unless there is
  evidence of a real hang.
- Benchmark code changes with matched build flags, a warm-up run, and an A/B
  comparison on the same short reproducible workload.

## Recommended workflow

- Keep and review notes in applicable files under `felixnotes/`.
  - `felixnotes/BUGFIXES.md` for bugs in the original GATO repository.
  - `felixnotes/IMPL_TIAGO.md` for Tiago-specific implementation details.
  - `felixnotes/TESTING.md` for the local build-backed testing setup.
