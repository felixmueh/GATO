# Local Git Hooks

This repository uses a repo-local hooks directory:

```bash
.githooks/
```

The intended setup is:

```bash
git config core.hooksPath .githooks
```

Current behavior:

- `post-commit` calls `../felixtools/commit_notes.sh`.
- `../felixtools/commit_notes.sh` mirrors `../AGENTS.md`, `felixnotes/`, and
  `../felixtools/` into a
  separate local Git repository
- default mirror repo path: `../GATO-notes`
- default mirror destination inside that repo: `GATO/`
- the script preserves relative paths, including files in subdirectories
- the script syncs deletions and moves
- the hook commits there automatically only when the mirrored Markdown files
  changed
- it does not push

Optional environment overrides:

```bash
NOTES_MIRROR_REPO=/path/to/notes-repo
NOTES_MIRROR_SUBDIR=SomeProject
```
