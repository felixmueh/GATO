#!/usr/bin/env bash
# Source in the experiment shell so subsequent commands inherit the session path.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source tiago_tools/start_session.sh' >&2
    exit 1
fi

_gato_start_session() {
    local repo_root session_root session_dir timestamp commit
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    commit="$(git -C "$repo_root" rev-parse --verify HEAD)" || return 1
    session_root="$repo_root/example_artifacts/real_tiago"
    timestamp="$(date +%Y%m%d_%H%M%S)" || return 1
    mkdir -p -- "$session_root" || return 1
    # Atomic creation also keeps repeated calls within the same second distinct.
    session_dir="$(mktemp -d "$session_root/${timestamp}_XXXXXX")" || return 1

    # Save tracked edits as well as their names. Untracked files are listed in
    # worktree.txt, but their contents are not included in the patch.
    if ! {
        printf '%s\n' "$commit" > "$session_dir/commit.txt" &&
        git -C "$repo_root" rev-parse --abbrev-ref HEAD > "$session_dir/branch.txt" &&
        git -C "$repo_root" status --short --untracked-files=all > "$session_dir/worktree.txt" &&
        git -C "$repo_root" diff --no-ext-diff --no-textconv --binary --submodule=diff HEAD \
            > "$session_dir/changes.patch"
    }; then
        printf 'Could not record Git state; incomplete session: %s\n' "$session_dir" >&2
        return 1
    fi

    # Keep the previous session selected if creating this one fails.
    export TIAGO_SESSION="$session_dir"
    printf 'TIAGO_SESSION=%s\n' "$TIAGO_SESSION"
}

if _gato_start_session; then
    unset -f _gato_start_session
else
    unset -f _gato_start_session
    return 1
fi
