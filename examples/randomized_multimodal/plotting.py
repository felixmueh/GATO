"""Offline scientific figures for the randomized multimodal experiment.

Only NumPy and Matplotlib are required.  Figures consume the retained JSON
records, never rerun optimization, and distinguish infeasible candidates from
feasible results.  Near-best means near the campaign's best-known reference.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np


BLUE = "#2078A8"
ORANGE = "#D77C26"
INK = "#233343"
GRAY = "#AAB2BA"
MODE_COLORS = {"upper": BLUE, "lower": ORANGE, "clockwise": ORANGE,
               "counterclockwise": BLUE, "cw": ORANGE, "ccw": BLUE,
               "positive": BLUE, "negative": ORANGE}
STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#A5AFB8",
         "axes.titleweight": "semibold", "figure.facecolor": "white",
         "savefig.facecolor": "white", "pdf.fonttype": 42,
         "svg.fonttype": "none"}


def _save(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _scene_map(payload):
    return {str(s["scene_id"]): s for s in payload.get("scenes", [])}


def _xy(values):
    array = np.asarray(values, dtype=float)
    return array[..., :2]


def _mode_color(lane, path=None, scene=None):
    mode = str(lane.get("mode", ""))
    if mode in MODE_COLORS:
        return MODE_COLORS[mode]
    if path is not None and scene is not None:
        start = _xy(scene["x0"])
        direction = _xy(scene["goal"]) - start
        offsets = _xy(path) - _xy(scene["center"])
        # Stable geometric coloring for integer/custom mode labels.
        closest = np.argmin(np.linalg.norm(offsets, axis=1))
        cross = direction[0] * offsets[closest, 1] - direction[1] * offsets[closest, 0]
        return BLUE if cross >= 0 else ORANGE
    return BLUE


def _scene_axis(ax, scene):
    center = _xy(scene["center"])
    radius = float(scene["radius"])
    buffer = float(scene.get("buffer", 0))
    if buffer > 0:
        ax.add_patch(plt.Circle(center, radius + buffer, facecolor="#E7EBEE",
                               edgecolor="#8997A2", linestyle=":", linewidth=1))
    ax.add_patch(plt.Circle(center, radius, facecolor="#697782", edgecolor="white",
                           linewidth=1, zorder=3))
    start, goal = _xy(scene["x0"]), _xy(scene["goal"])
    ax.scatter(*start, s=30, color=INK, zorder=8)
    ax.scatter(*goal, s=110, marker="*", color=INK, edgecolor="white", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(color="#DCE2E7", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _snapshot_scene(scene, snapshot):
    """Use the obstacle active at this solve, preserving the episode start."""
    active = dict(scene)
    details = snapshot.get("scene", snapshot)
    for key in ("center", "radius", "buffer", "goal"):
        if key in details:
            active[key] = details[key]
    if "obstacle_center" in snapshot:
        active["center"] = snapshot["obstacle_center"]
    return active


def _episode_limits(scene, snapshots, history):
    paths = [history] + [p for s in snapshots for p in s.get("candidate_trajectories", [])]
    for snapshot in snapshots:
        active = _snapshot_scene(scene, snapshot)
        radius = float(active["radius"]) + float(active.get("buffer", 0))
        center = _xy(active["center"])
        paths.append(np.array([center - radius, center + radius]))
    return _limits(scene, paths)


def _limits(scene, paths):
    points = [_xy(scene["x0"])[None], _xy(scene["goal"])[None]]
    radius = float(scene["radius"]) + float(scene.get("buffer", 0))
    points.extend([_xy(scene["center"])[None] + radius,
                   _xy(scene["center"])[None] - radius])
    for path in paths:
        array = _xy(path).reshape(-1, 2)
        points.append(array[np.isfinite(array).all(axis=1)])
    points = np.concatenate(points)
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    margin = max(float(np.max(hi - lo)) * 0.075, 0.05)
    return (lo - margin, hi + margin)


def _apply_limits(ax, limits):
    lo, hi = limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


def _full_route_color(history, path, scene, lane):
    """Keep route labels tied to executed prefix plus proposed future."""
    if history is None or len(history) == 0:
        return _mode_color(lane, path, scene)
    full = np.concatenate((_xy(history), _xy(path)[1:]), axis=0)
    if not np.isfinite(full).all():
        return GRAY
    delta = full - _xy(scene['center'])
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    return BLUE if angles[-1] - angles[0] > 0 else ORANGE


def _draw_candidates(ax, scene, snapshot, history=None):
    trajectories = snapshot.get("candidate_trajectories", [])
    lanes = snapshot.get("lanes", [])
    selected = int(snapshot.get("selected_lane", -1))
    for i, trajectory in enumerate(trajectories):
        lane = lanes[i] if i < len(lanes) else {}
        feasible = bool(lane.get("feasible", False))
        path = _xy(trajectory)
        ax.plot(*path.T, color=_full_route_color(history, path, scene, lane) if feasible else GRAY,
                linewidth=1.25, alpha=0.50 if feasible else 0.4,
                linestyle="-" if feasible else "--", zorder=4)
    if 0 <= selected < len(trajectories):
        lane = lanes[selected] if selected < len(lanes) else {}
        path = _xy(trajectories[selected])
        ax.plot(*path.T, color=_full_route_color(history, path, scene, lane), linewidth=3,
                linestyle="-" if lane.get("feasible", False) else "--", zorder=5)
    return sum(bool(lane.get("feasible", False)) for lane in lanes), len(trajectories)


def plot_mpc_stages(scene, episode, output_dir, name="mpc_stages", max_panels=4):
    """Plot predetermined evenly spaced recorded stages, including failures."""
    snapshots = episode.get("snapshots", [])
    if not snapshots:
        return []
    indices = np.unique(np.linspace(0, len(snapshots) - 1,
                                    min(max_panels, len(snapshots)), dtype=int))
    changes = [i for i in range(1, len(snapshots)) if not np.allclose(
        _snapshot_scene(scene, snapshots[i - 1])["center"],
        _snapshot_scene(scene, snapshots[i])["center"])]
    if changes and max_panels >= 4:
        switch = changes[0]
        indices = np.unique([0, max(0, switch - 1), switch, len(snapshots) - 1])
        if len(indices) < 4:
            indices = np.unique([*indices, (switch + len(snapshots) - 1) // 2])
    history = _xy(episode.get("executed_states", [scene["x0"]]))
    limits = _episode_limits(scene, snapshots, history)
    fig, axes = plt.subplots(1, len(indices), figsize=(4.0 * len(indices), 4.1),
                             squeeze=False, layout="constrained")
    for ax, index in zip(axes[0], indices):
        snapshot = snapshots[index]
        active = _snapshot_scene(scene, snapshot)
        _scene_axis(ax, active)
        tick = int(snapshot.get("tick", index))
        executed_steps = int(snapshot.get("executed_steps", tick * episode.get("execute", 1)))
        traveled = history[:min(executed_steps + 1, len(history))]
        count, total = _draw_candidates(ax, active, snapshot, traveled)
        if len(traveled):
            ax.plot(*traveled.T, color=INK, linewidth=2.8, zorder=6)
        current = _xy(snapshot.get("x0", history[min(executed_steps, len(history) - 1)]))
        ax.scatter(*current, s=40, facecolor="white", edgecolor=INK, linewidth=1.5, zorder=9)
        _apply_limits(ax, limits)
        change_label = " · obstacle moved" if index in changes else ""
        ax.set_title(f"MPC step {tick}{change_label}\n{count}/{total} feasible candidates", fontsize=11)
    handles = [Line2D([], [], color=INK, lw=2.8, label="Executed"),
               Line2D([], [], color=GRAY, lw=3, label="Selected: thick"),
               Line2D([], [], color=BLUE, lw=1.3, label="CCW route"),
               Line2D([], [], color=ORANGE, lw=1.3, label="CW route"),
               Line2D([], [], color=GRAY, ls="--", label="Infeasible")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(f"Randomized batch MPC · scene {scene['scene_id']} · B={episode['batch_size']}",
                 fontsize=14)
    return _save(fig, output_dir, name)


def plot_seed_improvement(scene, run, output_dir, name="seed_to_solution"):
    seeds = run.get("initial_trajectories", [])
    outputs = run.get("candidate_trajectories", [])
    if len(seeds) == 0 or len(outputs) == 0:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    limits = _limits(scene, list(seeds) + list(outputs))
    lanes = run.get("lanes", [])
    for i, ax in enumerate(axes):
        _scene_axis(ax, scene)
        for j, trajectory in enumerate(seeds if i == 0 else outputs):
            lane = lanes[j] if j < len(lanes) else {}
            path = _xy(trajectory)
            feasible = lane.get("initial_feasible" if i == 0 else "feasible", False)
            ax.plot(*path.T, color=_mode_color(lane, outputs[j], scene),
                    lw=1.3, alpha=0.7, ls="-" if feasible else "--")
        _apply_limits(ax, limits)
        ax.set_title("Inherited warm lane + obstacle-blind proposals" if i == 0 and "selected_polish" in run else "Obstacle-blind initial trajectories" if i == 0 else "After trajectory optimization")
    initial = sum(bool(l.get("initial_feasible", False)) for l in lanes)
    final = sum(bool(l.get("feasible", False)) for l in lanes)
    fig.suptitle(f"Same candidates, before and after optimization · feasible {initial} → {final}/{len(outputs)}",
                 fontsize=13)
    fig.supxlabel("Color follows the resulting route; dashed paths fail the feasibility checks.", fontsize=9)
    return _save(fig, output_dir, name)


def _cluster_interval(rows, key, seed=17):
    """Conservative bounded-mean interval; scene repeats remain clustered."""
    from summarize import success_stats
    result = success_stats(rows, key)
    return result['mean'], *result['interval95']


def plot_success(payload, output_dir, key="changed_state_runs", name="changed_state_success"):
    rows = payload.get(key, [])
    if not rows:
        return []
    budgets = sorted({int(row["batch_size"]) for row in rows})
    tolerance = 100*float(payload.get('config', {}).get('near_best_relative', .03))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for ax, metric, title, color in zip(axes, ("feasible", "near_best"),
                                   ("Feasible solution", f"Feasible, ≤{tolerance:g}% above best-known cost"),
                                   (BLUE, ORANGE)):
        stats = np.asarray([_cluster_interval([r for r in rows if int(r["batch_size"]) == b], metric)
                            for b in budgets])
        ax.plot(budgets, 100 * stats[:, 0], "o-", color=color, lw=2, markersize=6)
        ax.fill_between(budgets, 100 * stats[:, 1], 100 * stats[:, 2], color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Candidate batch size B")
        ax.set_ylabel("Scene-averaged success [%]")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.2)
        for b, stat in zip(budgets, stats):
            ax.annotate(f"{stat[0]:.0%}", (b, 100 * stat[0]), xytext=(0, 9 if stat[0] < .1 else -16),
                        textcoords="offset points", ha="center", fontsize=9)
    count = len({str(r["scene_id"]) for r in rows})
    label = "Same state after obstacle change" if key == "changed_state_runs" else "Static nominal control" if key == "runs" else "Static random single start"
    status = payload.get('_summary', {}).get('status', 'Development observations')
    fig.suptitle(f"{label} · {count} scenes · {status}", fontsize=13)
    fig.supxlabel("Shading: conservative 95% Hoeffding interval across independent scene means. Failures remain in denominators.", fontsize=9)
    return _save(fig, output_dir, name)


def plot_paired_scenes(payload, output_dir, key="changed_state_runs", name="changed_state_per_scene"):
    rows = payload.get(key, [])
    if not rows:
        return []
    budgets = sorted({int(r["batch_size"]) for r in rows})
    compared = sorted(set([budgets[0], budgets[-1]]))
    ids = sorted({str(r["scene_id"]) for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(ids) * 0.55), 6.5),
                             sharex=True, layout="constrained")
    for j, (budget, color) in enumerate(zip(compared, (BLUE, ORANGE))):
        for k, scene_id in enumerate(ids):
            group = [r for r in rows if str(r["scene_id"]) == scene_id and int(r["batch_size"]) == budget]
            if not group:
                continue
            x = k + (j - (len(compared) - 1) / 2) * 0.22
            feasible = [r for r in group if r.get("feasible") and r.get("cost") is not None]
            regrets = [100 * (float(r["cost"]) - float(r["best_known_cost"])) /
                       max(abs(float(r["best_known_cost"])), 1e-12)
                       for r in feasible if r.get("best_known_cost") is not None]
            if regrets:
                axes[0].scatter(np.full(len(regrets), x), regrets, s=14, color=color, alpha=0.22)
                axes[0].scatter(x, np.median(regrets), s=44, color=color, marker="_", linewidths=2)
            else:
                axes[0].scatter(x, 0, s=40, color=color, marker="x", zorder=4)
            successes = np.mean([bool(r.get("near_best", False)) for r in group])
            axes[1].bar(x, 100 * successes, width=0.20, color=color, alpha=0.85,
                        label=f"B={budget}" if k == 0 else None)
    axes[0].axhline(0, color=INK, lw=0.7)
    axes[0].set_ylabel("Feasible cost gap to\nbest-known reference [%]")
    axes[0].set_title("Per-scene outcomes: every trial retained", fontsize=13)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Near-best success [%]")
    axes[1].set_xticks(range(len(ids)), ids, rotation=45, ha="right")
    axes[1].set_xlabel("Scene")
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.supxlabel("Top: dots = feasible trials, horizontal marks = medians, × = no feasible trial.\n"
                  "Bottom: failed or suboptimal trials reduce success; cost gaps are shown only for feasible trials.", fontsize=9)
    return _save(fig, output_dir, name)


def animate_mpc(scene, episode, output_dir, name="mpc_candidates"):
    snapshots = episode.get("snapshots", [])
    if not snapshots:
        return []
    history = _xy(episode.get("executed_states", [scene["x0"]]))
    limits = _episode_limits(scene, snapshots, history)
    fig, ax = plt.subplots(figsize=(6.4, 5.5), layout="constrained")

    def draw(index):
        ax.clear()
        snapshot = snapshots[index]
        active = _snapshot_scene(scene, snapshot)
        _scene_axis(ax, active)
        tick = int(snapshot.get("tick", index))
        executed_steps = int(snapshot.get("executed_steps", tick * episode.get("execute", 1)))
        path = history[:min(executed_steps + 1, len(history))]
        feasible, total = _draw_candidates(ax, active, snapshot, path)
        ax.plot(*path.T, color=INK, lw=3, zorder=6)
        ax.scatter(*_xy(snapshot.get("x0", path[-1])), color=INK, s=40, zorder=9)
        _apply_limits(ax, limits)
        ax.set_title(f"MPC step {tick} · {feasible}/{total} feasible futures\n"
                     "Thick: selected · gray dashed: infeasible", fontsize=12)

    animation = FuncAnimation(fig, draw, frames=len(snapshots), interval=250)
    path = Path(output_dir) / f"{name}.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=4), dpi=100)
    plt.close(fig)
    return [path]


def plot_campaign(payload, output_dir, *, make_animation=False):
    """Write PNG/PDF figures and an optional GIF; return all generated paths.

    The representative is the lowest-scene/lowest-seed largest-batch run, chosen
    without looking at outcome quality.  Statistical plots include all runs.
    """
    paths = []
    scenes = _scene_map(payload)
    with plt.rc_context(STYLE):
        for key, name in [('changed_state_runs', 'changed_state'), ('runs', 'static_control'),
                          ('random_single_runs', 'static_random_single')]:
            paths.extend(plot_success(payload, output_dir, key, name+'_success'))
        paths.extend(plot_paired_scenes(payload, output_dir))
        # Fixed choice: lowest scene id, lowest proposal seed, largest batch.
        # Outcome quality never enters the representative selection.
        runs = payload.get('changed_state_runs') or payload.get('runs', [])
        if runs:
            run = sorted(runs, key=lambda r: (r['scene_id'], r['proposal_seed'], -r['batch_size']))[0]
            scene = scenes.get(str(run['scene_id']))
            if scene:
                active = dict(scene)
                for key in ('x0', 'center'):
                    if key in run:
                        active[key] = run[key]
                paths.extend(plot_seed_improvement(active, run, output_dir))
        episodes = payload.get('mpc', [])
        if episodes:
            first_scene = min(e['scene_id'] for e in episodes)
            for episode in sorted((e for e in episodes if e['scene_id'] == first_scene), key=lambda e: e['batch_size']):
                scene = scenes.get(str(episode['scene_id']))
                if scene:
                    paths.extend(plot_mpc_stages(scene, episode, output_dir,
                                                name=f"mpc_stages_b{episode['batch_size']}"))
            episode = sorted(episodes, key=lambda e: (e['scene_id'], -e['batch_size']))[0]
            scene = scenes.get(str(episode['scene_id']))
            if scene and make_animation:
                paths.extend(animate_mpc(scene, episode, output_dir))
    return paths
