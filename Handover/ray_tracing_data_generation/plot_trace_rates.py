import argparse
from pathlib import Path

import matplotlib
import numpy as np

DEFAULT_TRACE = "traces/simple_street_canyon_rss_trace.npz"

def load_trace(trace_file):
    trace = np.load(trace_file, allow_pickle=True)
    required = [
        "mobility_times_s",
        "trajectories",
        "bs_positions",
        "trajectory_rsrp_dbm",
        "coverage_x_values",
        "coverage_y_values",
        "coverage_grid_rsrp_dbm",
    ]
    missing = [key for key in required if key not in trace]
    if missing:
        raise KeyError(f"{trace_file} is missing keys: {', '.join(missing)}")

    return {key: trace[key] for key in required}

def output_paths(trace_file, output_dir=None, prefix=None):
    trace_file = Path(trace_file)
    output_dir = Path(output_dir) if output_dir is not None else trace_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or trace_file.stem
    return {
        "deployment": output_dir / f"{stem}_deployment.png",
        "coverage": output_dir / f"{stem}_coverage.png",
        "coverage_bs0": output_dir / f"{stem}_coverage_bs0.png",
        "coverage_bs1": output_dir / f"{stem}_coverage_bs1.png",
        "trajectory_rsrp": output_dir / f"{stem}_trajectory_rsrp.png",
    }

def plot_deployment(arrays, output_file, show=False):
    import matplotlib.pyplot as plt

    trajectories = np.asarray(arrays["trajectories"], dtype=float)
    bs_positions = np.asarray(arrays["bs_positions"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for ue_idx, trajectory in enumerate(trajectories):
        xy = trajectory[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.8, label=f"UE {ue_idx}")
        ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=28)
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="x", s=42)

    ax.scatter(
        bs_positions[:, 0],
        bs_positions[:, 1],
        marker="^",
        s=110,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.8,
        label="Base stations",
        zorder=5,
    )
    for bs_idx, (x_pos, y_pos) in enumerate(bs_positions[:, :2]):
        ax.annotate(
            f"BS {bs_idx}",
            (x_pos, y_pos),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Deployment and Horizontal UE Trajectories")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_coverage(arrays, output_file, show=False):
    import matplotlib.pyplot as plt

    x_values = np.asarray(arrays["coverage_x_values"], dtype=float)
    y_values = np.asarray(arrays["coverage_y_values"], dtype=float)
    grid_rsrp = np.asarray(arrays["coverage_grid_rsrp_dbm"], dtype=float)
    trajectories = np.asarray(arrays["trajectories"], dtype=float)
    bs_positions = np.asarray(arrays["bs_positions"], dtype=float)
    best_rsrp = np.nanmax(grid_rsrp, axis=2)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
    image = ax.imshow(
        best_rsrp,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="Best BS RSRP [dBm]")

    for ue_idx, trajectory in enumerate(trajectories):
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=1.5,
            color="white",
            alpha=0.85,
            label=f"UE {ue_idx}",
        )

    ax.scatter(
        bs_positions[:, 0],
        bs_positions[:, 1],
        marker="^",
        s=110,
        color="tab:red",
        edgecolor="black",
        linewidth=0.8,
        label="Base stations",
        zorder=5,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Coverage Map")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_coverage_per_bs(arrays, output_files, show=False):
    import matplotlib.pyplot as plt

    x_values = np.asarray(arrays["coverage_x_values"], dtype=float)
    y_values = np.asarray(arrays["coverage_y_values"], dtype=float)
    grid_rsrp = np.asarray(arrays["coverage_grid_rsrp_dbm"], dtype=float)
    trajectories = np.asarray(arrays["trajectories"], dtype=float)
    bs_positions = np.asarray(arrays["bs_positions"], dtype=float)
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]

    finite_values = grid_rsrp[np.isfinite(grid_rsrp)]
    if finite_values.size:
        vmin = float(np.nanpercentile(finite_values, 5))
        vmax = float(np.nanpercentile(finite_values, 95))
    else:
        vmin, vmax = None, None

    for bs_idx, output_file in enumerate(output_files):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        image = ax.imshow(
            grid_rsrp[:, :, bs_idx],
            origin="lower",
            extent=extent,
            cmap="viridis",
            aspect="equal",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax, label=f"BS {bs_idx} RSRP [dBm]")

        for ue_idx, trajectory in enumerate(trajectories):
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=1.5,
                color="white",
                alpha=0.85,
                label=f"UE {ue_idx}",
            )

        ax.scatter(
            bs_positions[:, 0],
            bs_positions[:, 1],
            marker="^",
            s=110,
            color="tab:red",
            edgecolor="black",
            linewidth=0.8,
            label="Base stations",
            zorder=5,
        )
        ax.scatter(
            bs_positions[bs_idx, 0],
            bs_positions[bs_idx, 1],
            marker="^",
            s=170,
            color="yellow",
            edgecolor="black",
            linewidth=1.0,
            label=f"BS {bs_idx}",
            zorder=6,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Coverage Map - BS {bs_idx}")
        ax.legend(loc="best", fontsize=8)
        fig.savefig(output_file, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_trajectory_rsrp(arrays, output_file, show=False):
    import matplotlib.pyplot as plt

    times_s = np.asarray(arrays["mobility_times_s"], dtype=float)
    rsrp = np.asarray(arrays["trajectory_rsrp_dbm"], dtype=float)
    num_steps, num_ues, num_bs = rsrp.shape

    fig, axes = plt.subplots(
        num_ues,
        1,
        figsize=(8, max(3.0, 2.8 * num_ues)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ue_idx in range(num_ues):
        ax = axes[ue_idx]
        for bs_idx in range(num_bs):
            ax.plot(
                times_s[:num_steps],
                rsrp[:, ue_idx, bs_idx],
                linewidth=1.8,
                label=f"BS {bs_idx}",
            )
        ax.set_title(f"UE {ue_idx}")
        ax.set_ylabel("RSRP [dBm]")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Trajectory RSS/RSRP at the Two Base Stations")
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_trace(trace_file, output_dir=None, output_prefix=None, show=False):
    if not show:
        matplotlib.use("Agg")

    arrays = load_trace(trace_file)
    paths = output_paths(trace_file, output_dir, output_prefix)
    plot_deployment(arrays, paths["deployment"], show=show)
    plot_coverage(arrays, paths["coverage"], show=show)
    plot_coverage_per_bs(
        arrays,
        [paths["coverage_bs0"], paths["coverage_bs1"]],
        show=show,
    )
    plot_trajectory_rsrp(arrays, paths["trajectory_rsrp"], show=show)
    return paths

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot simplified simple_street_canyon RSS trace outputs."
    )
    parser.add_argument("trace_file", nargs="?", default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    paths = plot_trace(
        args.trace_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        show=args.show,
    )
    for label, path in paths.items():
        print(f"Saved {label}: {path}")

if __name__ == "__main__":
    main()
