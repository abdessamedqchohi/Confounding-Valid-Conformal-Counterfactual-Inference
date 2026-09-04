import time

import numpy as np

from channel_simulation import (
    ChannelConfig,
    SionnaChannelSimulator,
    bs_positions_from_xy,
)
from link_abstraction import (
    LinkConfig,
    channel_gain_per_prb,
    compute_rsrp,
    linear_to_db,
)
from mobility import (
    build_coverage_grid,
    generate_horizontal_main_street_mobility,
)
from utils import (
    list_deployments,
    list_simulation_profiles,
    load_deployment_config,
    make_output_file,
    parse_args,
    print_run_header,
    save_rss_trace,
)

def build_channel_config(args):
    return ChannelConfig(
        frequency_hz=args.frequency_hz,
        path_max_depth=args.path_max_depth,
        path_num_samples=args.path_num_samples,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        normalize_cfr=False,
        seed=args.seed,
    )

def build_link_config(args):
    return LinkConfig(
        num_prbs=args.num_prbs,
        subcarriers_per_prb=args.subcarriers_per_prb,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        tx_power_dbm_per_bs=args.tx_power_dbm_per_bs,
    )

def build_subcarrier_frequencies(link_config):
    num_subcarriers = link_config.num_subcarriers
    spacing = link_config.subcarrier_spacing_hz
    return (np.arange(num_subcarriers) - 0.5 * (num_subcarriers - 1)) * spacing

def link_metrics_from_physical(physical, link_config):
    gain_linear = np.mean(channel_gain_per_prb(physical, link_config), axis=-1)
    return {
        "rsrp_dbm": compute_rsrp(physical, link_config)["rsrp_dbm"],
        "channel_gain_linear": gain_linear,
        "channel_gain_db": linear_to_db(gain_linear),
    }

def simulate_coverage_metric_batches(
    simulator,
    positions_xyz,
    frequencies_hz,
    link_config,
    batch_size,
    first_step=False,
    time_s=0.0,
):
    positions_xyz = np.asarray(positions_xyz, dtype=float)
    if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
        raise ValueError("positions_xyz must have shape [num_positions, 3]")

    outputs = {
        "rsrp_dbm": [],
        "channel_gain_linear": [],
        "channel_gain_db": [],
    }
    batch_size = int(batch_size)
    for start in range(0, len(positions_xyz), batch_size):
        end = min(start + batch_size, len(positions_xyz))
        segment = positions_xyz[start:end, None, :]
        physical = simulator.simulate_static_position_segment(
            segment,
            frequencies_hz=frequencies_hz,
            first_step=first_step and start == 0,
            time_s=time_s,
        )
        metrics = link_metrics_from_physical(physical, link_config)
        for key in outputs:
            outputs[key].append(metrics[key][:, 0, :])

    if not outputs["rsrp_dbm"]:
        empty = np.zeros((0, simulator.bs_positions.shape[0]), dtype=float)
        return {key: empty.copy() for key in outputs}
    return {key: np.vstack(value) for key, value in outputs.items()}

def coverage_values_to_grid(coverage_grid, coverage_values, num_bs):
    grid = np.full(
        (*coverage_grid["allowed_mask"].shape, num_bs),
        np.nan,
        dtype=float,
    )
    if len(coverage_grid["allowed_indices"]) > 0:
        yy = coverage_grid["allowed_indices"][:, 0]
        xx = coverage_grid["allowed_indices"][:, 1]
        grid[yy, xx, :] = coverage_values
    return grid

def sample_trajectory_values_from_coverage(coverage_grid, coverage_grid_values,
                                           trajectories):
    trajectories = np.asarray(trajectories, dtype=float)
    x_values = np.asarray(coverage_grid["x_values"], dtype=float)
    y_values = np.asarray(coverage_grid["y_values"], dtype=float)
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("coverage grid needs at least two x and y coordinates")

    dx = float(x_values[1] - x_values[0])
    dy = float(y_values[1] - y_values[0])
    xy = trajectories[:, :, :2]
    x_idx = np.rint((xy[:, :, 0] - x_values[0]) / dx).astype(int)
    y_idx = np.rint((xy[:, :, 1] - y_values[0]) / dy).astype(int)
    x_idx = np.clip(x_idx, 0, len(x_values) - 1)
    y_idx = np.clip(y_idx, 0, len(y_values) - 1)

    sampled = coverage_grid_values[y_idx, x_idx, :]

    missing = ~np.all(np.isfinite(sampled), axis=-1)
    if np.any(missing):
        valid = np.argwhere(np.all(np.isfinite(coverage_grid_values), axis=-1))
        if len(valid) == 0:
            raise ValueError("coverage grid has no finite values")
        valid_xy = np.column_stack([
            x_values[valid[:, 1]],
            y_values[valid[:, 0]],
        ])
        valid_values = coverage_grid_values[valid[:, 0], valid[:, 1], :]

        for ue_idx, time_idx in np.argwhere(missing):
            point_xy = xy[ue_idx, time_idx]
            nearest = int(np.argmin(np.sum((valid_xy - point_xy) ** 2, axis=1)))
            sampled[ue_idx, time_idx, :] = valid_values[nearest]

    return np.moveaxis(sampled, 0, 1)

def run_simulation(args=None):
    if args is None:
        args = parse_args()

    if args.list_deployments:
        print("Available deployments:")
        for deployment_name in list_deployments(args.deployment_config):
            print(f"  {deployment_name}")
        return None

    if args.list_simulation_profiles:
        print("Available simulation profiles:")
        for profile_name in list_simulation_profiles(args.simulation_config):
            print(f"  {profile_name}")
        return None

    deployment_config = load_deployment_config(
        config_file=args.deployment_config,
        deployment_name=args.deployment,
    )
    output_file = make_output_file(
        deployment_config,
        output_dir=args.output_dir,
        output_file=args.output_file,
    )
    bs_positions = bs_positions_from_xy(
        deployment_config["bs_xy"],
        deployment_config["bs_height"],
    )
    channel_config = build_channel_config(args)
    link_config = build_link_config(args)
    frequencies_hz = build_subcarrier_frequencies(link_config)

    coverage_grid = build_coverage_grid(
        deployment_config["mesh_file"],
        ue_height=args.ue_height,
        bounds=(
            args.coverage_x_min,
            args.coverage_x_max,
            args.coverage_y_min,
            args.coverage_y_max,
        ),
        stride=args.coverage_grid_stride,
    )
    trajectories, trajectory_velocities, mobility_metadata = (
        generate_horizontal_main_street_mobility(
            num_ues=args.num_ues,
            sim_time_s=args.sim_time_s,
            dt=args.mobility_dt,
            ue_height=args.ue_height,
            x_min=args.trajectory_x_min,
            x_max=args.trajectory_x_max,
            road_y=args.road_y,
            road_y_min=args.road_y_min,
            road_y_max=args.road_y_max,
            min_speed=args.min_speed,
            max_speed=args.max_speed,
            speed_noise_std=args.speed_noise_std,
            turn_probability_per_s=args.turn_probability_per_s,
            pause_probability_per_s=args.pause_probability_per_s,
            min_pause_duration_s=args.min_pause_duration_s,
            max_pause_duration_s=args.max_pause_duration_s,
            lateral_noise_std=args.lateral_noise_std,
            lateral_damping=args.lateral_damping,
            lateral_center_pull=args.lateral_center_pull,
            spread_initial_y=not args.random_initial_y,
            force_left_to_right=not args.random_walk_x,
            random_directions=not args.same_direction,
            seed=args.seed,
        )
    )
    mobility_times_s = mobility_metadata["times_s"]

    print_run_header(
        deployment_config,
        output_file,
        args,
        num_coverage_points=len(coverage_grid["points_xyz"]),
    )
    for ue_idx, speed in enumerate(mobility_metadata["signed_speed_mps"]):
        print(f"UE {ue_idx}: signed speed {speed:.2f} m/s")

    initial_positions = coverage_grid["points_xyz"][:1]
    if len(initial_positions) == 0:
        initial_positions = trajectories[:, 0, :]

    simulator = SionnaChannelSimulator(
        bs_positions,
        config=channel_config,
        scene_name=deployment_config["scene_name"],
    )
    print("Initializing Sionna scene...")
    simulator.reset(initial_positions)

    start_time = time.time()
    print("Computing coverage map RSS/RSRP and channel gain...")
    coverage_metrics = simulate_coverage_metric_batches(
        simulator,
        coverage_grid["points_xyz"],
        frequencies_hz=frequencies_hz,
        link_config=link_config,
        batch_size=args.coverage_batch_size,
        first_step=True,
        time_s=0.0,
    )
    print(
        f"Coverage map done in {time.time() - start_time:.1f}s "
        f"for {len(coverage_metrics['rsrp_dbm'])} points"
    )
    coverage_rsrp_dbm = coverage_metrics["rsrp_dbm"]
    coverage_channel_gain_linear = coverage_metrics["channel_gain_linear"]
    coverage_channel_gain_db = coverage_metrics["channel_gain_db"]
    coverage_grid_rsrp_dbm = coverage_values_to_grid(
        coverage_grid,
        coverage_rsrp_dbm,
        num_bs=bs_positions.shape[0],
    )
    coverage_grid_channel_gain_linear = coverage_values_to_grid(
        coverage_grid,
        coverage_channel_gain_linear,
        num_bs=bs_positions.shape[0],
    )
    coverage_grid_channel_gain_db = coverage_values_to_grid(
        coverage_grid,
        coverage_channel_gain_db,
        num_bs=bs_positions.shape[0],
    )

    print("Sampling trajectory RSS/RSRP and channel gain from the coverage map...")
    trajectory_rsrp_dbm = sample_trajectory_values_from_coverage(
        coverage_grid,
        coverage_grid_rsrp_dbm,
        trajectories,
    )
    trajectory_channel_gain_linear = sample_trajectory_values_from_coverage(
        coverage_grid,
        coverage_grid_channel_gain_linear,
        trajectories,
    )
    trajectory_channel_gain_db = sample_trajectory_values_from_coverage(
        coverage_grid,
        coverage_grid_channel_gain_db,
        trajectories,
    )

    simulation_config = vars(args).copy()
    simulation_config["mobility_metadata"] = mobility_metadata
    save_rss_trace(
        output_file=output_file,
        deployment_config=deployment_config,
        simulation_config=simulation_config,
        bs_positions=bs_positions,
        mobility_times_s=mobility_times_s,
        trajectories=trajectories,
        trajectory_velocities=trajectory_velocities,
        trajectory_rsrp_dbm=trajectory_rsrp_dbm,
        trajectory_channel_gain_linear=trajectory_channel_gain_linear,
        trajectory_channel_gain_db=trajectory_channel_gain_db,
        coverage_grid=coverage_grid,
        coverage_rsrp_dbm=coverage_rsrp_dbm,
        coverage_channel_gain_linear=coverage_channel_gain_linear,
        coverage_channel_gain_db=coverage_channel_gain_db,
    )

    print()
    print(f"Saved RSS trace to {output_file}")
    print("trajectory_rsrp_dbm shape:", trajectory_rsrp_dbm.shape)
    print("trajectory_channel_gain_linear shape:", trajectory_channel_gain_linear.shape)
    print("coverage_grid_rsrp_dbm shape:", (*coverage_grid["allowed_mask"].shape, bs_positions.shape[0]))

    return {
        "output_file": output_file,
        "bs_positions": bs_positions,
        "trajectories": trajectories,
        "trajectory_rsrp_dbm": trajectory_rsrp_dbm,
        "trajectory_channel_gain_linear": trajectory_channel_gain_linear,
        "coverage_rsrp_dbm": coverage_rsrp_dbm,
        "coverage_channel_gain_linear": coverage_channel_gain_linear,
    }

if __name__ == "__main__":
    run_simulation()
