import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT_DIR = "traces"
DEPLOYMENT_CONFIG_FILE = "deployment_config.json"
SIMULATION_CONFIG_FILE = "simulation_config.json"

def config_default(profile, key, fallback):
    return profile.get(key, fallback)

def slugify_name(name):
    return str(name).strip().lower().replace(" ", "_")

def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def list_deployments(config_file=DEPLOYMENT_CONFIG_FILE):
    config = load_json(config_file)
    if "deployments" not in config:
        return [config.get("scene_name", "deployment")]
    return list(config["deployments"])

def list_simulation_profiles(config_file=SIMULATION_CONFIG_FILE):
    config_path = Path(config_file)
    if not config_path.exists():
        return []
    config = load_json(config_path)
    if "profiles" not in config:
        return [config_path.stem]
    return list(config["profiles"])

def load_simulation_config(config_file=SIMULATION_CONFIG_FILE,
                           profile_name=None):
    config_path = Path(config_file)
    if not config_path.exists():
        return {}, None

    config = load_json(config_path)
    profiles = config.get("profiles")
    if profiles is None:
        return config, config_path.stem

    selected = profile_name or config.get("active_profile")
    if selected is None:
        raise KeyError(f"{config_path} has profiles but no active_profile")
    if selected not in profiles:
        available = ", ".join(profiles)
        raise KeyError(
            f"{config_path} profile '{selected}' is not defined. "
            f"Available profiles: {available}"
        )
    return profiles[selected], selected

def load_deployment_config(config_file=DEPLOYMENT_CONFIG_FILE,
                           deployment_name=None):
    config_path = Path(config_file)
    config = load_json(config_path)

    if "deployments" in config:
        selected = deployment_name or config.get("active_deployment")
        deployments = config["deployments"]
        if selected not in deployments:
            available = ", ".join(deployments)
            raise KeyError(
                f"{config_path} deployment '{selected}' is not defined. "
                f"Available deployments: {available}"
            )
        deployment = deployments[selected]
        deployment_id = selected
    else:
        deployment = config
        deployment_id = config.get("scene_name", "deployment")

    for key in ["scene_name", "mesh_file", "bs_height", "bs_xy"]:
        if key not in deployment:
            raise KeyError(f"{config_path} is missing required key '{key}'")

    bs_xy = np.asarray(deployment["bs_xy"], dtype=float)
    if bs_xy.shape != (2, 2):
        raise ValueError(
            "This simplified simulator expects exactly two base stations "
            "with bs_xy shape [2, 2]"
        )

    return {
        "deployment_id": str(deployment_id),
        "scene_name": str(deployment["scene_name"]),
        "mesh_file": str(deployment["mesh_file"]),
        "bs_height": float(deployment["bs_height"]),
        "bs_xy": bs_xy,
    }

def make_output_file(deployment_config, output_dir=DEFAULT_OUTPUT_DIR,
                     output_file=None):
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment_id = slugify_name(deployment_config["deployment_id"])
    return output_dir / f"{deployment_id}_rss_trace.npz"

def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--simulation-config", default=SIMULATION_CONFIG_FILE)
    config_parser.add_argument("--simulation-profile", default=None)
    config_args, _ = config_parser.parse_known_args()
    profile, selected_profile = load_simulation_config(
        config_args.simulation_config,
        config_args.simulation_profile,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compute a simple_street_canyon coverage map and RSS/RSRP traces "
            "for UEs moving along the horizontal main street."
        ),
        parents=[config_parser],
    )
    parser.set_defaults(simulation_profile=selected_profile)
    parser.add_argument("--list-deployments", action="store_true")
    parser.add_argument("--list-simulation-profiles", action="store_true")
    parser.add_argument(
        "--deployment-config",
        default=config_default(profile, "deployment_config", DEPLOYMENT_CONFIG_FILE),
    )
    parser.add_argument(
        "--deployment",
        default=config_default(profile, "deployment", "simple_street_canyon"),
    )
    parser.add_argument(
        "--output-dir",
        default=config_default(profile, "output_dir", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--output-file",
        default=config_default(profile, "output_file", None),
    )

    parser.add_argument("--num-ues", type=int,
                        default=config_default(profile, "num_ues", 5))
    parser.add_argument("--sim-time-s", type=float,
                        default=config_default(profile, "sim_time_s", 120.0))
    parser.add_argument("--mobility-dt", type=float,
                        default=config_default(profile, "mobility_dt", 1.0))
    parser.add_argument("--ue-height", type=float,
                        default=config_default(profile, "ue_height", 1.5))
    parser.add_argument("--road-y", type=float,
                        default=config_default(profile, "road_y", 0.0))
    parser.add_argument("--road-y-min", type=float,
                        default=config_default(profile, "road_y_min", -3.0))
    parser.add_argument("--road-y-max", type=float,
                        default=config_default(profile, "road_y_max", 3.0))
    parser.add_argument("--trajectory-x-min", type=float,
                        default=config_default(profile, "trajectory_x_min", -75.0))
    parser.add_argument("--trajectory-x-max", type=float,
                        default=config_default(profile, "trajectory_x_max", 75.0))
    parser.add_argument("--min-speed", type=float,
                        default=config_default(profile, "min_speed", 0.8))
    parser.add_argument("--max-speed", type=float,
                        default=config_default(profile, "max_speed", 1.6))
    parser.add_argument("--speed-noise-std", type=float,
                        default=config_default(profile, "speed_noise_std", 0.15))
    parser.add_argument("--turn-probability-per-s", type=float,
                        default=config_default(profile, "turn_probability_per_s", 0.01))
    parser.add_argument("--pause-probability-per-s", type=float,
                        default=config_default(profile, "pause_probability_per_s", 0.005))
    parser.add_argument("--min-pause-duration-s", type=float,
                        default=config_default(profile, "min_pause_duration_s", 2.0))
    parser.add_argument("--max-pause-duration-s", type=float,
                        default=config_default(profile, "max_pause_duration_s", 5.0))
    parser.add_argument("--lateral-noise-std", type=float,
                        default=config_default(profile, "lateral_noise_std", 0.03))
    parser.add_argument("--lateral-damping", type=float,
                        default=config_default(profile, "lateral_damping", 0.9))
    parser.add_argument("--lateral-center-pull", type=float,
                        default=config_default(profile, "lateral_center_pull", 0.08))
    parser.add_argument(
        "--random-initial-y",
        action="store_true",
        default=config_default(profile, "random_initial_y", False),
        help=(
            "Sample initial lateral offsets independently at random instead "
            "of spreading UEs across the road band."
        ),
    )
    parser.add_argument(
        "--random-walk-x",
        action="store_true",
        default=config_default(profile, "random_walk_x", False),
        help=(
            "Use stochastic reflective x motion instead of forcing each UE "
            "to traverse from trajectory-x-min to trajectory-x-max."
        ),
    )
    parser.add_argument(
        "--same-direction",
        action="store_true",
        default=config_default(profile, "same_direction", False),
        help="Force all UEs to move in the positive x direction.",
    )

    parser.add_argument("--coverage-x-min", type=float,
                        default=config_default(profile, "coverage_x_min", -85.0))
    parser.add_argument("--coverage-x-max", type=float,
                        default=config_default(profile, "coverage_x_max", 85.0))
    parser.add_argument("--coverage-y-min", type=float,
                        default=config_default(profile, "coverage_y_min", -25.0))
    parser.add_argument("--coverage-y-max", type=float,
                        default=config_default(profile, "coverage_y_max", 25.0))
    parser.add_argument("--coverage-grid-stride", type=int,
                        default=config_default(profile, "coverage_grid_stride", 1))
    parser.add_argument("--coverage-batch-size", type=int,
                        default=config_default(profile, "coverage_batch_size", 128))

    parser.add_argument("--frequency-hz", type=float,
                        default=config_default(profile, "frequency_hz", 3.5e9))
    parser.add_argument("--path-max-depth", type=int,
                        default=config_default(profile, "path_max_depth", 2))
    parser.add_argument("--path-num-samples", type=int,
                        default=config_default(profile, "path_num_samples", 20000))
    parser.add_argument("--num-prbs", type=int,
                        default=config_default(profile, "num_prbs", 50))
    parser.add_argument("--subcarriers-per-prb", type=int,
                        default=config_default(profile, "subcarriers_per_prb", 12))
    parser.add_argument("--subcarrier-spacing-hz", type=float,
                        default=config_default(profile, "subcarrier_spacing_hz", 30e3))
    parser.add_argument("--tx-power-dbm-per-bs", type=float,
                        default=config_default(profile, "tx_power_dbm_per_bs", 43.0))
    parser.add_argument("--seed", type=int,
                        default=config_default(profile, "seed", 1))

    args = parser.parse_args()
    validate_args(parser, args)
    return args

def validate_args(parser, args):
    if args.num_ues < 1:
        parser.error("--num-ues must be >= 1")
    if args.mobility_dt <= 0.0:
        parser.error("--mobility-dt must be positive")
    if args.trajectory_x_min >= args.trajectory_x_max:
        parser.error("--trajectory-x-min must be smaller than --trajectory-x-max")
    if args.road_y_min >= args.road_y_max:
        parser.error("--road-y-min must be smaller than --road-y-max")
    if args.min_speed < 0.0 or args.max_speed < args.min_speed:
        parser.error("speed bounds must satisfy 0 <= --min-speed <= --max-speed")
    if args.speed_noise_std < 0.0 or args.lateral_noise_std < 0.0:
        parser.error("noise standard deviations must be non-negative")
    if not 0.0 <= args.lateral_damping <= 1.0:
        parser.error("--lateral-damping must be in [0, 1]")
    if args.lateral_center_pull < 0.0:
        parser.error("--lateral-center-pull must be non-negative")
    if args.min_pause_duration_s < 0.0 or args.max_pause_duration_s < args.min_pause_duration_s:
        parser.error("pause durations must satisfy 0 <= min <= max")
    if args.coverage_x_min >= args.coverage_x_max:
        parser.error("--coverage-x-min must be smaller than --coverage-x-max")
    if args.coverage_y_min >= args.coverage_y_max:
        parser.error("--coverage-y-min must be smaller than --coverage-y-max")
    if args.coverage_grid_stride < 1:
        parser.error("--coverage-grid-stride must be >= 1")
    if args.coverage_batch_size < 1:
        parser.error("--coverage-batch-size must be >= 1")
    if args.num_prbs < 1 or args.subcarriers_per_prb < 1:
        parser.error("--num-prbs and --subcarriers-per-prb must be >= 1")

def print_run_header(deployment_config, output_file, args, num_coverage_points):
    print("Scenario:", deployment_config["deployment_id"])
    print("Sionna scene:", deployment_config["scene_name"])
    print("Mobility mesh:", deployment_config["mesh_file"])
    print("Base stations:", deployment_config["bs_xy"].tolist())
    print("UEs:", args.num_ues)
    print("Coverage points:", num_coverage_points)
    print("Output trace:", output_file)

def save_rss_trace(
    output_file,
    deployment_config,
    simulation_config,
    bs_positions,
    mobility_times_s,
    trajectories,
    trajectory_velocities,
    trajectory_rsrp_dbm,
    trajectory_channel_gain_linear,
    trajectory_channel_gain_db,
    coverage_grid,
    coverage_rsrp_dbm,
    coverage_channel_gain_linear,
    coverage_channel_gain_db,
):
    num_bs = bs_positions.shape[0]
    coverage_grid_rsrp_dbm = np.full(
        (*coverage_grid["allowed_mask"].shape, num_bs),
        np.nan,
        dtype=float,
    )
    if len(coverage_grid["allowed_indices"]) > 0:
        yy = coverage_grid["allowed_indices"][:, 0]
        xx = coverage_grid["allowed_indices"][:, 1]
        coverage_grid_rsrp_dbm[yy, xx, :] = coverage_rsrp_dbm
    coverage_grid_channel_gain_linear = np.full_like(
        coverage_grid_rsrp_dbm,
        np.nan,
    )
    coverage_grid_channel_gain_db = np.full_like(
        coverage_grid_rsrp_dbm,
        np.nan,
    )
    if len(coverage_grid["allowed_indices"]) > 0:
        yy = coverage_grid["allowed_indices"][:, 0]
        xx = coverage_grid["allowed_indices"][:, 1]
        coverage_grid_channel_gain_linear[yy, xx, :] = coverage_channel_gain_linear
        coverage_grid_channel_gain_db[yy, xx, :] = coverage_channel_gain_db

    np.savez_compressed(
        output_file,
        deployment_config=np.asarray(deployment_config, dtype=object),
        simulation_config=np.asarray(simulation_config, dtype=object),
        scene_name=deployment_config["scene_name"],
        mesh_file=deployment_config["mesh_file"],
        bs_positions=bs_positions,
        mobility_times_s=mobility_times_s,
        trajectories=trajectories,
        trajectory_velocities=trajectory_velocities,
        trajectory_rsrp_dbm=trajectory_rsrp_dbm,
        trajectory_channel_gain_linear=trajectory_channel_gain_linear,
        trajectory_channel_gain_db=trajectory_channel_gain_db,
        coverage_x_values=coverage_grid["x_values"],
        coverage_y_values=coverage_grid["y_values"],
        coverage_allowed_mask=coverage_grid["allowed_mask"],
        coverage_points_xy=coverage_grid["points_xy"],
        coverage_points_xyz=coverage_grid["points_xyz"],
        coverage_rsrp_dbm=coverage_rsrp_dbm,
        coverage_grid_rsrp_dbm=coverage_grid_rsrp_dbm,
        coverage_channel_gain_linear=coverage_channel_gain_linear,
        coverage_channel_gain_db=coverage_channel_gain_db,
        coverage_grid_channel_gain_linear=coverage_grid_channel_gain_linear,
        coverage_grid_channel_gain_db=coverage_grid_channel_gain_db,
    )
