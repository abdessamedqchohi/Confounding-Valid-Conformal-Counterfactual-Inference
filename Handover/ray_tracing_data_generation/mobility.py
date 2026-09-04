import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MESH_FILE = "meshes/meshese_simple_street_canyon.npz"
DEFAULT_DEPLOYMENT_CONFIG_FILE = "deployment_config.json"

def load_allowed_mesh(mesh_file=DEFAULT_MESH_FILE):
    mesh = np.load(mesh_file)
    required = ["allowed_mask", "x_values", "y_values"]
    missing = [key for key in required if key not in mesh]
    if missing:
        raise KeyError(f"{mesh_file} is missing keys: {', '.join(missing)}")

    return {
        "allowed_mask": mesh["allowed_mask"].astype(bool),
        "x_values": np.asarray(mesh["x_values"], dtype=float),
        "y_values": np.asarray(mesh["y_values"], dtype=float),
        "grid_step": float(mesh["grid_step"]) if "grid_step" in mesh else None,
    }

def normalize_bounds(bounds):
    if bounds is None:
        return None
    if len(bounds) != 4:
        raise ValueError("bounds must be (x_min, x_max, y_min, y_max)")

    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bounds must satisfy x_min < x_max and y_min < y_max")
    return x_min, x_max, y_min, y_max

def mesh_indices_in_bounds(x_values, y_values, bounds=None):
    bounds = normalize_bounds(bounds)
    if bounds is None:
        return np.arange(len(x_values)), np.arange(len(y_values))

    x_min, x_max, y_min, y_max = bounds
    x_idx = np.flatnonzero((x_values >= x_min) & (x_values <= x_max))
    y_idx = np.flatnonzero((y_values >= y_min) & (y_values <= y_max))
    if len(x_idx) == 0 or len(y_idx) == 0:
        raise ValueError("bounds do not overlap the mesh grid")
    return x_idx, y_idx

def apply_deployment_bounds(allowed_mask, x_values, y_values, bounds=None):
    x_idx, y_idx = mesh_indices_in_bounds(x_values, y_values, bounds)
    bounded = np.zeros_like(allowed_mask, dtype=bool)
    bounded[np.ix_(y_idx, x_idx)] = allowed_mask[np.ix_(y_idx, x_idx)]
    return bounded

def build_coverage_grid(mesh_file, ue_height=1.5, bounds=None, stride=1):
    mesh = load_allowed_mesh(mesh_file)
    x_idx, y_idx = mesh_indices_in_bounds(
        mesh["x_values"],
        mesh["y_values"],
        bounds,
    )
    stride = int(stride)
    if stride < 1:
        raise ValueError("coverage stride must be >= 1")

    x_idx = x_idx[::stride]
    y_idx = y_idx[::stride]
    allowed = mesh["allowed_mask"][np.ix_(y_idx, x_idx)]
    yy, xx = np.nonzero(allowed)
    points_xy = np.column_stack([
        mesh["x_values"][x_idx[xx]],
        mesh["y_values"][y_idx[yy]],
    ])
    points_xyz = np.column_stack([
        points_xy,
        np.full(len(points_xy), float(ue_height)),
    ])

    return {
        "x_values": mesh["x_values"][x_idx],
        "y_values": mesh["y_values"][y_idx],
        "allowed_mask": allowed,
        "allowed_indices": np.column_stack([yy, xx]),
        "points_xy": points_xy,
        "points_xyz": points_xyz,
    }

def generate_horizontal_main_street_mobility(
    num_ues=1,
    sim_time_s=120.0,
    dt=1.0,
    ue_height=1.5,
    x_min=-75.0,
    x_max=75.0,
    road_y=0.0,
    road_y_min=-3.0,
    road_y_max=3.0,
    min_speed=0.8,
    max_speed=1.6,
    speed_noise_std=0.15,
    turn_probability_per_s=0.01,
    pause_probability_per_s=0.005,
    min_pause_duration_s=2.0,
    max_pause_duration_s=5.0,
    lateral_noise_std=0.03,
    lateral_damping=0.9,
    lateral_center_pull=0.08,
    spread_initial_y=True,
    force_left_to_right=True,
    random_directions=True,
    seed=1,
    **_unused,
):
    if num_ues < 1:
        raise ValueError("num_ues must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if sim_time_s < 0.0:
        raise ValueError("sim_time_s must be non-negative")
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max")
    if road_y_min >= road_y_max:
        raise ValueError("road_y_min must be smaller than road_y_max")
    if min_speed < 0.0 or max_speed < min_speed:
        raise ValueError("speed bounds must satisfy 0 <= min_speed <= max_speed")
    if speed_noise_std < 0.0 or lateral_noise_std < 0.0:
        raise ValueError("noise standard deviations must be non-negative")
    if not 0.0 <= lateral_damping <= 1.0:
        raise ValueError("lateral_damping must be in [0, 1]")
    if lateral_center_pull < 0.0:
        raise ValueError("lateral_center_pull must be non-negative")
    if min_pause_duration_s < 0.0 or max_pause_duration_s < min_pause_duration_s:
        raise ValueError("pause durations must satisfy 0 <= min <= max")

    rng = np.random.default_rng(seed)
    num_steps = int(np.floor(sim_time_s / dt)) + 1
    times_s = np.arange(num_steps, dtype=float) * float(dt)

    initial_x = rng.uniform(x_min, x_max, size=num_ues)
    speeds = rng.uniform(min_speed, max_speed, size=num_ues)
    directions = rng.choice([-1.0, 1.0], size=num_ues) if random_directions else np.ones(num_ues)
    if spread_initial_y and num_ues > 1:
        road_width = road_y_max - road_y_min
        margin = 0.05 * road_width
        initial_y = np.linspace(
            road_y_min + margin,
            road_y_max - margin,
            num_ues,
        )
        initial_y = initial_y[rng.permutation(num_ues)]
    else:
        initial_y = rng.uniform(road_y_min, road_y_max, size=num_ues)
    preferred_y = initial_y.copy()

    trajectories = np.zeros((num_ues, num_steps, 3), dtype=float)
    velocities = np.zeros_like(trajectories)
    signed_speeds = np.zeros((num_ues, num_steps), dtype=float)
    pause_state = np.zeros((num_ues, num_steps), dtype=bool)

    turn_probability = min(1.0, float(turn_probability_per_s) * float(dt))
    pause_probability = min(1.0, float(pause_probability_per_s) * float(dt))
    speed_noise_scale = float(speed_noise_std) * np.sqrt(float(dt))
    lateral_noise_scale = float(lateral_noise_std) * np.sqrt(float(dt))

    for ue_idx in range(num_ues):
        if force_left_to_right:
            speed = float(speeds[ue_idx])
            pause_steps_left = 0
            raw_step_distances = np.zeros(num_steps - 1, dtype=float)

            for step in range(1, num_steps):
                if pause_steps_left <= 0 and rng.random() < pause_probability:
                    pause_duration = rng.uniform(min_pause_duration_s, max_pause_duration_s)
                    pause_steps_left = int(np.round(pause_duration / dt))

                paused = pause_steps_left > 0
                if paused:
                    pause_steps_left -= 1
                    pause_state[ue_idx, step] = True
                    raw_step_distances[step - 1] = 0.0
                    continue

                speed = float(np.clip(
                    speed + rng.normal(0.0, speed_noise_scale),
                    min_speed,
                    max_speed,
                ))
                raw_step_distances[step - 1] = speed * dt

            if np.sum(raw_step_distances) <= 0.0:
                raw_step_distances[:] = 1.0

            progress = np.concatenate([[0.0], np.cumsum(raw_step_distances)])
            progress /= progress[-1]
            x_series = x_min + progress * (x_max - x_min)

            y = float(initial_y[ue_idx])
            lateral_velocity = 0.0
            y_series = np.zeros(num_steps, dtype=float)
            y_series[0] = y
            for step in range(1, num_steps):
                lateral_velocity = (
                    lateral_damping * lateral_velocity
                    + rng.normal(0.0, lateral_noise_scale)
                    - lateral_center_pull * (y - preferred_y[ue_idx])
                )
                y = y + lateral_velocity * dt
                if y > road_y_max:
                    y = road_y_max - (y - road_y_max)
                    lateral_velocity *= -0.5
                elif y < road_y_min:
                    y = road_y_min + (road_y_min - y)
                    lateral_velocity *= -0.5
                y_series[step] = float(np.clip(y, road_y_min, road_y_max))

            trajectories[ue_idx, :, 0] = x_series
            trajectories[ue_idx, :, 1] = y_series
            trajectories[ue_idx, :, 2] = ue_height
            velocities[ue_idx, 1:, 0] = np.diff(x_series) / dt
            velocities[ue_idx, 1:, 1] = np.diff(y_series) / dt
            signed_speeds[ue_idx, :] = velocities[ue_idx, :, 0]
            continue

        x = float(initial_x[ue_idx])
        y = float(initial_y[ue_idx])
        speed = float(speeds[ue_idx])
        direction = float(directions[ue_idx])
        lateral_velocity = 0.0
        pause_steps_left = 0

        trajectories[ue_idx, 0] = [x, y, ue_height]
        velocities[ue_idx, 0, 0] = direction * speed
        signed_speeds[ue_idx, 0] = direction * speed

        for step in range(1, num_steps):
            previous_x = x
            previous_y = y

            if pause_steps_left <= 0 and rng.random() < pause_probability:
                pause_duration = rng.uniform(min_pause_duration_s, max_pause_duration_s)
                pause_steps_left = int(np.round(pause_duration / dt))

            paused = pause_steps_left > 0
            if paused:
                pause_steps_left -= 1
                longitudinal_step = 0.0
            else:
                if rng.random() < turn_probability:
                    direction *= -1.0

                speed = float(np.clip(
                    speed + rng.normal(0.0, speed_noise_scale),
                    min_speed,
                    max_speed,
                ))
                longitudinal_step = direction * speed * dt

            x = x + longitudinal_step
            while x < x_min or x > x_max:
                if x > x_max:
                    x = x_max - (x - x_max)
                    direction = -1.0
                elif x < x_min:
                    x = x_min + (x_min - x)
                    direction = 1.0

            lateral_velocity = (
                lateral_damping * lateral_velocity
                + rng.normal(0.0, lateral_noise_scale)
                - lateral_center_pull * (y - preferred_y[ue_idx])
            )
            y = y + lateral_velocity * dt
            if y > road_y_max:
                y = road_y_max - (y - road_y_max)
                lateral_velocity *= -0.5
            elif y < road_y_min:
                y = road_y_min + (road_y_min - y)
                lateral_velocity *= -0.5
            y = float(np.clip(y, road_y_min, road_y_max))

            trajectories[ue_idx, step] = [x, y, ue_height]
            velocities[ue_idx, step, 0] = (x - previous_x) / dt
            velocities[ue_idx, step, 1] = (y - previous_y) / dt
            signed_speeds[ue_idx, step] = direction * speed if not paused else 0.0
            pause_state[ue_idx, step] = paused

    metadata = {
        "model": "stochastic_horizontal_main_street",
        "times_s": times_s,
        "initial_x": initial_x,
        "initial_y": initial_y,
        "preferred_y": preferred_y,
        "road_y": float(road_y),
        "road_y_min": float(road_y_min),
        "road_y_max": float(road_y_max),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "initial_speed_mps": speeds,
        "signed_speed_mps": signed_speeds[:, 0],
        "signed_speed_time_series_mps": signed_speeds,
        "pause_state": pause_state,
        "speed_noise_std": float(speed_noise_std),
        "turn_probability_per_s": float(turn_probability_per_s),
        "pause_probability_per_s": float(pause_probability_per_s),
        "min_pause_duration_s": float(min_pause_duration_s),
        "max_pause_duration_s": float(max_pause_duration_s),
        "lateral_noise_std": float(lateral_noise_std),
        "lateral_damping": float(lateral_damping),
        "lateral_center_pull": float(lateral_center_pull),
        "spread_initial_y": bool(spread_initial_y),
        "force_left_to_right": bool(force_left_to_right),
        "random_directions": bool(random_directions),
        "seed": seed,
    }
    return trajectories, velocities, metadata

def generate_fixed_duration_waypoint_mobility(*args, **kwargs):
    trajectories, _velocities, metadata = generate_horizontal_main_street_mobility(
        *args,
        **kwargs,
    )
    return trajectories, metadata

def load_deployment_config(config_file=DEFAULT_DEPLOYMENT_CONFIG_FILE,
                           deployment_name=None):
    with Path(config_file).open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "deployments" in config:
        selected = deployment_name or config.get("active_deployment")
        config = config["deployments"][selected]

    return {
        "scene_name": str(config["scene_name"]),
        "mesh_file": str(config["mesh_file"]),
        "bs_height": float(config["bs_height"]),
        "bs_xy": np.asarray(config["bs_xy"], dtype=float),
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate simple horizontal main-street UE mobility."
    )
    parser.add_argument("--num-ues", type=int, default=1)
    parser.add_argument("--sim-time-s", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--ue-height", type=float, default=1.5)
    parser.add_argument("--x-min", type=float, default=-75.0)
    parser.add_argument("--x-max", type=float, default=75.0)
    parser.add_argument("--road-y", type=float, default=0.0)
    parser.add_argument("--road-y-min", type=float, default=-3.0)
    parser.add_argument("--road-y-max", type=float, default=3.0)
    parser.add_argument("--min-speed", type=float, default=0.8)
    parser.add_argument("--max-speed", type=float, default=1.6)
    parser.add_argument("--speed-noise-std", type=float, default=0.15)
    parser.add_argument("--turn-probability-per-s", type=float, default=0.01)
    parser.add_argument("--pause-probability-per-s", type=float, default=0.005)
    parser.add_argument("--min-pause-duration-s", type=float, default=2.0)
    parser.add_argument("--max-pause-duration-s", type=float, default=5.0)
    parser.add_argument("--lateral-noise-std", type=float, default=0.03)
    parser.add_argument("--lateral-damping", type=float, default=0.9)
    parser.add_argument("--lateral-center-pull", type=float, default=0.08)
    parser.add_argument("--random-initial-y", action="store_true")
    parser.add_argument("--random-walk-x", action="store_true")
    parser.add_argument("--same-direction", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-trace", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    trajectories, velocities, metadata = generate_horizontal_main_street_mobility(
        num_ues=args.num_ues,
        sim_time_s=args.sim_time_s,
        dt=args.dt,
        ue_height=args.ue_height,
        x_min=args.x_min,
        x_max=args.x_max,
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
    print(f"Generated {trajectories.shape[0]} UE trajectories with {trajectories.shape[1]} steps")
    for ue_idx, speed in enumerate(metadata["signed_speed_mps"]):
        print(f"UE {ue_idx}: signed speed {speed:.2f} m/s")

    if args.save_trace:
        np.savez_compressed(
            args.save_trace,
            trajectories=trajectories,
            velocities=velocities,
            metadata=np.asarray(metadata, dtype=object),
        )
        print("Saved trace:", args.save_trace)

if __name__ == "__main__":
    main()
