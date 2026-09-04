import argparse
import glob
from pathlib import Path
import os
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an observational handover dataset from RSS/channel-gain "
            "traces produced by simulate.py."
        )
    )
    parser.add_argument(
        "--trace-glob",
        default="ray_tracing_data_generation/traces/*_rss_trace.npz",
        help="Glob selecting generated trace files.",
    )
    parser.add_argument(
        "--output-file",
        default="data/handover_dataset.npy",
        help="Output NPZ dataset path.",
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--int-frac", type=float, default=0.05)
    parser.add_argument("--history-s", type=float, default=10.0)
    parser.add_argument("--outcome-horizon-s", type=float, default=10.0)
    parser.add_argument(
        "--min-opportunity-steps",
        type=int,
        default=10,
        help=(
            "Minimum consecutive samples for RSS_BS1 < RSS_BS2 before a "
            "time is considered a handover opportunity."
        ),
    )
    parser.add_argument("--load-min", type=float, default=0.05)
    parser.add_argument("--load-max", type=float, default=0.95)
    parser.add_argument(
        "--proposal-margin-db",
        type=float,
        default=5.0,
        help="RSS2-RSS1 margin where BS1 proposal probability is 0.5.",
    )
    parser.add_argument(
        "--proposal-slope",
        type=float,
        default=0.8,
        help="Sigmoid slope for BS1 proposal probability per dB.",
    )
    parser.add_argument(
        "--accept-load-threshold",
        type=float,
        default=0.65,
        help="BS2 load where acceptance probability is 0.5.",
    )
    parser.add_argument(
        "--accept-slope",
        type=float,
        default=8.0,
        help="Base sigmoid slope for BS2 acceptance probability.",
    )
    parser.add_argument(
        "--confounding-strength",
        type=float,
        default=1.0,
        help=(
            "Scalar in [0, 1] multiplying --accept-slope. Zero makes acceptance "
            "independent of the hidden BS2 load; one uses the full base slope."
        ),
    )
    parser.add_argument(
        "--min-resource",
        type=float,
        default=0.05,
        help="Minimum multiplicative resource factor despite high load.",
    )
    parser.add_argument(
        "--gain-reference",
        default="auto",
        help=(
            "Reference channel gain for log2(1+gain/ref). Use 'auto' for "
            "the global median positive gain, or pass a positive float."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=200000)
    args = parser.parse_args()
    validate_args(parser, args)
    return args

def validate_args(parser, args):
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if not 0.0 <= args.int_frac <= 1.0:
        parser.error("--int-frac must be in [0, 1]")
    if args.history_s < 0.0:
        parser.error("--history-s must be non-negative")
    if args.outcome_horizon_s <= 0.0:
        parser.error("--outcome-horizon-s must be positive")
    if args.min_opportunity_steps < 1:
        parser.error("--min-opportunity-steps must be >= 1")
    if not 0.0 <= args.load_min <= args.load_max <= 1.0:
        parser.error("loads must satisfy 0 <= --load-min <= --load-max <= 1")
    if not 0.0 <= args.min_resource <= 1.0:
        parser.error("--min-resource must be in [0, 1]")
    if not 0.0 <= args.confounding_strength <= 1.0:
        parser.error("--confounding-strength must be in [0, 1]")
    if args.max_attempts < args.num_samples:
        parser.error("--max-attempts must be at least --num-samples")
    if args.gain_reference != "auto":
        try:
            gain_reference = float(args.gain_reference)
        except ValueError:
            parser.error("--gain-reference must be 'auto' or a positive float")
        if gain_reference <= 0.0:
            parser.error("--gain-reference must be positive")

def load_trace(trace_file):
    trace = np.load(trace_file, allow_pickle=True)
    required = [
        "mobility_times_s",
        "trajectory_rsrp_dbm",
        "trajectory_channel_gain_linear",
    ]
    missing = [key for key in required if key not in trace]
    if missing:
        raise KeyError(f"{trace_file} is missing keys: {', '.join(missing)}")

    times_s = np.asarray(trace["mobility_times_s"], dtype=float)
    rss_dbm = np.asarray(trace["trajectory_rsrp_dbm"], dtype=float)
    gain_linear = np.asarray(trace["trajectory_channel_gain_linear"], dtype=float)
    if rss_dbm.ndim != 3 or rss_dbm.shape[2] != 2:
        raise ValueError(f"{trace_file}: trajectory_rsrp_dbm must have shape [time, ue, 2]")
    if gain_linear.shape != rss_dbm.shape:
        raise ValueError(f"{trace_file}: channel gain shape must match RSRP shape")
    if len(times_s) != rss_dbm.shape[0]:
        raise ValueError(f"{trace_file}: mobility_times_s length does not match traces")

    dt = estimate_dt(times_s)
    return {
        "trace_file": str(trace_file),
        "times_s": times_s,
        "dt": dt,
        "rss_dbm": rss_dbm,
        "gain_linear": gain_linear,
    }

def estimate_dt(times_s):
    if len(times_s) < 2:
        raise ValueError("trace needs at least two time samples")
    diffs = np.diff(times_s)
    positive = diffs[diffs > 0.0]
    if len(positive) == 0:
        raise ValueError("trace times must be increasing")
    return float(np.median(positive))

def load_traces(trace_glob):
    trace_files = sorted(Path(path) for path in glob.glob(trace_glob))
    if not trace_files:
        raise FileNotFoundError(f"No trace files matched: {trace_glob}")
    return [load_trace(trace_file) for trace_file in trace_files]

def resolve_gain_reference(traces, gain_reference):
    if gain_reference != "auto":
        return float(gain_reference)

    positives = []
    for trace in traces:
        values = trace["gain_linear"]
        values = values[np.isfinite(values) & (values > 0.0)]
        if len(values) > 0:
            positives.append(values)
    if not positives:
        raise ValueError("Cannot infer gain reference: no positive channel gains")
    return float(np.median(np.concatenate(positives)))

def consecutive_true_counts(mask):
    counts = np.zeros(len(mask), dtype=int)
    current = 0
    for idx, value in enumerate(mask):
        current = current + 1 if value else 0
        counts[idx] = current
    return counts

def candidate_opportunities(trace, history_steps, horizon_steps,
                            min_opportunity_steps):
    rss = trace["rss_dbm"]
    num_times, num_ues, _ = rss.shape
    candidates = []
    start_time = max(history_steps - 1, 0)
    stop_time = num_times - horizon_steps - 1
    if stop_time < start_time:
        return candidates

    for ue_idx in range(num_ues):
        opportunity = rss[:, ue_idx, 0] < rss[:, ue_idx, 1]
        opportunity &= consecutive_true_counts(opportunity) >= min_opportunity_steps
        eligible = np.flatnonzero(opportunity[start_time:stop_time + 1])
        if len(eligible) > 0:
            candidates.append((ue_idx, int(eligible[0] + start_time)))
    return candidates

def resource_factor(load, min_resource):
    return np.clip(1.0 - load, min_resource, 1.0)

def future_effectiveness(gain_linear, load, time_idx, ue_idx, bs_idx,
                         horizon_steps, gain_reference, min_resource):
    start = time_idx + 1
    end = time_idx + horizon_steps + 1
    future_gain = gain_linear[start:end, ue_idx, bs_idx]
    spectral_efficiency = np.log2(1.0 + np.maximum(future_gain, 0.0) / gain_reference)
    return float(resource_factor(load, min_resource) * np.mean(spectral_efficiency))

def sample_loads(rng, args):
    return rng.uniform(args.load_min, args.load_max, size=2)

def build_sample(trace, ue_idx, time_idx, history_steps, horizon_steps,
                 gain_reference, rng, args, is_int):
    rss = trace["rss_dbm"]
    gain = trace["gain_linear"]
    history_start = max(0, time_idx - history_steps + 1)
    rss_window = rss[history_start:time_idx + 1, ue_idx, :]
    if len(rss_window) < history_steps:
        pad = np.repeat(rss_window[:1], history_steps - len(rss_window), axis=0)
        rss_window = np.vstack([pad, rss_window])

    load_bs1, load_bs2 = sample_loads(rng, args)
    rss_diff_window = rss_window[:, 1] - rss_window[:, 0]
    mean_rss_diff_db = float(np.mean(rss_diff_window))

    proposal_probability = np.clip(
        float(sigmoid(
            args.proposal_slope * (mean_rss_diff_db - args.proposal_margin_db)
        )),
        0.1,
        0.9,
    )
    acceptance_probability = np.clip(
        float(sigmoid(
            args.accept_slope
            * args.confounding_strength
            * (args.accept_load_threshold - load_bs2)
        )),
        0.1,
        0.9,
    )
    proposed = rng.random() < proposal_probability
    accepted = proposed and (rng.random() < acceptance_probability)
    if is_int:
        treatment = 0 if rng.random() < 0.5 else 1
    else:
        treatment = int(accepted)
    y0 = future_effectiveness(
        gain,
        load_bs1,
        time_idx,
        ue_idx,
        bs_idx=0,
        horizon_steps=horizon_steps,
        gain_reference=gain_reference,
        min_resource=args.min_resource,
    )
    y1 = future_effectiveness(
        gain,
        load_bs2,
        time_idx,
        ue_idx,
        bs_idx=1,
        horizon_steps=horizon_steps,
        gain_reference=gain_reference,
        min_resource=args.min_resource,
    )

    x1 = np.concatenate([rss_window[:, 0], [load_bs1]])
    x2 = rss_window[:, 1]
    x = np.concatenate([x1, x2])
    return {
        "X": x,
        "X1": x1,
        "X2": x2,
        "U": load_bs2,
        "Y0": y0,
        "Y1": y1,
        "T": treatment,
        "Y_obs": y1 if treatment else y0,
        "is_interventional": int(is_int),
        "proposal_probability": proposal_probability,
        "acceptance_probability": acceptance_probability,
    }

def generate_dataset(traces, args):
    rng = np.random.default_rng(args.seed)
    gain_reference = resolve_gain_reference(traces, args.gain_reference)
    reference_dt = traces[0]["dt"]
    history_steps = max(1, int(np.ceil(args.history_s / reference_dt)))
    horizon_steps = max(1, int(np.ceil(args.outcome_horizon_s / reference_dt)))

    per_trace_candidates = []
    for trace in traces:
        if not np.isclose(trace["dt"], reference_dt):
            raise ValueError(
                "All traces must have the same mobility time step so X has "
                "a fixed length. Found "
                f"{reference_dt:.6g}s and {trace['dt']:.6g}s."
            )
        candidates = candidate_opportunities(
            trace,
            history_steps,
            horizon_steps,
            args.min_opportunity_steps,
        )
        per_trace_candidates.append({
            "trace": trace,
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "candidates": candidates,
        })

    usable = [item for item in per_trace_candidates if item["candidates"]]
    if not usable:
        raise ValueError(
            "No handover opportunities found. Need times where RSS_BS1 < RSS_BS2 "
            "with enough future samples for the outcome horizon."
        )

    samples = []
    attempts = 0
    num_interventional = int(round(args.int_frac * args.num_samples))
    while len(samples) < args.num_samples and attempts < args.max_attempts:
        attempts += 1
        item = usable[rng.integers(len(usable))]
        ue_idx, time_idx = item["candidates"][rng.integers(len(item["candidates"]))]
        if len(samples) < num_interventional:
            is_int = 1
        else:
            is_int = 0
        samples.append(build_sample(
            item["trace"],
            ue_idx,
            time_idx,
            item["history_steps"],
            item["horizon_steps"],
            gain_reference,
            rng,
            args,
            is_int,
        ))

    if len(samples) < args.num_samples:
        raise RuntimeError(
            f"Generated only {len(samples)} samples after {attempts} attempts"
        )

    return samples, gain_reference

def save_dataset(samples, gain_reference, args, traces):
    output_file = None if args.output_file is None else Path(args.output_file)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    trace_files = np.asarray([trace["trace_file"] for trace in traces])

    x1 = np.asarray([sample["X1"] for sample in samples], dtype=float)
    x2 = np.asarray([sample["X2"] for sample in samples], dtype=float)
    x = np.concatenate([x1, x2], axis=1)
    u = np.asarray([sample["U"] for sample in samples], dtype=float)
    y0 = np.asarray([sample["Y0"] for sample in samples], dtype=float)
    y1 = np.asarray([sample["Y1"] for sample in samples], dtype=float)
    t = np.asarray([sample["T"] for sample in samples], dtype=int)
    y_obs = np.asarray([sample["Y_obs"] for sample in samples], dtype=float)
    is_interventional = np.asarray(
        [sample["is_interventional"] for sample in samples],
        dtype=int,
    )
    proposal_probability = np.asarray(
        [sample["proposal_probability"] for sample in samples],
        dtype=float,
    )
    acceptance_probability = np.asarray(
        [sample["acceptance_probability"] for sample in samples],
        dtype=float,
    )
    dict= {
        'X': x,
        'T': t,
        'is_interventional': is_interventional,
        'Y_obs': y_obs,
        'Y0': y0,
        'Y1': y1,
        'p_prop': proposal_probability,
        'p_accept': acceptance_probability,
        'confounding_strength': np.full(len(samples), args.confounding_strength, dtype=float),
        'accept_slope_base': np.full(len(samples), args.accept_slope, dtype=float),
        'accept_slope_effective': np.full(
            len(samples),
            args.accept_slope * args.confounding_strength,
            dtype=float,
        ),
    }
    if output_file is not None:
        np.save(output_file, dict, allow_pickle=True)
    return dict

def main():
    args = parse_args()
    traces = load_traces(args.trace_glob)
    samples, gain_reference = generate_dataset(traces, args)
    save_dataset(samples, gain_reference, args, traces)

    treatment_rate = np.mean([sample["T"] for sample in samples])
    interventional_rate = np.mean([sample["is_interventional"] for sample in samples])
    print(f"Loaded {len(traces)} trace file(s)")
    print(f"Saved {len(samples)} handover samples to {args.output_file}")
    print(f"Treatment rate: {treatment_rate:.3f}")
    print(f"Interventional sample rate: {interventional_rate:.3f}")
    print(f"Confounding strength: {args.confounding_strength:.3f}")
    print(f"Effective accept slope: {args.accept_slope * args.confounding_strength:.3f}")
    print(f"Gain reference: {gain_reference:.3e}")

if __name__ == "__main__":
    main()
