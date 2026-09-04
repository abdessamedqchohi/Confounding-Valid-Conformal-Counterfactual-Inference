from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class ChannelConfig:
    frequency_hz: float = 3.5e9
    path_max_depth: int = 2
    path_num_samples: int = 20000
    path_max_num_paths_per_src: int = int(1e6)
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    normalize_cfr: bool = False
    seed: int = 1

def to_numpy(x):
    if isinstance(x, (tuple, list)):
        return tuple(to_numpy(item) for item in x)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

def to_complex(x):
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError("Expected tuple/list to contain real and imaginary parts")
        return np.asarray(x[0]) + 1j * np.asarray(x[1])
    return np.asarray(x)

def bs_positions_from_xy(bs_xy, height):
    bs_xy = np.asarray(bs_xy, dtype=float)
    if bs_xy.ndim != 2 or bs_xy.shape[1] != 2:
        raise ValueError("bs_xy must have shape [num_bs, 2]")
    z = np.full((bs_xy.shape[0], 1), float(height), dtype=float)
    return np.column_stack([bs_xy, z])

def reduce_cfr_to_links(h_freq):
    h_freq = to_complex(h_freq)

    if h_freq.ndim == 6:
        
        h = np.sum(h_freq, axis=(1, 3))
        return np.moveaxis(h, 2, 0)

    if h_freq.ndim == 7:
        
        h = np.sum(h_freq, axis=(0, 2, 4))
        return np.moveaxis(h, 2, 0)

    if h_freq.ndim == 5:
        
        h = np.sum(h_freq, axis=(1, 3))
        return h[None, ...]

    raise ValueError(f"Unexpected CFR shape: {h_freq.shape}")

class SionnaChannelSimulator:

    def __init__(self, bs_positions, config=None,
                 scene_name="simple_street_canyon"):
        self.bs_positions = np.asarray(bs_positions, dtype=float)
        self.config = config or ChannelConfig()
        self.scene_name = scene_name
        self.scene = None
        self.path_solver = None
        self.users = []
        self._receiver_class = None

    def reset(self, initial_ue_positions):
        load_scene, Transmitter, Receiver, PlanarArray, PathSolver, rt_scene = (
            self._import_sionna_rt()
        )

        self.scene = load_scene(self._resolve_scene_path(rt_scene))
        self._configure_scene(PlanarArray)
        self._add_base_stations(Transmitter)
        self._receiver_class = Receiver
        self.users = self._add_users(Receiver, initial_ue_positions)
        self.path_solver = PathSolver() if PathSolver is not None else None

    def simulate_static_position_segment(self, ue_positions_segment,
                                         frequencies_hz, first_step=False,
                                         time_s=0.0):
        del time_s
        if self.scene is None:
            raise RuntimeError("Call reset() before simulate_static_position_segment()")

        ue_positions_segment = np.asarray(ue_positions_segment, dtype=float)
        if ue_positions_segment.ndim != 3 or ue_positions_segment.shape[2] != 3:
            raise ValueError(
                "ue_positions_segment must have shape [num_steps, num_ues, 3]"
            )

        num_steps, num_ues, _ = ue_positions_segment.shape
        flattened_positions = ue_positions_segment.reshape(-1, 3)
        self._ensure_receiver_pool(flattened_positions)
        self._update_user_positions(flattened_positions)

        paths = self._compute_paths(first_step=first_step)
        h_freq = self._get_cfr_numpy(
            paths,
            frequencies_hz=frequencies_hz,
            num_time_steps=1,
            sampling_frequency=1.0,
        )
        h_links = reduce_cfr_to_links(h_freq)

        actual_num_receivers = flattened_positions.shape[0]
        if h_links.shape[0] == 1 and h_links.shape[1] >= actual_num_receivers:
            trimmed = h_links[0, :actual_num_receivers]
            h_links = trimmed.reshape(num_steps, num_ues, *trimmed.shape[1:])
        elif h_links.shape[0] >= actual_num_receivers:
            trimmed = h_links[:actual_num_receivers]
            h_links = trimmed.reshape(num_steps, num_ues, *trimmed.shape[1:])

        return {"channel_frequency_response": h_links}

    def _import_sionna_rt(self):
        from sionna.rt import (
            load_scene,
            Transmitter,
            Receiver,
            PlanarArray,
        )
        try:
            from sionna.rt import PathSolver
        except ImportError:
            PathSolver = None

        import sionna.rt.scene as rt_scene
        return load_scene, Transmitter, Receiver, PlanarArray, PathSolver, rt_scene

    def _resolve_scene_path(self, rt_scene):
        scene_path = Path(self.scene_name)
        if scene_path.exists():
            return str(scene_path)
        return getattr(rt_scene, self.scene_name)

    def _configure_scene(self, PlanarArray):
        self.scene.frequency = self.config.frequency_hz
        self.scene.synthetic_array = True
        self.scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

    def _add_base_stations(self, Transmitter):
        for bs_idx, position in enumerate(self.bs_positions):
            self.scene.add(Transmitter(
                name=f"bs_{bs_idx}",
                position=position.tolist(),
            ))

    def _add_users(self, Receiver, positions):
        users = []
        for ue_idx, position in enumerate(positions):
            ue = Receiver(name=f"ue_{ue_idx}", position=position.tolist())
            self.scene.add(ue)
            users.append(ue)
        return users

    def _ensure_receiver_pool(self, positions):
        if len(self.users) == len(positions):
            return

        for ue in self.users:
            self.scene.remove(ue.name)
        self.users = self._add_users(self._receiver_class, positions)

    def _update_user_positions(self, positions):
        for ue, position in zip(self.users, positions):
            ue.position = np.asarray(position, dtype=float).tolist()

    def _compute_paths(self, first_step=False):
        cfg = self.config
        if self.path_solver is not None:
            return self.path_solver(
                scene=self.scene,
                max_depth=cfg.path_max_depth,
                max_num_paths_per_src=cfg.path_max_num_paths_per_src,
                samples_per_src=cfg.path_num_samples,
                synthetic_array=True,
                los=cfg.los,
                specular_reflection=cfg.specular_reflection,
                diffuse_reflection=cfg.diffuse_reflection,
                refraction=cfg.refraction,
                diffraction=cfg.diffraction,
                seed=cfg.seed,
            )

        return self.scene.compute_paths(
            max_depth=cfg.path_max_depth,
            num_samples=cfg.path_num_samples,
            los=cfg.los,
            reflection=cfg.specular_reflection,
            diffraction=cfg.diffraction,
            scattering=cfg.diffuse_reflection,
            check_scene=first_step,
        )

    def _get_cfr_numpy(self, paths, frequencies_hz,
                       num_time_steps, sampling_frequency):
        h_freq = paths.cfr(
            frequencies=np.asarray(frequencies_hz, dtype=float),
            sampling_frequency=sampling_frequency,
            num_time_steps=num_time_steps,
            normalize_delays=False,
            normalize=self.config.normalize_cfr,
            out_type="numpy",
        )
        return to_numpy(h_freq)
