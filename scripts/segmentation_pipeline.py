from __future__ import annotations

import argparse
import colorsys
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PREPROCESSOR_CHOICES = ["vip", "r3m", "liv", "clip", "vc1", "dinov2", "resnet"]
SEGMENTATION_ALGORITHM_CHOICES = [
    "uvd",
    "goal_distance",
    "window_mean",
    "kernel_cpd",
    "hmm",
    "hsmm",
]
SMOOTH_METHOD_CHOICES = ["kernel", "savgol", "none"]


@dataclass(frozen=True)
class CommonConfig:
    preprocessor_name: str = "vip"
    device: str = "cuda"
    embed_batch_size: int = 64
    embedding_cache: str | None = None
    no_embedding_cache: bool = False
    segmentation_algorithm: str = "kernel_cpd"


@dataclass(frozen=True)
class UvdConfig:
    smooth_method: str = "kernel"
    normalize_curve: bool = False
    window_length: int | None = None
    min_interval: int = 1
    gamma: float = 0.02
    extrema_prominence: float | None = 0.4


@dataclass(frozen=True)
class GoalDistanceConfig:
    smooth_kernel: int = 5
    prominence: float = 0.1
    min_segment_len: int = 10


@dataclass(frozen=True)
class WindowMeanConfig:
    window: int = 30
    prominence: float = 0.1
    min_segment_len: int = 10


@dataclass(frozen=True)
class KernelCpdConfig:
    window: int = 30
    gamma: float = 0.01
    prominence: float = 0.4
    min_segment_len: int = 10


@dataclass(frozen=True)
class HmmConfig:
    num_states: int = 4
    stay_bias: float = 2.0
    min_segment_len: int = 10


@dataclass(frozen=True)
class HsmmConfig:
    num_states: int = 4
    switch_cost: float = 5.0
    min_duration: int = 10
    max_duration: int | None = None


@dataclass(frozen=True)
class SegmentationConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    uvd: UvdConfig = field(default_factory=UvdConfig)
    goal_distance: GoalDistanceConfig = field(default_factory=GoalDistanceConfig)
    window_mean: WindowMeanConfig = field(default_factory=WindowMeanConfig)
    kernel_cpd: KernelCpdConfig = field(default_factory=KernelCpdConfig)
    hmm: HmmConfig = field(default_factory=HmmConfig)
    hsmm: HsmmConfig = field(default_factory=HsmmConfig)


SEGMENTATION_DEFAULTS = SegmentationConfig()


def build_cli_defaults(overrides: dict[str, object] | None = None) -> dict[str, object]:
    defaults = {
        "preprocessor_name": SEGMENTATION_DEFAULTS.common.preprocessor_name,
        "device": SEGMENTATION_DEFAULTS.common.device,
        "embed_batch_size": SEGMENTATION_DEFAULTS.common.embed_batch_size,
        "embedding_cache": SEGMENTATION_DEFAULTS.common.embedding_cache,
        "segmentation_algorithm": SEGMENTATION_DEFAULTS.common.segmentation_algorithm,
        "uvd_smooth_method": SEGMENTATION_DEFAULTS.uvd.smooth_method,
        "uvd_window_length": SEGMENTATION_DEFAULTS.uvd.window_length,
        "uvd_min_interval": SEGMENTATION_DEFAULTS.uvd.min_interval,
        "uvd_gamma": SEGMENTATION_DEFAULTS.uvd.gamma,
        "uvd_extrema_prominence": SEGMENTATION_DEFAULTS.uvd.extrema_prominence,
        "goal_distance_smooth_kernel": SEGMENTATION_DEFAULTS.goal_distance.smooth_kernel,
        "goal_distance_prominence": SEGMENTATION_DEFAULTS.goal_distance.prominence,
        "goal_distance_min_segment_len": SEGMENTATION_DEFAULTS.goal_distance.min_segment_len,
        "window_mean_window": SEGMENTATION_DEFAULTS.window_mean.window,
        "window_mean_prominence": SEGMENTATION_DEFAULTS.window_mean.prominence,
        "window_mean_min_segment_len": SEGMENTATION_DEFAULTS.window_mean.min_segment_len,
        "kernel_cpd_window": SEGMENTATION_DEFAULTS.kernel_cpd.window,
        "kernel_cpd_gamma": SEGMENTATION_DEFAULTS.kernel_cpd.gamma,
        "kernel_cpd_prominence": SEGMENTATION_DEFAULTS.kernel_cpd.prominence,
        "kernel_cpd_min_segment_len": SEGMENTATION_DEFAULTS.kernel_cpd.min_segment_len,
        "hmm_num_states": SEGMENTATION_DEFAULTS.hmm.num_states,
        "hmm_stay_bias": SEGMENTATION_DEFAULTS.hmm.stay_bias,
        "hmm_min_segment_len": SEGMENTATION_DEFAULTS.hmm.min_segment_len,
        "hsmm_num_states": SEGMENTATION_DEFAULTS.hsmm.num_states,
        "hsmm_switch_cost": SEGMENTATION_DEFAULTS.hsmm.switch_cost,
        "hsmm_min_duration": SEGMENTATION_DEFAULTS.hsmm.min_duration,
        "hsmm_max_duration": SEGMENTATION_DEFAULTS.hsmm.max_duration,
    }
    if overrides:
        defaults.update(overrides)
    return defaults


def segmentation_config_from_args(args) -> SegmentationConfig:
    return SegmentationConfig(
        common=CommonConfig(
            preprocessor_name=args.preprocessor_name,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
            embedding_cache=args.embedding_cache,
            no_embedding_cache=args.no_embedding_cache,
            segmentation_algorithm=args.segmentation_algorithm,
        ),
        uvd=UvdConfig(
            smooth_method=args.uvd_smooth_method,
            normalize_curve=args.uvd_normalize_curve,
            window_length=args.uvd_window_length,
            min_interval=args.uvd_min_interval,
            gamma=args.uvd_gamma,
            extrema_prominence=args.uvd_extrema_prominence,
        ),
        goal_distance=GoalDistanceConfig(
            smooth_kernel=args.goal_distance_smooth_kernel,
            prominence=args.goal_distance_prominence,
            min_segment_len=args.goal_distance_min_segment_len,
        ),
        window_mean=WindowMeanConfig(
            window=args.window_mean_window,
            prominence=args.window_mean_prominence,
            min_segment_len=args.window_mean_min_segment_len,
        ),
        kernel_cpd=KernelCpdConfig(
            window=args.kernel_cpd_window,
            gamma=args.kernel_cpd_gamma,
            prominence=args.kernel_cpd_prominence,
            min_segment_len=args.kernel_cpd_min_segment_len,
        ),
        hmm=HmmConfig(
            num_states=args.hmm_num_states,
            stay_bias=args.hmm_stay_bias,
            min_segment_len=args.hmm_min_segment_len,
        ),
        hsmm=HsmmConfig(
            num_states=args.hsmm_num_states,
            switch_cost=args.hsmm_switch_cost,
            min_duration=args.hsmm_min_duration,
            max_duration=args.hsmm_max_duration,
        ),
    )


def algorithm_params_dict(config: SegmentationConfig) -> dict[str, object]:
    if config.common.segmentation_algorithm == "uvd":
        params = asdict(config.uvd)
        if params["smooth_method"] == "none":
            params["smooth_method"] = None
        return params
    if config.common.segmentation_algorithm == "goal_distance":
        return asdict(config.goal_distance)
    if config.common.segmentation_algorithm == "window_mean":
        return asdict(config.window_mean)
    if config.common.segmentation_algorithm == "kernel_cpd":
        return asdict(config.kernel_cpd)
    if config.common.segmentation_algorithm == "hmm":
        return asdict(config.hmm)
    return asdict(config.hsmm)


def add_common_cli_args(
    parser: argparse.ArgumentParser,
    defaults: dict[str, object] | None = None,
) -> argparse.ArgumentParser:
    defaults = build_cli_defaults(defaults)

    parser.add_argument("video_file", help="Path to the input video.")

    common = parser.add_argument_group("Common options")
    common.add_argument(
        "--preprocessor_name",
        default=defaults["preprocessor_name"],
        choices=PREPROCESSOR_CHOICES,
        help="Frozen visual encoder used for segmentation.",
    )
    common.add_argument(
        "--device",
        default=defaults["device"],
        help='Inference device. Defaults to "cuda".',
    )
    common.add_argument(
        "--embed_batch_size",
        type=int,
        default=defaults["embed_batch_size"],
        help="Number of frames per embedding batch.",
    )
    common.add_argument(
        "--embedding_cache",
        default=defaults["embedding_cache"],
        help=(
            "Optional path to a .npy embedding cache. "
            "Defaults to a file under the sibling segments/ directory."
        ),
    )
    common.add_argument(
        "--no_embedding_cache",
        action="store_true",
        help="Disable loading/saving cached frame embeddings.",
    )
    common.add_argument(
        "--segmentation_algorithm",
        default=defaults["segmentation_algorithm"],
        choices=SEGMENTATION_ALGORITHM_CHOICES,
        help=(
            "Boundary generation algorithm. "
            "'uvd' uses the repo's decomposition logic; "
            "'goal_distance' uses smoothed final-frame-distance peaks/valleys; "
            "'window_mean' uses changes in the embedding sequence itself; "
            "'kernel_cpd', 'hmm', and 'hsmm' are goal-free embedding backends."
        ),
    )

    uvd = parser.add_argument_group("UVD options")
    uvd.add_argument(
        "--uvd_smooth_method",
        dest="uvd_smooth_method",
        default=defaults["uvd_smooth_method"],
        choices=SMOOTH_METHOD_CHOICES,
        help="Curve smoothing method used before picking milestone candidates.",
    )
    uvd.add_argument(
        "--uvd_normalize_curve",
        dest="uvd_normalize_curve",
        action="store_true",
        help="Enable distance-curve normalization before decomposition.",
    )
    uvd.add_argument(
        "--uvd_window_length",
        dest="uvd_window_length",
        type=int,
        default=defaults["uvd_window_length"],
        help="Optional backward window length for milestone search.",
    )
    uvd.add_argument(
        "--uvd_min_interval",
        dest="uvd_min_interval",
        type=int,
        default=defaults["uvd_min_interval"],
        help="Minimum distance between neighboring segmentation milestones in frames.",
    )
    uvd.add_argument(
        "--uvd_gamma",
        dest="uvd_gamma",
        type=float,
        default=defaults["uvd_gamma"],
        help="Kernel smoothing strength for smooth_method=kernel.",
    )
    uvd.add_argument(
        "--uvd_extrema_prominence",
        dest="uvd_extrema_prominence",
        type=float,
        default=defaults["uvd_extrema_prominence"],
        help="Minimum prominence for UVD extrema candidates.",
    )

    goal = parser.add_argument_group("Goal-distance options")
    goal.add_argument(
        "--goal_distance_smooth_kernel",
        dest="goal_distance_smooth_kernel",
        type=int,
        default=defaults["goal_distance_smooth_kernel"],
        help="Moving-average kernel size for goal_distance segmentation.",
    )
    goal.add_argument(
        "--goal_distance_prominence",
        dest="goal_distance_prominence",
        type=float,
        default=defaults["goal_distance_prominence"],
        help="Peak/valley prominence for goal_distance segmentation.",
    )
    goal.add_argument(
        "--goal_distance_min_segment_len",
        dest="goal_distance_min_segment_len",
        type=int,
        default=defaults["goal_distance_min_segment_len"],
        help="Minimum segment length in frames for goal_distance segmentation.",
    )

    window_mean = parser.add_argument_group("Window-mean options")
    window_mean.add_argument(
        "--window_mean_window",
        dest="window_mean_window",
        type=int,
        default=defaults["window_mean_window"],
        help="Window size used by window_mean segmentation.",
    )
    window_mean.add_argument(
        "--window_mean_prominence",
        dest="window_mean_prominence",
        type=float,
        default=defaults["window_mean_prominence"],
        help="Peak prominence for window_mean segmentation.",
    )
    window_mean.add_argument(
        "--window_mean_min_segment_len",
        dest="window_mean_min_segment_len",
        type=int,
        default=defaults["window_mean_min_segment_len"],
        help="Minimum segment length in frames for window_mean segmentation.",
    )

    kernel = parser.add_argument_group("Kernel-CPD options")
    kernel.add_argument(
        "--kernel_cpd_window",
        dest="kernel_cpd_window",
        type=int,
        default=defaults["kernel_cpd_window"],
        help="Window size used by kernel_cpd segmentation.",
    )
    kernel.add_argument(
        "--kernel_cpd_gamma",
        dest="kernel_cpd_gamma",
        type=float,
        default=defaults["kernel_cpd_gamma"],
        help="RBF gamma used by kernel_cpd segmentation.",
    )
    kernel.add_argument(
        "--kernel_cpd_prominence",
        dest="kernel_cpd_prominence",
        type=float,
        default=defaults["kernel_cpd_prominence"],
        help="Peak prominence for kernel_cpd segmentation.",
    )
    kernel.add_argument(
        "--kernel_cpd_min_segment_len",
        dest="kernel_cpd_min_segment_len",
        type=int,
        default=defaults["kernel_cpd_min_segment_len"],
        help="Minimum segment length in frames for kernel_cpd segmentation.",
    )

    hmm = parser.add_argument_group("HMM options")
    hmm.add_argument(
        "--hmm_num_states",
        dest="hmm_num_states",
        type=int,
        default=defaults["hmm_num_states"],
        help="Number of latent states for hmm segmentation.",
    )
    hmm.add_argument(
        "--hmm_stay_bias",
        dest="hmm_stay_bias",
        type=float,
        default=defaults["hmm_stay_bias"],
        help="Switch penalty for hmm segmentation.",
    )
    hmm.add_argument(
        "--hmm_min_segment_len",
        dest="hmm_min_segment_len",
        type=int,
        default=defaults["hmm_min_segment_len"],
        help="Minimum segment length in frames for hmm segmentation.",
    )

    hsmm = parser.add_argument_group("HSMM options")
    hsmm.add_argument(
        "--hsmm_num_states",
        dest="hsmm_num_states",
        type=int,
        default=defaults["hsmm_num_states"],
        help="Number of latent states for hsmm segmentation.",
    )
    hsmm.add_argument(
        "--hsmm_switch_cost",
        dest="hsmm_switch_cost",
        type=float,
        default=defaults["hsmm_switch_cost"],
        help="State switch cost for hsmm segmentation.",
    )
    hsmm.add_argument(
        "--hsmm_min_duration",
        dest="hsmm_min_duration",
        type=int,
        default=defaults["hsmm_min_duration"],
        help="Minimum duration in frames for hsmm segmentation.",
    )
    hsmm.add_argument(
        "--hsmm_max_duration",
        dest="hsmm_max_duration",
        type=int,
        default=defaults["hsmm_max_duration"],
        help="Optional maximum duration in frames for hsmm segmentation.",
    )
    return parser


def build_embedding_cache_path(
    video_file: str,
    preprocessor_name: str,
    embedding_cache: str | None = None,
) -> str:
    if embedding_cache is not None:
        return os.path.expandvars(os.path.expanduser(embedding_cache))
    video_file = os.path.expandvars(os.path.expanduser(video_file))
    video_dir = os.path.dirname(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    segments_dir = os.path.join(video_dir, "segments")
    return os.path.join(segments_dir, f"{video_name}.embeddings_{preprocessor_name}.npy")


def palette(n: int) -> list[tuple[int, int, int]]:
    colors = []
    for idx in range(max(n, 1)):
        hue = idx / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        colors.append(tuple(int(channel * 255) for channel in rgb))
    return colors


def compute_embeddings_in_batches(preprocessor, frames: np.ndarray, batch_size: int) -> np.ndarray:
    batch_size = max(1, int(batch_size))
    outputs = []
    total = len(frames)
    batch_iter = range(0, total, batch_size)
    if tqdm is not None:
        batch_iter = tqdm(
            batch_iter,
            total=(total + batch_size - 1) // batch_size,
            desc="Embedding batches",
            unit="batch",
        )
    for start in batch_iter:
        end = min(start + batch_size, total)
        outputs.append(preprocessor.process(frames[start:end], return_numpy=True))
    return np.concatenate(outputs, axis=0)


def load_or_compute_embeddings(
    video_file: str,
    frames: np.ndarray,
    preprocessor_name: str,
    device: str,
    batch_size: int,
    embedding_cache: str | None = None,
    no_embedding_cache: bool = False,
) -> tuple[np.ndarray, str | None, bool]:
    cache_path = None
    if not no_embedding_cache:
        cache_path = build_embedding_cache_path(
            video_file=video_file,
            preprocessor_name=preprocessor_name,
            embedding_cache=embedding_cache,
        )
        if os.path.exists(cache_path):
            rep = np.load(cache_path)
            if rep.ndim != 2 or rep.shape[0] != len(frames):
                raise ValueError(
                    f"Embedding cache shape {rep.shape} does not match video length {len(frames)}"
                )
            return rep, cache_path, True

    import uvd

    preprocessor = uvd.get_preprocessor(preprocessor_name, device=device)
    rep = compute_embeddings_in_batches(
        preprocessor,
        frames,
        batch_size=batch_size,
    )
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, rep)
    return rep, cache_path, False


def compute_distance_curve(
    embeddings: np.ndarray,
    normalize_curve: bool,
) -> np.ndarray:
    goal_embedding = embeddings[-1]
    distances = np.linalg.norm(embeddings - goal_embedding, axis=1)
    if normalize_curve:
        distances = distances / max(float(distances[0]), 1e-8)
    return distances


def smooth_distance_curve(
    distances: np.ndarray,
    smooth_method: str | None,
    gamma: float,
) -> np.ndarray:
    if smooth_method == "kernel":
        from uvd.decomp.kernel_reg import KernelRegression

        x = np.arange(len(distances))
        kr = KernelRegression(kernel="rbf", gamma=gamma)
        kr.fit(x.reshape(-1, 1), distances)
        return kr.predict(x.reshape(-1, 1))
    if smooth_method == "savgol":
        from scipy.signal import savgol_filter

        window = min(len(distances) if len(distances) % 2 == 1 else len(distances) - 1, 85)
        window = max(window, 3)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            return savgol_filter(
                distances,
                window_length=window,
                polyorder=min(2, window - 1),
                mode="nearest",
            )
    return distances.copy()


def compute_segment_goal_curve(
    embeddings: np.ndarray,
    milestone_indices: np.ndarray,
    normalize_curve: bool,
    smooth_method: str | None,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_curve = np.zeros(len(embeddings), dtype=np.float32)
    smooth_curve = np.zeros(len(embeddings), dtype=np.float32)
    starts = (
        np.concatenate([[0], np.array(milestone_indices[:-1], dtype=np.int64) + 1])
        if len(milestone_indices) > 0
        else np.array([0], dtype=np.int64)
    )
    ends = (
        np.array(milestone_indices, dtype=np.int64)
        if len(milestone_indices) > 0
        else np.array([len(embeddings) - 1], dtype=np.int64)
    )
    for start, end in zip(starts, ends):
        goal_embedding = embeddings[end]
        distances = np.linalg.norm(embeddings[start : end + 1] - goal_embedding, axis=1)
        if normalize_curve:
            distances = distances / max(float(distances[0]), 1e-8)
        raw_curve[start : end + 1] = distances
        smooth_curve[start : end + 1] = smooth_distance_curve(
            distances,
            smooth_method=smooth_method,
            gamma=gamma,
        )
    return raw_curve, smooth_curve


def assign_segments(num_frames: int, milestone_indices: Iterable[int]) -> np.ndarray:
    milestones = np.array(sorted(set(int(i) for i in milestone_indices)), dtype=np.int64)
    if milestones.size == 0:
        milestones = np.array([num_frames - 1], dtype=np.int64)
    milestones = np.clip(milestones, 0, num_frames - 1)
    starts = np.concatenate([[0], milestones[:-1] + 1])
    segment_ids = np.empty(num_frames, dtype=np.int64)
    for seg_idx, (start, end) in enumerate(zip(starts, milestones)):
        segment_ids[start : end + 1] = seg_idx
    if milestones[-1] < num_frames - 1:
        segment_ids[milestones[-1] + 1 :] = len(milestones) - 1
    return segment_ids


def segment_goal_distance(
    embeddings: np.ndarray,
    smooth_kernel: int,
    prominence: float,
    min_segment_len: int,
) -> np.ndarray:
    from scipy.signal import find_peaks

    goal_embedding = embeddings[-1]
    distances = np.linalg.norm(embeddings - goal_embedding, axis=1)
    if len(distances) == 0:
        return np.array([], dtype=np.int64)
    distances = distances / max(float(distances[0]), 1e-8)

    kernel = max(1, int(smooth_kernel))
    kernel_arr = np.ones(kernel, dtype=np.float32) / kernel
    distances_smooth = np.convolve(distances, kernel_arr, mode="same")

    peaks, _ = find_peaks(distances_smooth, prominence=prominence)
    valleys, _ = find_peaks(-distances_smooth, prominence=prominence)
    idxs = sorted(peaks.tolist() + valleys.tolist())

    filtered_idxs = []
    for idx in idxs:
        if filtered_idxs and idx - filtered_idxs[-1] == 1:
            continue
        filtered_idxs.append(idx)

    boundaries = [0] + filtered_idxs + [len(distances)]
    filtered_boundaries = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - filtered_boundaries[-1] >= min_segment_len:
            filtered_boundaries.append(boundary)
        elif boundary == len(distances):
            filtered_boundaries[-1] = boundary

    return np.array([boundary - 1 for boundary in filtered_boundaries[1:]], dtype=np.int64)


def segment_window_mean(
    embeddings: np.ndarray,
    window: int,
    prominence: float,
    min_segment_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import find_peaks

    traj_length = len(embeddings)
    if traj_length == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    window = max(1, int(window))
    scores = np.zeros(traj_length, dtype=np.float32)
    for idx in range(window, traj_length - window):
        left = embeddings[idx - window : idx].mean(axis=0)
        right = embeddings[idx : idx + window].mean(axis=0)
        scores[idx] = np.linalg.norm(left - right)

    peaks, _ = find_peaks(scores, prominence=prominence)
    filtered_idxs = []
    for idx in peaks.tolist():
        if filtered_idxs and idx - filtered_idxs[-1] < min_segment_len:
            if scores[idx] > scores[filtered_idxs[-1]]:
                filtered_idxs[-1] = idx
            continue
        filtered_idxs.append(idx)

    boundaries = [0] + filtered_idxs + [traj_length]
    filtered_boundaries = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - filtered_boundaries[-1] >= min_segment_len:
            filtered_boundaries.append(boundary)
        elif boundary == traj_length:
            filtered_boundaries[-1] = boundary

    return (
        np.array([boundary - 1 for boundary in filtered_boundaries[1:]], dtype=np.int64),
        scores,
    )


def _rbf_kernel_matrix(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    dist_sq = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return np.exp(-gamma * dist_sq)


def segment_kernel_cpd(
    embeddings: np.ndarray,
    window: int,
    gamma: float,
    prominence: float,
    min_segment_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import find_peaks

    traj_length = len(embeddings)
    if traj_length == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    window = max(2, int(window))
    gamma = max(float(gamma), 1e-8)
    scores = np.zeros(traj_length, dtype=np.float32)
    for idx in range(window, traj_length - window):
        left = embeddings[idx - window : idx]
        right = embeddings[idx : idx + window]
        k_xx = _rbf_kernel_matrix(left, left, gamma)
        k_yy = _rbf_kernel_matrix(right, right, gamma)
        k_xy = _rbf_kernel_matrix(left, right, gamma)
        scores[idx] = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())

    peaks, _ = find_peaks(scores, prominence=prominence)
    filtered_idxs = []
    for idx in peaks.tolist():
        if filtered_idxs and idx - filtered_idxs[-1] < min_segment_len:
            if scores[idx] > scores[filtered_idxs[-1]]:
                filtered_idxs[-1] = idx
            continue
        filtered_idxs.append(idx)

    boundaries = [0] + filtered_idxs + [traj_length]
    filtered_boundaries = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - filtered_boundaries[-1] >= min_segment_len:
            filtered_boundaries.append(boundary)
        elif boundary == traj_length:
            filtered_boundaries[-1] = boundary

    return (
        np.array([boundary - 1 for boundary in filtered_boundaries[1:]], dtype=np.int64),
        scores,
    )


def _simple_kmeans(
    embeddings: np.ndarray,
    num_states: int,
    num_iters: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    traj_length = len(embeddings)
    num_states = max(1, min(int(num_states), traj_length))
    init_idx = np.linspace(0, traj_length - 1, num_states, dtype=int)
    centers = embeddings[init_idx].copy()
    labels = np.zeros(traj_length, dtype=np.int64)
    for _ in range(num_iters):
        dist_sq = ((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist_sq.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for state in range(num_states):
            mask = labels == state
            if np.any(mask):
                centers[state] = embeddings[mask].mean(axis=0)
    return centers, labels


def _estimate_emission_costs(embeddings: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def segment_hmm(
    embeddings: np.ndarray,
    num_states: int,
    stay_bias: float,
    min_segment_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers, _ = _simple_kmeans(embeddings, num_states=num_states)
    emission_cost = _estimate_emission_costs(embeddings, centers)
    num_states = centers.shape[0]
    stay_bias = max(float(stay_bias), 0.0)
    switch_cost = np.full((num_states, num_states), stay_bias, dtype=np.float32)
    np.fill_diagonal(switch_cost, 0.0)

    traj_length = len(embeddings)
    dp = np.zeros((traj_length, num_states), dtype=np.float32)
    prev = np.zeros((traj_length, num_states), dtype=np.int64)
    dp[0] = emission_cost[0]
    for t in range(1, traj_length):
        candidate = dp[t - 1][:, None] + switch_cost
        prev[t] = candidate.argmin(axis=0)
        dp[t] = emission_cost[t] + candidate.min(axis=0)

    states = np.zeros(traj_length, dtype=np.int64)
    states[-1] = dp[-1].argmin()
    for t in range(traj_length - 1, 0, -1):
        states[t - 1] = prev[t, states[t]]

    milestones = np.where(np.diff(states) != 0)[0]
    if milestones.size > 0:
        filtered = [int(milestones[0])]
        for idx in milestones[1:]:
            if idx - filtered[-1] >= min_segment_len:
                filtered.append(int(idx))
        milestones = np.array(filtered, dtype=np.int64)
    milestones = np.concatenate([milestones, [traj_length - 1]])
    state_scores = np.zeros(traj_length, dtype=np.float32)
    state_scores[1:] = (states[1:] != states[:-1]).astype(np.float32)
    return milestones, state_scores


def _segment_sse_cost(
    cumulative_sum: np.ndarray,
    cumulative_sq_norm: np.ndarray,
    start: int,
    end: int,
    center: np.ndarray,
) -> float:
    seg_sum = cumulative_sum[end] - cumulative_sum[start]
    seg_sq = cumulative_sq_norm[end] - cumulative_sq_norm[start]
    length = end - start
    return float(seg_sq - 2.0 * center.dot(seg_sum) + length * center.dot(center))


def segment_hsmm(
    embeddings: np.ndarray,
    num_states: int,
    switch_cost: float,
    min_duration: int,
    max_duration: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    traj_length = len(embeddings)
    centers, _ = _simple_kmeans(embeddings, num_states=num_states)
    num_states = centers.shape[0]
    min_duration = max(1, int(min_duration))
    max_duration = traj_length if max_duration is None else max(min_duration, int(max_duration))

    cumulative_sum = np.zeros((traj_length + 1, embeddings.shape[1]), dtype=np.float64)
    cumulative_sum[1:] = np.cumsum(embeddings, axis=0)
    cumulative_sq_norm = np.zeros(traj_length + 1, dtype=np.float64)
    cumulative_sq_norm[1:] = np.cumsum((embeddings * embeddings).sum(axis=1), axis=0)

    dp = np.full((traj_length + 1, num_states), np.inf, dtype=np.float64)
    prev_time = np.full((traj_length + 1, num_states), -1, dtype=np.int64)
    prev_state = np.full((traj_length + 1, num_states), -1, dtype=np.int64)

    for end in range(min_duration, traj_length + 1):
        start_min = max(0, end - max_duration)
        start_max = end - min_duration
        for start in range(start_min, start_max + 1):
            for state in range(num_states):
                seg_cost = _segment_sse_cost(
                    cumulative_sum,
                    cumulative_sq_norm,
                    start,
                    end,
                    centers[state],
                )
                if start == 0:
                    total_cost = seg_cost
                    from_state = -1
                else:
                    prev_costs = dp[start] + switch_cost
                    prev_costs[state] = dp[start, state]
                    from_state = int(prev_costs.argmin())
                    total_cost = seg_cost + prev_costs[from_state]
                if total_cost < dp[end, state]:
                    dp[end, state] = total_cost
                    prev_time[end, state] = start
                    prev_state[end, state] = from_state

    states = []
    boundaries = []
    cur_end = traj_length
    cur_state = int(dp[cur_end].argmin())
    while cur_end > 0 and cur_state >= 0:
        start = int(prev_time[cur_end, cur_state])
        if start < 0:
            break
        states.append(cur_state)
        boundaries.append(cur_end - 1)
        next_state = int(prev_state[cur_end, cur_state])
        cur_end = start
        cur_state = next_state
    boundaries = np.array(list(reversed(boundaries)), dtype=np.int64)
    if boundaries.size == 0 or boundaries[-1] != traj_length - 1:
        boundaries = np.concatenate([boundaries, [traj_length - 1]])

    scores = np.zeros(traj_length, dtype=np.float32)
    for boundary in boundaries[:-1]:
        scores[int(boundary)] = 1.0
    return boundaries, scores


def segment_embeddings(
    embeddings: np.ndarray,
    config: SegmentationConfig,
) -> np.ndarray:
    algorithm = config.common.segmentation_algorithm
    if algorithm == "goal_distance":
        indices = segment_goal_distance(
            embeddings,
            smooth_kernel=config.goal_distance.smooth_kernel,
            prominence=config.goal_distance.prominence,
            min_segment_len=config.goal_distance.min_segment_len,
        )
    elif algorithm == "window_mean":
        indices, _scores = segment_window_mean(
            embeddings,
            window=config.window_mean.window,
            prominence=config.window_mean.prominence,
            min_segment_len=config.window_mean.min_segment_len,
        )
    elif algorithm == "kernel_cpd":
        indices, _scores = segment_kernel_cpd(
            embeddings,
            window=config.kernel_cpd.window,
            gamma=config.kernel_cpd.gamma,
            prominence=config.kernel_cpd.prominence,
            min_segment_len=config.kernel_cpd.min_segment_len,
        )
    elif algorithm == "hmm":
        indices, _scores = segment_hmm(
            embeddings,
            num_states=config.hmm.num_states,
            stay_bias=config.hmm.stay_bias,
            min_segment_len=config.hmm.min_segment_len,
        )
    elif algorithm == "hsmm":
        indices, _scores = segment_hsmm(
            embeddings,
            num_states=config.hsmm.num_states,
            switch_cost=config.hsmm.switch_cost,
            min_duration=config.hsmm.min_duration,
            max_duration=config.hsmm.max_duration,
        )
    else:
        import uvd

        smooth_method = None if config.uvd.smooth_method == "none" else config.uvd.smooth_method
        _, decomp_meta = uvd.decomp_trajectories(
            "embed",
            embeddings,
            normalize_curve=config.uvd.normalize_curve,
            min_interval=config.uvd.min_interval,
            smooth_method=smooth_method,
            extrema_prominence=config.uvd.extrema_prominence,
            gamma=config.uvd.gamma,
            window_length=config.uvd.window_length,
        )
        indices = np.array(decomp_meta.milestone_indices, dtype=np.int64)

    indices = np.array(sorted(set(int(i) for i in indices)), dtype=np.int64)
    if indices.size == 0 or indices[-1] != len(embeddings) - 1:
        indices = np.concatenate([indices, [len(embeddings) - 1]])
    indices = np.clip(indices, 0, len(embeddings) - 1)
    return indices
