from __future__ import annotations

import argparse
import os

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively tune UVD segmentation parameters without rendering videos. "
            "Embeddings are computed once and reused."
        )
    )
    parser.add_argument("video_file", help="Path to the input video.")
    parser.add_argument(
        "--preprocessor_name",
        default="vip",
        choices=["vip", "r3m", "liv", "clip", "vc1", "dinov2", "resnet"],
        help="Frozen visual encoder used for segmentation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device. Defaults to "cpu".',
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=64,
        help="Number of frames per embedding batch.",
    )
    parser.add_argument(
        "--smooth_method",
        default="kernel",
        choices=["kernel", "savgol", "none"],
        help="Curve smoothing method used for UVD decomposition.",
    )
    parser.add_argument(
        "--normalize_curve",
        action="store_true",
        help="Enable distance-curve normalization before decomposition.",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=None,
        help="Optional backward window length for milestone search.",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        default=18,
        help="Initial minimum interval in frames.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.08,
        help="Initial kernel gamma.",
    )
    parser.add_argument(
        "--extrema_prominence",
        type=float,
        default=0.0,
        help="Initial extrema prominence. Zero means disabled.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=0,
        help="Optional hard cap on segment count. Zero means disabled.",
    )
    parser.add_argument(
        "--selection_mode",
        default="merge",
        choices=["merge", "topk"],
        help="How to reduce segment count when max_segments is enabled.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    if str(args.device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import decord
    import uvd
    from matplotlib.widgets import Button
    from matplotlib.widgets import Slider

    import render_segmented_video as rsv

    smooth_method = None if args.smooth_method == "none" else args.smooth_method

    print(f"[1/3] Opening video: {args.video_file}", flush=True)
    vr = decord.VideoReader(os.path.expandvars(os.path.expanduser(args.video_file)))
    frames = vr[:].asnumpy()
    print(f"[2/3] Computing embeddings once in batches of {args.embed_batch_size}", flush=True)
    preprocessor = uvd.get_preprocessor(args.preprocessor_name, device=args.device)
    rep = rsv.compute_embeddings_in_batches(
        preprocessor,
        frames,
        batch_size=args.embed_batch_size,
    )
    base_curve = compute_distance_curve(rep, normalize_curve=args.normalize_curve)
    print("[3/3] Launching interactive plot", flush=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.9)

    x = np.arange(len(base_curve))
    raw_line, = ax.plot(x, base_curve, color="0.55", linewidth=1.5, label="distance")
    smooth_line, = ax.plot(
        x,
        smooth_distance_curve(base_curve, smooth_method=smooth_method, gamma=args.gamma),
        color="tab:blue",
        linewidth=2.5,
        label="smoothed score",
    )
    vline_artists = []
    span_artists = []
    info_text = ax.text(
        0.01,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10,
        family="monospace",
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Interactive UVD Segmentation Tuning")
    ax.legend(loc="upper right")

    slider_specs = [
        ("gamma", 0.001, 0.5, args.gamma, 0.001),
        ("prominence", 0.0, 1.0, args.extrema_prominence, 0.005),
        ("min_interval", 1, min(300, max(30, len(base_curve) // 2)), args.min_interval, 1),
        ("max_segments", 0, 50, args.max_segments, 1),
    ]
    sliders = {}
    for idx, (name, vmin, vmax, initial, step) in enumerate(slider_specs):
        slider_ax = fig.add_axes([0.12, 0.22 - idx * 0.045, 0.72, 0.025])
        sliders[name] = Slider(
            slider_ax,
            name,
            vmin,
            vmax,
            valinit=initial,
            valstep=step,
        )

    update_ax = fig.add_axes([0.86, 0.07, 0.1, 0.05])
    update_button = Button(update_ax, "Update")
    status_text = fig.text(
        0.12,
        0.03,
        "Ready",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    cache: dict[tuple[float, float | None, int, int | None, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pending_update = False

    def mark_dirty(_=None):
        nonlocal pending_update
        pending_update = True
        status_text.set_text("Parameters changed. Click Update or press Enter.")
        fig.canvas.draw_idle()

    def recompute(_=None):
        nonlocal vline_artists, span_artists, pending_update
        gamma = float(sliders["gamma"].val)
        extrema_prominence = float(sliders["prominence"].val)
        extrema_prominence = None if extrema_prominence <= 0 else extrema_prominence
        min_interval = int(sliders["min_interval"].val)
        max_segments = int(sliders["max_segments"].val)
        max_segments = None if max_segments <= 0 else max_segments
        cache_key = (
            round(gamma, 6),
            None if extrema_prominence is None else round(extrema_prominence, 6),
            min_interval,
            max_segments,
            args.selection_mode,
        )
        if cache_key in cache:
            raw_indices, indices, smooth_curve = cache[cache_key]
        else:
            _, decomp_meta = uvd.decomp_trajectories(
                "embed",
                rep,
                normalize_curve=args.normalize_curve,
                min_interval=min_interval,
                smooth_method=smooth_method,
                gamma=gamma,
                window_length=args.window_length,
                extrema_prominence=extrema_prominence,
            )
            raw_indices = np.array(decomp_meta.milestone_indices, dtype=np.int64)
            if args.selection_mode == "topk" and max_segments is not None:
                milestone_scores = rsv.compute_milestone_scores(
                    rep,
                    normalize_curve=args.normalize_curve,
                    min_interval=min_interval,
                    window_length=args.window_length,
                    smooth_method=smooth_method,
                    gamma=gamma,
                    extrema_prominence=extrema_prominence,
                )
                indices = rsv.select_topk_segments(
                    raw_indices,
                    milestone_scores=milestone_scores,
                    max_segments=max_segments,
                    min_interval=min_interval,
                )
            else:
                indices = rsv.limit_segment_count(
                    raw_indices,
                    embeddings=rep,
                    num_frames=len(frames),
                    max_segments=max_segments,
                )
            smooth_curve = smooth_distance_curve(
                base_curve,
                smooth_method=smooth_method,
                gamma=gamma,
            )
            cache[cache_key] = (raw_indices, indices, smooth_curve)

        segment_ids = rsv.assign_segments(len(frames), indices)
        segment_colors = rsv.palette(int(segment_ids.max()) + 1)
        smooth_line.set_ydata(smooth_curve)

        for artist in vline_artists + span_artists:
            artist.remove()
        vline_artists = []
        span_artists = []

        starts = np.concatenate([[0], np.array(indices[:-1], dtype=np.int64) + 1]) if len(indices) > 0 else np.array([0])
        ends = np.array(indices, dtype=np.int64) if len(indices) > 0 else np.array([len(frames) - 1])
        for seg_idx, (start, end) in enumerate(zip(starts, ends)):
            color = np.array(segment_colors[seg_idx]) / 255.0
            span_artists.append(ax.axvspan(start, end, color=color, alpha=0.12, linewidth=0))
        for idx in indices:
            vline_artists.append(ax.axvline(int(idx), color="black", linestyle="--", linewidth=1.0, alpha=0.7))

        info_text.set_text(
            "\n".join(
                [
                    f"segments: {len(indices)}",
                    f"raw milestones: {len(raw_indices)}",
                    f"gamma: {gamma:.3f}",
                    f"prominence: {0.0 if extrema_prominence is None else extrema_prominence:.3f}",
                    f"min_interval: {min_interval}",
                    f"indices: {list(map(int, indices[:12]))}" + (" ..." if len(indices) > 12 else ""),
                ]
            )
        )
        pending_update = False
        status_text.set_text("Updated")
        ax.set_xlim(0, len(base_curve) - 1)
        ymax = max(base_curve.max(), smooth_curve.max())
        ax.set_ylim(min(base_curve.min(), smooth_curve.min()) - 0.02 * ymax, ymax * 1.08 if ymax > 0 else 1.0)
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(mark_dirty)

    def on_key_press(event):
        if event.key in {"enter", "return"}:
            recompute()

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    update_button.on_clicked(recompute)

    recompute()
    plt.show()


if __name__ == "__main__":
    main()
