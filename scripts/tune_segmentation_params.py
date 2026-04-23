from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import segmentation_pipeline as sp


SLIDER_SPEC_BUILDERS = {
    "uvd": lambda cfg, curve_len: [
        ("gamma", 0.001, 0.5, cfg.uvd.gamma, 0.001),
        ("prominence", 0.0, 1.0, cfg.uvd.extrema_prominence or 0.0, 0.005),
        ("min_interval", 1, min(300, max(30, curve_len // 2)), cfg.uvd.min_interval, 1),
    ],
    "goal_distance": lambda cfg, curve_len: [
        (
            "goal_distance_kernel",
            1,
            101,
            cfg.goal_distance.smooth_kernel,
            2,
        ),
        (
            "goal_distance_prominence",
            0.0,
            1.0,
            cfg.goal_distance.prominence,
            0.005,
        ),
        (
            "goal_distance_min_len",
            1,
            min(300, max(30, curve_len // 2)),
            cfg.goal_distance.min_segment_len,
            1,
        ),
    ],
    "window_mean": lambda cfg, curve_len: [
        (
            "window_mean_window",
            1,
            min(300, max(30, curve_len // 2)),
            cfg.window_mean.window,
            1,
        ),
        (
            "window_mean_prominence",
            0.0,
            5.0,
            cfg.window_mean.prominence,
            0.01,
        ),
        (
            "window_mean_min_len",
            1,
            min(300, max(30, curve_len // 2)),
            cfg.window_mean.min_segment_len,
            1,
        ),
    ],
    "kernel_cpd": lambda cfg, curve_len: [
        (
            "kcpd_window",
            2,
            min(300, max(30, curve_len // 2)),
            cfg.kernel_cpd.window,
            1,
        ),
        ("kcpd_gamma", 0.0001, 1.0, cfg.kernel_cpd.gamma, 0.0001),
        (
            "kcpd_prominence",
            0.0,
            1.0,
            cfg.kernel_cpd.prominence,
            0.001,
        ),
        (
            "kcpd_min_len",
            1,
            min(300, max(30, curve_len // 2)),
            cfg.kernel_cpd.min_segment_len,
            1,
        ),
    ],
    "hmm": lambda cfg, curve_len: [
        ("hmm_states", 2, 10, cfg.hmm.num_states, 1),
        ("hmm_bias", 0.0, 20.0, cfg.hmm.stay_bias, 0.1),
        (
            "hmm_min_len",
            1,
            min(300, max(30, curve_len // 2)),
            cfg.hmm.min_segment_len,
            1,
        ),
    ],
    "hsmm": lambda cfg, curve_len: [
        ("hsmm_states", 2, 10, cfg.hsmm.num_states, 1),
        ("hsmm_switch", 0.0, 20.0, cfg.hsmm.switch_cost, 0.1),
        (
            "hsmm_min_dur",
            1,
            min(300, max(30, curve_len // 2)),
            cfg.hsmm.min_duration,
            1,
        ),
        (
            "hsmm_max_dur",
            1,
            min(600, max(60, curve_len)),
            cfg.hsmm.max_duration if cfg.hsmm.max_duration is not None else min(600, max(60, curve_len)),
            1,
        ),
    ],
}


INFO_TEXT_BUILDERS = {
    "uvd": lambda cfg, curve_mode, indices: [
        f"segments: {len(indices)}",
        "algorithm: uvd",
        f"curve_mode: {curve_mode}",
    ],
    "goal_distance": lambda cfg, curve_mode, indices: [
        f"segments: {len(indices)}",
        "algorithm: goal_distance",
        f"curve_mode: {curve_mode}",
    ],
    "window_mean": lambda cfg, curve_mode, indices: [
        f"segments: {len(indices)}",
        "algorithm: window_mean",
        f"curve_mode: {curve_mode}",
    ],
    "kernel_cpd": lambda cfg, curve_mode, indices: [
        f"segments: {len(indices)}",
        "algorithm: kernel_cpd",
        f"curve_mode: {curve_mode}",
    ],
    "hmm": lambda cfg, curve_mode, indices: [
        f"segments: {len(indices)}",
        "algorithm: hmm",
        f"curve_mode: {curve_mode}",
    ],
    "hsmm": lambda cfg, curve_mode, indices: [
        f"segments: {len(indices)}",
        "algorithm: hsmm",
        f"curve_mode: {curve_mode}",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively tune segmentation parameters without rendering videos. "
            "Embeddings are computed once and reused."
        )
    )
    sp.add_common_cli_args(
        parser,
        defaults=sp.build_cli_defaults(),
    )
    return parser.parse_args()


def configure_runtime_env(device: str) -> None:
    if str(device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


def configure_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )


def build_slider_specs(config: sp.SegmentationConfig, curve_len: int) -> list[tuple[str, float, float, float, float]]:
    return SLIDER_SPEC_BUILDERS[config.common.segmentation_algorithm](config, curve_len)


def read_config_from_sliders(
    base_config: sp.SegmentationConfig,
    sliders: dict,
) -> sp.SegmentationConfig:
    algorithm = base_config.common.segmentation_algorithm
    if algorithm == "uvd":
        return replace(
            base_config,
            uvd=replace(
                base_config.uvd,
                gamma=float(sliders["gamma"].val),
                extrema_prominence=(
                    None if float(sliders["prominence"].val) <= 0 else float(sliders["prominence"].val)
                ),
                min_interval=int(sliders["min_interval"].val),
            ),
        )
    if algorithm == "goal_distance":
        return replace(
            base_config,
            goal_distance=replace(
                base_config.goal_distance,
                smooth_kernel=int(sliders["goal_distance_kernel"].val),
                prominence=float(sliders["goal_distance_prominence"].val),
                min_segment_len=int(sliders["goal_distance_min_len"].val),
            ),
        )
    if algorithm == "window_mean":
        return replace(
            base_config,
            window_mean=replace(
                base_config.window_mean,
                window=int(sliders["window_mean_window"].val),
                prominence=float(sliders["window_mean_prominence"].val),
                min_segment_len=int(sliders["window_mean_min_len"].val),
            ),
        )
    if algorithm == "kernel_cpd":
        return replace(
            base_config,
            kernel_cpd=replace(
                base_config.kernel_cpd,
                window=int(sliders["kcpd_window"].val),
                gamma=float(sliders["kcpd_gamma"].val),
                prominence=float(sliders["kcpd_prominence"].val),
                min_segment_len=int(sliders["kcpd_min_len"].val),
            ),
        )
    if algorithm == "hmm":
        return replace(
            base_config,
            hmm=replace(
                base_config.hmm,
                num_states=int(sliders["hmm_states"].val),
                stay_bias=float(sliders["hmm_bias"].val),
                min_segment_len=int(sliders["hmm_min_len"].val),
            ),
        )
    return replace(
        base_config,
        hsmm=replace(
            base_config.hsmm,
            num_states=int(sliders["hsmm_states"].val),
            switch_cost=float(sliders["hsmm_switch"].val),
            min_duration=int(sliders["hsmm_min_dur"].val),
            max_duration=int(sliders["hsmm_max_dur"].val),
        ),
    )


def build_cache_key(config: sp.SegmentationConfig, curve_mode: str) -> tuple[str, str]:
    return curve_mode, repr(config)


def compute_plot_curves(
    rep: np.ndarray,
    base_curve: np.ndarray,
    config: sp.SegmentationConfig,
    curve_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = sp.segment_embeddings(rep, config=config)
    smooth_method = None if config.uvd.smooth_method == "none" else config.uvd.smooth_method
    if curve_mode == "segment_goal":
        raw_curve, smooth_curve = sp.compute_segment_goal_curve(
            rep,
            indices,
            normalize_curve=config.uvd.normalize_curve,
            smooth_method=smooth_method,
            gamma=config.uvd.gamma,
        )
    else:
        raw_curve = base_curve
        smooth_curve = sp.smooth_distance_curve(
            base_curve,
            smooth_method=smooth_method,
            gamma=config.uvd.gamma,
        )
    return indices, raw_curve, smooth_curve


def update_segment_artists(ax, indices, segment_colors, vline_artists, span_artists):
    for artist in vline_artists + span_artists:
        artist.remove()
    vline_artists.clear()
    span_artists.clear()

    starts = (
        np.concatenate([[0], np.array(indices[:-1], dtype=np.int64) + 1])
        if len(indices) > 0
        else np.array([0])
    )
    ends = (
        np.array(indices, dtype=np.int64)
        if len(indices) > 0
        else np.array([0], dtype=np.int64)
    )
    for seg_idx, (start, end) in enumerate(zip(starts, ends)):
        color = np.array(segment_colors[seg_idx]) / 255.0
        span_artists.append(ax.axvspan(start, end, color=color, alpha=0.12, linewidth=0))
    for idx in indices:
        vline_artists.append(
            ax.axvline(int(idx), color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        )


def update_plot_data(ax, raw_line, smooth_line, raw_curve, smooth_curve, base_curve):
    raw_line.set_ydata(raw_curve)
    smooth_line.set_ydata(smooth_curve)
    ax.set_xlim(0, len(base_curve) - 1)
    ymax = max(raw_curve.max(), smooth_curve.max())
    ymin = min(raw_curve.min(), smooth_curve.min())
    ax.set_ylim(ymin - 0.02 * max(ymax, 1e-6), ymax * 1.08 if ymax > 0 else 1.0)


def build_info_text(config: sp.SegmentationConfig, curve_mode: str, indices: np.ndarray) -> str:
    return "\n".join(
        INFO_TEXT_BUILDERS[config.common.segmentation_algorithm](config, curve_mode, indices)
    )


def main():
    args = parse_args()
    configure_runtime_env(args.device)

    import torch
    import matplotlib.pyplot as plt
    import decord
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button
    from matplotlib.widgets import RadioButtons
    from matplotlib.widgets import Slider

    del torch

    base_config = sp.segmentation_config_from_args(args)

    print(f"[1/3] Opening video: {args.video_file}", flush=True)
    vr = decord.VideoReader(os.path.expandvars(os.path.expanduser(args.video_file)))
    frames = vr[:].asnumpy()
    cache_path = (
        None
        if base_config.common.no_embedding_cache
        else sp.build_embedding_cache_path(
            video_file=args.video_file,
            preprocessor_name=base_config.common.preprocessor_name,
            embedding_cache=base_config.common.embedding_cache,
        )
    )
    print(
        (
            f"[2/3] Loading cached embeddings: {cache_path}"
            if cache_path is not None and os.path.exists(cache_path)
            else f"[2/3] Computing embeddings once in batches of {base_config.common.embed_batch_size}"
        ),
        flush=True,
    )
    rep, cache_path, loaded_from_cache = sp.load_or_compute_embeddings(
        video_file=args.video_file,
        frames=frames,
        preprocessor_name=base_config.common.preprocessor_name,
        device=base_config.common.device,
        batch_size=base_config.common.embed_batch_size,
        embedding_cache=base_config.common.embedding_cache,
        no_embedding_cache=base_config.common.no_embedding_cache,
    )
    if cache_path is not None and not loaded_from_cache:
        print(f"[2/3] Saved embeddings to: {cache_path}", flush=True)
    base_curve = sp.compute_distance_curve(rep, normalize_curve=base_config.uvd.normalize_curve)
    print("[3/3] Launching interactive plot", flush=True)

    configure_plot_style(plt)
    fig = plt.figure(figsize=(15.5, 9.5))
    image_ax = fig.add_axes([0.06, 0.53, 0.36, 0.38])
    ax = fig.add_axes([0.46, 0.53, 0.50, 0.38])
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.16, top=0.93)

    x = np.arange(len(base_curve))
    current_frame_idx = 0
    image_artist = image_ax.imshow(frames[current_frame_idx])
    image_ax.set_title(f"Frame {current_frame_idx + 1}/{len(frames)}")
    frame_h, frame_w = frames[current_frame_idx].shape[:2]
    image_border = Rectangle(
        (0, 0),
        frame_w - 1,
        frame_h - 1,
        fill=False,
        linewidth=8.0,
        edgecolor=(1.0, 1.0, 1.0),
    )
    image_ax.add_patch(image_border)
    image_ax.axis("off")

    initial_smooth_method = (
        None if base_config.uvd.smooth_method == "none" else base_config.uvd.smooth_method
    )
    raw_line, = ax.plot(x, base_curve, color="0.55", linewidth=1.5, label="distance")
    smooth_line, = ax.plot(
        x,
        sp.smooth_distance_curve(
            base_curve,
            smooth_method=initial_smooth_method,
            gamma=base_config.uvd.gamma,
        ),
        color="tab:blue",
        linewidth=2.5,
        label="smoothed score",
    )
    cursor_line = ax.axvline(current_frame_idx, color="tab:red", linewidth=2.0, alpha=0.9)
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
        fontsize=14,
        family="monospace",
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Interactive Segmentation Tuning")
    ax.legend(loc="upper right")

    param_left = 0.09
    param_width = 0.64
    param_top = 0.42
    control_left = 0.78
    control_width = 0.17

    sliders = {}
    slider_specs = build_slider_specs(base_config, len(base_curve))
    slider_height = 0.034
    slider_gap = 0.022
    for idx, (name, vmin, vmax, initial, step) in enumerate(slider_specs):
        slider_y = param_top - (idx + 1) * (slider_height + slider_gap)
        slider_ax = fig.add_axes([param_left, slider_y, param_width, slider_height])
        slider_ax.set_facecolor("#f4f6f8")
        sliders[name] = Slider(
            slider_ax,
            name,
            vmin,
            vmax,
            valinit=initial,
            valstep=step,
            color="#4c78a8",
            initcolor="none",
        )
        sliders[name].label.set_fontsize(14)
        sliders[name].valtext.set_fontsize(13)

    fig.text(
        param_left,
        0.115,
        "Frame Navigation",
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    frame_slider_ax = fig.add_axes([param_left, 0.075, 0.84, 0.05])
    frame_slider_ax.set_facecolor("#1f2430")
    frame_slider = Slider(
        frame_slider_ax,
        "frame",
        0,
        max(0, len(frames) - 1),
        valinit=current_frame_idx,
        valstep=1,
        color="#f28e2b",
        initcolor="none",
    )
    frame_slider.label.set_fontsize(14)
    frame_slider.label.set_color("white")
    frame_slider.valtext.set_fontsize(14)
    frame_slider.valtext.set_color("white")
    if hasattr(frame_slider, "poly"):
        frame_slider.poly.set_alpha(0.95)
    if hasattr(frame_slider, "track"):
        frame_slider.track.set_facecolor("#3a4252")
    frame_slider_ax.tick_params(colors="white")

    fig.text(
        control_left,
        0.42,
        "Controls",
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    curve_mode_ax = fig.add_axes([control_left, 0.28, control_width, 0.11])
    curve_mode_radio = RadioButtons(curve_mode_ax, ("final_goal", "segment_goal"), active=0)
    for label in curve_mode_radio.labels:
        label.set_fontsize(14)
    update_ax = fig.add_axes([control_left, 0.20, control_width, 0.055])
    update_button = Button(update_ax, "Update")
    update_button.label.set_fontsize(14)
    status_text = fig.text(
        control_left,
        0.15,
        "Ready",
        ha="left",
        va="bottom",
        fontsize=13,
    )

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    pending_update = False
    current_segment_ids = np.zeros(len(frames), dtype=np.int64)
    current_segment_colors = [(255, 255, 255)]

    def mark_dirty(_=None):
        nonlocal pending_update
        pending_update = True
        status_text.set_text("Parameters changed. Click Update or press Enter.")
        fig.canvas.draw_idle()

    def update_frame_preview(frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
        image_artist.set_data(frames[frame_idx])
        image_ax.set_title(f"Frame {frame_idx + 1}/{len(frames)}")
        seg_idx = int(current_segment_ids[frame_idx])
        border_color = np.array(current_segment_colors[seg_idx]) / 255.0
        image_border.set_edgecolor(border_color)
        cursor_line.set_xdata([frame_idx, frame_idx])
        fig.canvas.draw_idle()

    def recompute(_=None):
        nonlocal pending_update, current_segment_ids, current_segment_colors
        config = read_config_from_sliders(base_config, sliders)
        curve_mode = curve_mode_radio.value_selected
        cache_key = build_cache_key(config, curve_mode)
        if cache_key in cache:
            indices, raw_curve, smooth_curve = cache[cache_key]
        else:
            indices, raw_curve, smooth_curve = compute_plot_curves(
                rep=rep,
                base_curve=base_curve,
                config=config,
                curve_mode=curve_mode,
            )
            cache[cache_key] = (indices, raw_curve, smooth_curve)

        segment_ids = sp.assign_segments(len(frames), indices)
        segment_colors = sp.palette(int(segment_ids.max()) + 1)
        current_segment_ids = segment_ids
        current_segment_colors = segment_colors

        update_plot_data(ax, raw_line, smooth_line, raw_curve, smooth_curve, base_curve)
        update_segment_artists(ax, indices, segment_colors, vline_artists, span_artists)
        info_text.set_text(build_info_text(config, curve_mode, indices))

        pending_update = False
        status_text.set_text("Updated")
        update_frame_preview(int(frame_slider.val))
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(mark_dirty)
    curve_mode_radio.on_clicked(mark_dirty)
    frame_slider.on_changed(lambda val: update_frame_preview(int(val)))

    def on_key_press(event):
        if event.key in {"enter", "return"}:
            recompute()
            return
        if event.key == "left":
            frame_slider.set_val(max(0, int(frame_slider.val) - 1))
        elif event.key == "right":
            frame_slider.set_val(min(len(frames) - 1, int(frame_slider.val) + 1))

    def on_graph_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        frame_slider.set_val(int(np.clip(round(event.xdata), 0, len(frames) - 1)))

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("button_press_event", on_graph_click)
    update_button.on_clicked(recompute)

    recompute()
    plt.show()


if __name__ == "__main__":
    main()
