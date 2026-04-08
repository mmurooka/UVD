from __future__ import annotations

import argparse
import colorsys
import os
import subprocess
import sys
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


np = None
Image = None
ImageColor = None
ImageDraw = None
ImageFont = None


def import_rendering_dependencies():
    global np, Image, ImageColor, ImageDraw, ImageFont
    if np is None:
        import numpy as _np

        np = _np
    if Image is None:
        from PIL import Image as _Image
        from PIL import ImageColor as _ImageColor
        from PIL import ImageDraw as _ImageDraw
        from PIL import ImageFont as _ImageFont

        Image = _Image
        ImageColor = _ImageColor
        ImageDraw = _ImageDraw
        ImageFont = _ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a copy of a video with UVD segment overlays. "
            "The output is CPU-only by default."
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
        help='Inference device. Defaults to "cpu" to avoid GPU memory pressure.',
    )
    parser.add_argument(
        "--suffix",
        default="_uvd_segments",
        help="Suffix added before the output extension.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output fps. Defaults to the source video fps when available.",
    )
    parser.add_argument(
        "--border_size",
        type=int,
        default=12,
        help="Thickness of the colored border.",
    )
    parser.add_argument(
        "--progress_height",
        type=int,
        default=28,
        help="Height of the timeline/progress bar.",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.4,
        help="Overlay text scale.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Overlay text thickness.",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        default=18,
        help=(
            "Minimum distance between neighboring segmentation milestones in frames. "
            "Increase this to reduce over-segmentation."
        ),
    )
    parser.add_argument(
        "--smooth_method",
        default="kernel",
        choices=["kernel", "savgol", "none"],
        help="Curve smoothing method used before picking milestone candidates.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.08,
        help=(
            "Kernel smoothing strength for smooth_method=kernel. "
            "Larger values usually make the curve more local and can increase splits."
        ),
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=None,
        help=(
            "Optional backward window length for milestone search. "
            "Leave unset to use the full history."
        ),
    )
    parser.add_argument(
        "--normalize_curve",
        action="store_true",
        help="Enable distance-curve normalization before decomposition.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help=(
            "Maximum number of output segments. "
            "If exceeded, short neighboring segments are merged until this limit is met."
        ),
    )
    parser.add_argument(
        "--selection_mode",
        default="merge",
        choices=["merge", "topk"],
        help=(
            "How to reduce segments when --max_segments is set. "
            "'merge' merges short neighboring segments; "
            "'topk' keeps the strongest milestone candidates."
        ),
    )
    return parser.parse_args()


def build_output_path(video_file: str, suffix: str) -> str:
    video_file = os.path.expandvars(os.path.expanduser(video_file))
    base, ext = os.path.splitext(video_file)
    ext = ext or ".mp4"
    return f"{base}{suffix}{ext}"


def palette(n: int) -> list[tuple[int, int, int]]:
    colors = []
    for idx in range(max(n, 1)):
        hue = idx / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        colors.append(tuple(int(channel * 255) for channel in rgb))
    return colors


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


def limit_segment_count(
    milestone_indices: Iterable[int],
    embeddings: np.ndarray,
    num_frames: int,
    max_segments: int | None,
) -> np.ndarray:
    milestones = np.array(sorted(set(int(i) for i in milestone_indices)), dtype=np.int64)
    if milestones.size == 0 or milestones[-1] != num_frames - 1:
        milestones = np.concatenate([milestones, [num_frames - 1]])
    milestones = np.clip(milestones, 0, num_frames - 1)

    if max_segments is None or max_segments <= 0 or len(milestones) <= max_segments:
        return milestones

    while len(milestones) > max_segments:
        starts = np.concatenate([[0], milestones[:-1] + 1])
        lengths = milestones - starts + 1
        shortest_seg_idx = int(np.argmin(lengths))
        if len(milestones) == 1:
            break

        boundary_scores = np.linalg.norm(
            embeddings[milestones[:-1]] - embeddings[milestones[1:]],
            axis=1,
        )
        if shortest_seg_idx == 0:
            remove_idx = 0
        elif shortest_seg_idx == len(milestones) - 1:
            remove_idx = len(milestones) - 2
        else:
            left_score = boundary_scores[shortest_seg_idx - 1]
            right_score = boundary_scores[shortest_seg_idx]
            remove_idx = shortest_seg_idx - 1 if left_score <= right_score else shortest_seg_idx
        milestones = np.delete(milestones, remove_idx)

    return milestones


def select_topk_segments(
    milestone_indices: Iterable[int],
    milestone_scores: dict[int, float],
    max_segments: int | None,
    min_interval: int,
) -> np.ndarray:
    milestones = np.array(sorted(set(int(i) for i in milestone_indices)), dtype=np.int64)
    if (
        max_segments is None
        or max_segments <= 0
        or milestones.size <= max_segments
        or milestones.size == 0
    ):
        return milestones

    terminal_milestone = int(milestones[-1])
    selected = [terminal_milestone]
    other_milestones = milestones[milestones != terminal_milestone]
    scores = np.array([milestone_scores.get(int(m), float("-inf")) for m in other_milestones])
    order = np.argsort(scores)[::-1]
    spacing = max(1, int(min_interval))
    for idx in order:
        milestone = int(other_milestones[idx])
        if all(abs(milestone - kept) >= spacing for kept in selected):
            selected.append(milestone)
        if len(selected) >= max_segments:
            break

    if len(selected) < max_segments:
        for milestone in other_milestones[::-1]:
            milestone = int(milestone)
            if milestone not in selected:
                selected.append(milestone)
            if len(selected) >= max_segments:
                break

    return np.array(sorted(selected), dtype=np.int64)


def compute_milestone_scores(
    embeddings: np.ndarray,
    normalize_curve: bool,
    min_interval: int,
    window_length: int | None,
    smooth_method: str | None,
    gamma: float,
) -> dict[int, float]:
    from scipy.signal import argrelextrema, savgol_filter

    from uvd.decomp.kernel_reg import KernelRegression

    traj_length = embeddings.shape[0]
    cur_goal_idx = traj_length - 1
    cur_embeddings = embeddings[
        max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
    ]
    milestone_scores = {cur_goal_idx: float("inf")}

    while cur_goal_idx > (window_length or min_interval):
        goal_embedding = cur_embeddings[-1]
        distances = np.linalg.norm(cur_embeddings - goal_embedding, axis=1)
        if normalize_curve:
            denom = np.linalg.norm(cur_embeddings[0] - goal_embedding)
            distances = distances / max(float(denom), 1e-8)

        x = np.arange(
            max(0, cur_goal_idx - (window_length or cur_goal_idx)), cur_goal_idx + 1
        )
        if smooth_method == "kernel":
            kr = KernelRegression(kernel="rbf", gamma=gamma)
            kr.fit(x.reshape(-1, 1), distances)
            distance_smoothed = kr.predict(x.reshape(-1, 1))
        elif smooth_method == "savgol":
            sg_window = min(len(distances) if len(distances) % 2 == 1 else len(distances) - 1, 85)
            sg_window = max(sg_window, 3)
            if sg_window % 2 == 0:
                sg_window -= 1
            if sg_window >= 3:
                distance_smoothed = savgol_filter(
                    distances, window_length=sg_window, polyorder=min(2, sg_window - 1), mode="nearest"
                )
            else:
                distance_smoothed = distances
        else:
            distance_smoothed = distances

        extrema_indices = argrelextrema(distance_smoothed, np.greater)[0]
        x_extrema = x[extrema_indices]

        update_goal = False
        for i in range(len(x_extrema) - 1, -1, -1):
            if cur_goal_idx < min_interval:
                break
            if (
                cur_goal_idx - x_extrema[i] > min_interval
                and x_extrema[i] > min_interval
            ):
                cur_goal_idx = int(x_extrema[i])
                milestone_scores[cur_goal_idx] = float(distance_smoothed[extrema_indices[i]])
                update_goal = True
                break

        if not update_goal or cur_goal_idx < min_interval:
            break
        cur_embeddings = embeddings[
            max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
        ]

    return milestone_scores


def draw_border(frame: np.ndarray, color: tuple[int, int, int], border_size: int) -> np.ndarray:
    if border_size <= 0:
        return frame
    h, w, _ = frame.shape
    framed = np.full((h + border_size * 2, w + border_size * 2, 3), color, dtype=np.uint8)
    framed[border_size:-border_size, border_size:-border_size] = frame
    return framed


def draw_progress_bar(
    frame: np.ndarray,
    segment_ids: np.ndarray,
    segment_colors: list[tuple[int, int, int]],
    frame_idx: int,
    current_segment: int,
    progress_height: int,
) -> np.ndarray:
    if progress_height <= 0:
        return frame

    h, w, _ = frame.shape
    bar = np.full((progress_height, w, 3), 18, dtype=np.uint8)
    total_frames = len(segment_ids)
    x_edges = np.linspace(0, w, total_frames + 1, dtype=np.int64)
    for idx in range(total_frames):
        x0, x1 = x_edges[idx], x_edges[idx + 1]
        if x1 <= x0:
            continue
        color = segment_colors[int(segment_ids[idx])]
        bar[:, x0:x1] = np.array(color, dtype=np.uint8)

    current_x = min(w - 1, x_edges[min(frame_idx + 1, total_frames)] - 1)
    pil_bar = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil_bar)
    draw.line(
        [(current_x, 0), (current_x, progress_height - 1)],
        fill=(255, 255, 255),
        width=3,
    )
    label = f"segment {current_segment + 1}/{len(segment_colors)}"
    draw.text(
        (12, max(4, progress_height // 2 - 8)),
        label,
        fill=(255, 255, 255),
        font=load_font(max(12, progress_height - 10)),
    )
    bar = np.array(pil_bar, dtype=np.uint8)
    return np.concatenate([frame, bar], axis=0)


def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    segment_ids: np.ndarray,
    segment_colors: list[tuple[int, int, int]],
    border_size: int,
    progress_height: int,
    font_scale: float,
    thickness: int,
) -> np.ndarray:
    current_segment = int(segment_ids[frame_idx])
    total_segments = len(segment_colors)
    color = segment_colors[current_segment]
    annotated = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(annotated, "RGBA")

    label = f"{current_segment + 1}/{total_segments}"
    font = load_font(max(16, int(24 * font_scale)))
    pad = 14
    box_x0, box_y0 = 16, 16
    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=thickness)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_x1 = box_x0 + text_w + 2 * pad
    box_y1 = box_y0 + text_h + 2 * pad
    draw.rounded_rectangle(
        (box_x0, box_y0, box_x1, box_y1),
        radius=10,
        fill=(0, 0, 0, 140),
    )
    draw.text(
        (box_x0 + pad, box_y0 + pad),
        label,
        fill=color,
        font=font,
        stroke_width=max(1, thickness // 2),
        stroke_fill=tuple(ImageColor.getrgb("white")),
    )

    annotated = np.array(annotated, dtype=np.uint8)
    annotated = draw_border(annotated, color, border_size)
    annotated = draw_progress_bar(
        annotated,
        segment_ids=segment_ids,
        segment_colors=segment_colors,
        frame_idx=frame_idx,
        current_segment=current_segment,
        progress_height=progress_height,
    )
    return annotated


def save_video_fallback_ffmpeg(video: np.ndarray, output_path: str, fps: float) -> None:
    h, w = video.shape[1:3]
    command = [
        "/usr/bin/ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        proc.stdin.write(video.astype(np.uint8, copy=False).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main():
    args = parse_args()
    video_file = os.path.expandvars(os.path.expanduser(args.video_file))
    output_path = build_output_path(video_file, args.suffix)

    if str(args.device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    import torch

    import_rendering_dependencies()

    import decord

    import uvd
    import uvd.utils as U

    print(f"[1/5] Opening video: {video_file}", flush=True)
    vr = decord.VideoReader(video_file)
    fps = args.fps or float(vr.get_avg_fps() or 30.0)
    print(f"[2/5] Loading frames into memory at source fps={fps:.3f}", flush=True)
    frames = vr[:].asnumpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video as (T, H, W, 3), got {frames.shape}")
    print(
        f"[2/5] Loaded {len(frames)} frames with shape {frames.shape[1:]}",
        flush=True,
    )

    print(
        f"[3/5] Running UVD segmentation on {args.device} with "
        f"preprocessor={args.preprocessor_name}",
        flush=True,
    )
    preprocessor = uvd.get_preprocessor(args.preprocessor_name, device=args.device)
    rep = preprocessor.process(frames, return_numpy=True)
    smooth_method = None if args.smooth_method == "none" else args.smooth_method
    _, decomp_meta = uvd.decomp_trajectories(
        "embed",
        rep,
        normalize_curve=args.normalize_curve,
        min_interval=args.min_interval,
        smooth_method=smooth_method,
        gamma=args.gamma,
        window_length=args.window_length,
    )
    raw_indices = np.array(decomp_meta.milestone_indices, dtype=np.int64)
    if args.selection_mode == "topk":
        milestone_scores = compute_milestone_scores(
            rep,
            normalize_curve=args.normalize_curve,
            min_interval=args.min_interval,
            window_length=args.window_length,
            smooth_method=smooth_method,
            gamma=args.gamma,
        )
        indices = select_topk_segments(
            raw_indices,
            milestone_scores=milestone_scores,
            max_segments=args.max_segments,
            min_interval=args.min_interval,
        )
    else:
        indices = limit_segment_count(
            raw_indices,
            embeddings=rep,
            num_frames=len(frames),
            max_segments=args.max_segments,
        )
    print(
        f"[3/5] Segmentation complete. Detected {len(raw_indices)} milestones"
        + (
            f", reduced to {len(indices)} by --max_segments={args.max_segments}"
            if args.max_segments is not None and len(indices) != len(raw_indices)
            else ""
        )
        + f": "
        f"{list(map(int, indices))}",
        flush=True,
    )
    segment_ids = assign_segments(len(frames), indices)
    segment_colors = palette(int(segment_ids.max()) + 1)
    print(
        f"[4/5] Rendering overlay frames for {len(segment_colors)} segments",
        flush=True,
    )

    frame_iter = range(len(frames))
    if tqdm is not None:
        frame_iter = tqdm(
            frame_iter,
            total=len(frames),
            desc="Rendering frames",
            unit="frame",
        )

    rendered = np.stack(
        [
            annotate_frame(
                frame=frames[idx],
                frame_idx=idx,
                segment_ids=segment_ids,
                segment_colors=segment_colors,
                border_size=args.border_size,
                progress_height=args.progress_height,
                font_scale=args.font_scale,
                thickness=args.thickness,
            )
            for idx in frame_iter
        ],
        axis=0,
    )

    print(f"[5/5] Writing output video: {output_path}", flush=True)
    try:
        U.save_video(rendered, output_path, fps=fps, compress=False)
    except ImportError as exc:
        print(
            "[5/5] torchvision video writer unavailable; falling back to ffmpeg",
            flush=True,
        )
        save_video_fallback_ffmpeg(rendered, output_path, fps=fps)
    print(f"Saved segmented video to: {output_path}")
    print(f"Detected {len(segment_colors)} segments.")
    print(f"Milestone frame indices: {list(map(int, indices))}")


if __name__ == "__main__":
    main()
