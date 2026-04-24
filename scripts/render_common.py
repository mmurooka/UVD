from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass


np = None
Image = None
ImageColor = None
ImageDraw = None
ImageFont = None


@dataclass(frozen=True)
class RenderConfig:
    suffix: str = ".segments"
    fps: float | None = None
    border_size: int = 12
    progress_height: int = 28
    font_scale: float = 1.4
    text_thickness: int = 3


RENDER_DEFAULTS = RenderConfig()


def build_render_cli_defaults(overrides: dict[str, object] | None = None) -> dict[str, object]:
    defaults = {
        "render_suffix": RENDER_DEFAULTS.suffix,
        "render_fps": RENDER_DEFAULTS.fps,
        "render_border_size": RENDER_DEFAULTS.border_size,
        "render_progress_height": RENDER_DEFAULTS.progress_height,
        "render_font_scale": RENDER_DEFAULTS.font_scale,
        "render_text_thickness": RENDER_DEFAULTS.text_thickness,
    }
    if overrides:
        defaults.update(overrides)
    return defaults


def add_render_cli_args(
    parser: argparse.ArgumentParser,
    defaults: dict[str, object] | None = None,
) -> argparse.ArgumentParser:
    defaults = build_render_cli_defaults(defaults)
    render = parser.add_argument_group("Render options")
    render.add_argument(
        "--render_suffix",
        dest="render_suffix",
        default=defaults["render_suffix"],
        help="Suffix added before the output extension.",
    )
    render.add_argument(
        "--render_fps",
        dest="render_fps",
        type=float,
        default=defaults["render_fps"],
        help="Override output fps. Defaults to the source video fps when available.",
    )
    render.add_argument(
        "--render_border_size",
        dest="render_border_size",
        type=int,
        default=defaults["render_border_size"],
        help="Thickness of the colored border.",
    )
    render.add_argument(
        "--render_progress_height",
        dest="render_progress_height",
        type=int,
        default=defaults["render_progress_height"],
        help="Height of the timeline/progress bar.",
    )
    render.add_argument(
        "--render_font_scale",
        dest="render_font_scale",
        type=float,
        default=defaults["render_font_scale"],
        help="Overlay text scale.",
    )
    render.add_argument(
        "--render_text_thickness",
        dest="render_text_thickness",
        type=int,
        default=defaults["render_text_thickness"],
        help="Overlay text thickness.",
    )
    return parser


def import_numpy_dependency():
    global np
    if np is None:
        import numpy as _np

        np = _np


def import_rendering_dependencies():
    global Image, ImageColor, ImageDraw, ImageFont
    import_numpy_dependency()
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


def build_output_path(video_file: str, suffix: str) -> str:
    video_file = os.path.expandvars(os.path.expanduser(video_file))
    video_dir = os.path.dirname(video_file)
    video_name, ext = os.path.splitext(os.path.basename(video_file))
    output_dir = os.path.join(video_dir, "segments")
    ext = ext or ".mp4"
    return os.path.join(output_dir, f"{video_name}{suffix}{ext}")


def build_metadata_path(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return f"{base}.yaml"


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


def render_segmented_frames(
    frames: np.ndarray,
    segment_ids: np.ndarray,
    segment_colors: list[tuple[int, int, int]],
    border_size: int,
    progress_height: int,
    font_scale: float,
    thickness: int,
    tqdm_module=None,
) -> np.ndarray:
    frame_iter = range(len(frames))
    if tqdm_module is not None:
        frame_iter = tqdm_module(
            frame_iter,
            total=len(frames),
            desc="Rendering frames",
            unit="frame",
        )
    return np.stack(
        [
            annotate_frame(
                frame=frames[idx],
                frame_idx=idx,
                segment_ids=segment_ids,
                segment_colors=segment_colors,
                border_size=border_size,
                progress_height=progress_height,
                font_scale=font_scale,
                thickness=thickness,
            )
            for idx in frame_iter
        ],
        axis=0,
    )


def save_video_ffmpeg(video: np.ndarray, output_path: str, fps: float) -> None:
    h, w = video.shape[1:3]
    command = [
        "/usr/bin/ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-hide_banner",
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
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(video.astype(np.uint8, copy=False).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
