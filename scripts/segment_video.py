from __future__ import annotations

import argparse
import os

import render_common as rsvc
import segmentation_common as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a copy of a video with segmentation overlays."
    )
    sc.add_common_cli_args(
        parser,
        defaults=sc.build_cli_defaults(),
    )
    rsvc.add_render_cli_args(parser)
    return parser.parse_args()


def write_boundary_yaml(
    output_path: str,
    video_path: str,
    milestone_indices,
    fps: float,
    segmentation_config: sc.SegmentationConfig,
) -> str:
    import yaml

    metadata_path = rsvc.build_metadata_path(output_path)
    boundary_secs = [0.0]
    boundary_secs.extend(round((int(idx) + 1) / fps, 6) for idx in milestone_indices)
    metadata = {
        "video_path": os.path.abspath(video_path),
        "segmentation_algorithm": segmentation_config.common.segmentation_algorithm,
        "segmentation_params": {
            "preprocessor_name": segmentation_config.common.preprocessor_name,
            "device": segmentation_config.common.device,
            "embed_batch_size": segmentation_config.common.embed_batch_size,
            **sc.algorithm_params_dict(segmentation_config),
        },
        "boundaries_sec": boundary_secs,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)
    return metadata_path


def main():
    args = parse_args()
    segmentation_config = sc.segmentation_config_from_args(args)
    video_file = os.path.expandvars(os.path.expanduser(args.video_file))
    output_path = rsvc.build_output_path(video_file, args.render_suffix)

    if str(args.device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    import torch

    del torch
    rsvc.import_rendering_dependencies()

    import decord

    print(f"[1/5] Opening video: {video_file}", flush=True)
    vr = decord.VideoReader(video_file)
    fps = args.render_fps or float(vr.get_avg_fps() or 30.0)
    print(f"[2/5] Loading frames into memory at source fps={fps:.3f}", flush=True)
    frames = vr[:].asnumpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video as (T, H, W, 3), got {frames.shape}")
    print(
        f"[2/5] Loaded {len(frames)} frames with shape {frames.shape[1:]}",
        flush=True,
    )

    print(
        f"[3/5] Running {args.segmentation_algorithm} segmentation on {args.device} "
        f"with preprocessor={args.preprocessor_name}",
        flush=True,
    )
    cache_path = None if args.no_embedding_cache else sc.build_embedding_cache_path(
        video_file=video_file,
        preprocessor_name=args.preprocessor_name,
        embedding_cache=args.embedding_cache,
    )
    print(
        (
            f"[3/5] Loading cached embeddings: {cache_path}"
            if cache_path is not None and os.path.exists(cache_path)
            else f"[3/5] Computing embeddings in batches of {args.embed_batch_size}"
        ),
        flush=True,
    )
    rep, cache_path, loaded_from_cache = sc.load_or_compute_embeddings(
        video_file=video_file,
        frames=frames,
        preprocessor_name=segmentation_config.common.preprocessor_name,
        device=segmentation_config.common.device,
        batch_size=segmentation_config.common.embed_batch_size,
        embedding_cache=segmentation_config.common.embedding_cache,
        no_embedding_cache=segmentation_config.common.no_embedding_cache,
    )
    if cache_path is not None and not loaded_from_cache:
        print(f"[3/5] Saved embeddings to: {cache_path}", flush=True)
    indices = sc.segment_embeddings(embeddings=rep, config=segmentation_config)
    print(
        f"[3/5] Segmentation complete. Detected {len(indices)} milestones: "
        f"{list(map(int, indices))}",
        flush=True,
    )
    segment_ids = sc.assign_segments(len(frames), indices)
    segment_colors = sc.palette(int(segment_ids.max()) + 1)
    print(
        f"[4/5] Rendering overlay frames for {len(segment_colors)} segments",
        flush=True,
    )
    rendered = rsvc.render_segmented_frames(
        frames=frames,
        segment_ids=segment_ids,
        segment_colors=segment_colors,
        border_size=args.render_border_size,
        progress_height=args.render_progress_height,
        font_scale=args.render_font_scale,
        thickness=args.render_text_thickness,
        tqdm_module=sc.tqdm,
    )

    print(f"[5/5] Writing output video: {output_path}", flush=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rsvc.save_video_ffmpeg(rendered, output_path, fps=fps)
    metadata_path = write_boundary_yaml(
        output_path=output_path,
        video_path=video_file,
        milestone_indices=indices,
        fps=fps,
        segmentation_config=segmentation_config,
    )
    print(f"Saved segmented video to: {output_path}")
    print(f"Saved boundaries yaml to: {metadata_path}")
    print(f"Detected {len(segment_colors)} segments.")
    print(f"Milestone frame indices: {list(map(int, indices))}")


if __name__ == "__main__":
    main()
