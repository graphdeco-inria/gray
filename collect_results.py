import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated, Optional
from dataclasses import dataclass, field
import tyro
from tyro.conf import arg, Positional


@dataclass
class RenderCLI:
    directory: Positional[str]  # * Super directory containing one or more trained models
    iteration: Annotated[Optional[int], arg(aliases=["-t"])] = None


cli = tyro.cli(RenderCLI)

# * Collect all results
results = []
for scene in sorted(os.listdir(cli.directory)):
    scene_dir = os.path.join(cli.directory, scene)
    results_path = os.path.join(scene_dir, "results.json")
    time_path = os.path.join(scene_dir, "time.csv")
    fps_path = os.path.join(scene_dir, "fps.csv")
    gaussians_path = os.path.join(scene_dir, "num_gaussians.csv")
    traversal_stats_path = os.path.join(scene_dir, "traversal_stats.csv")
    geometry_stats_path = os.path.join(scene_dir, "geometry_stats.csv")

    # * Read JSON metrics
    with open(results_path, "r") as f:
        data = json.load(f)
    if cli.iteration is None:
        test_path = os.path.join(scene_dir, "test")
        iter_options = []
        for item in os.listdir(test_path):
            if item.isdigit():
                iter_options.append(int(item))
        if not iter_options:
            continue
        target_iter_str = f"{max(iter_options):05d}"
    else:
        target_iter_str = str(cli.iteration)
        if target_iter_str not in data:
            continue
    iter_data = data[target_iter_str]
    if "PSNR" not in iter_data or "SSIM" not in iter_data or "LPIPS" not in iter_data:
        continue
    test_psnr = float(iter_data["PSNR"])
    test_ssim = float(iter_data["SSIM"])
    test_lpips = float(iter_data["LPIPS"])

    # * Read training time
    time_df = pd.read_csv(time_path, sep=r"\s+")
    time_row = time_df[time_df["iteration"] == int(target_iter_str)]
    assert not time_row.empty
    elapsed_str = str(time_row["elapsed_time"].values[0])
    time_obj = datetime.strptime(elapsed_str, "%H:%M:%S")
    elapsed_sec = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    elapsed_formatted = str(timedelta(seconds=round(elapsed_sec)))

    # * Read FPS
    with open(fps_path, "r") as f:
        fps_content = f.read().strip()
        fps = float(fps_content)

    # * Read Gaussian counts
    gaussians_df = pd.read_csv(gaussians_path, sep=r"\s+", names=["iteration", "num_gaussians"])
    start_gaussians = int(gaussians_df.iloc[1]["num_gaussians"])  # First row
    final_gaussians = int(gaussians_df.iloc[-1]["num_gaussians"])  # Last row

    # * Read init/final scales from stats.csv (if present)
    init_scale_str = "-"
    final_scale_str = "-"
    if os.path.exists(geometry_stats_path):
        try:
            with open(geometry_stats_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # Filter out header lines
            data_lines = [ln for ln in lines if not ln.lower().startswith("iteration")]
            if data_lines:
                # Tokenize by whitespace: iteration opacity scale anisotropy
                def parse_scale_from_line(ln: str) -> str:
                    parts = ln.split()
                    if len(parts) < 3:
                        return "-"
                    return parts[2]

                init_scale_str = parse_scale_from_line(data_lines[0])
                final_scale_str = parse_scale_from_line(data_lines[-1])
        except Exception:
            # Leave defaults if parsing fails
            pass

    # * Read traversal stats to compute % gaussians skipped
    skipped_pct_str = "-"
    if os.path.exists(traversal_stats_path):
        try:
            with open(traversal_stats_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # Filter out header lines
            data_lines = [ln for ln in lines if not ln.lower().startswith("iteration")]
            if data_lines:
                # Last iteration: iteration num_hit_per_ray num_accum_per_ray
                parts = data_lines[-1].split()
                if len(parts) >= 3:
                    num_hit = float(parts[1])
                    num_accum = float(parts[2])
                    if num_hit > 0:
                        skipped_pct = (1.0 - num_accum / num_hit) * 100.0
                        skipped_pct_str = f"{skipped_pct:.2f}"
        except Exception:
            pass

    # * Append data for scene
    results.append(
        (
            scene,
            test_psnr,
            test_ssim,
            test_lpips,
            elapsed_formatted,
            elapsed_sec,
            fps,
            start_gaussians,
            final_gaussians,
            init_scale_str,
            final_scale_str,
            skipped_pct_str,
        )
    )

assert len(results) == 13

# * Exit if no results found
if not results:
    print(f"No matching data found in {cli.directory}")
    sys.exit(0)

# * Compute averages
avg_psnr = sum(r[1] for r in results) / len(results)
avg_ssim = sum(r[2] for r in results) / len(results)
avg_lpips = sum(r[3] for r in results) / len(results)
avg_time = sum(r[5] for r in results) / len(results)
avg_fps = sum(r[6] for r in results) / len(results)
avg_start_gaussians = sum(r[7] for r in results) / len(results)
avg_final_gaussians = sum(r[8] for r in results) / len(results)
avg_time_formatted = str(timedelta(seconds=round(avg_time)))


# * Compute average init/final scales (mean±std) across scenes
def _parse_scale(scale_str: str):
    try:
        if isinstance(scale_str, str) and "±" in scale_str:
            mean_str, std_str = scale_str.split("±", 1)
            return float(mean_str), float(std_str)
    except Exception:
        pass
    return None


init_pairs = []
final_pairs = []
skipped_pcts = []
for r in results:
    init_parsed = _parse_scale(r[9])
    final_parsed = _parse_scale(r[10])
    if init_parsed:
        init_pairs.append(init_parsed)
    if final_parsed:
        final_pairs.append(final_parsed)
    # Parse skipped percentage
    try:
        if r[11] != "-":
            skipped_pcts.append(float(r[11]))
    except Exception:
        pass

if init_pairs:
    avg_init_mean = sum(p[0] for p in init_pairs) / len(init_pairs)
    avg_init_std = sum(p[1] for p in init_pairs) / len(init_pairs)
    avg_init_scale_str = f"{avg_init_mean:.4f}±{avg_init_std:.4f}"
else:
    avg_init_scale_str = "-"

if final_pairs:
    avg_final_mean = sum(p[0] for p in final_pairs) / len(final_pairs)
    avg_final_std = sum(p[1] for p in final_pairs) / len(final_pairs)
    avg_final_scale_str = f"{avg_final_mean:.4f}±{avg_final_std:.4f}"
else:
    avg_final_scale_str = "-"

if skipped_pcts:
    avg_skipped_pct = sum(skipped_pcts) / len(skipped_pcts)
    avg_skipped_pct_str = f"{avg_skipped_pct:.2f}"
else:
    avg_skipped_pct_str = "-"

# * Print results
print()
print(cli.directory)
print(
    f"Scene      PSNR    SSIM  LPIPS     Time     FPS  Start#G  Final#G  InitScale        FinalScale       Skipped%"
)
print("-" * 106)
for (
    scene,
    psnr,
    ssim,
    lpips,
    time_str,
    _,
    fps,
    start_g,
    final_g,
    init_scale,
    final_scale,
    skipped_pct,
) in results:
    print(
        f"{scene:<10} {psnr:5.2f}  {ssim:5.3f}  {lpips:5.3f}  {time_str}  {fps:6.2f} {start_g:8.0f} {final_g:8.0f}  {init_scale:<15} {final_scale:<15}  {skipped_pct:>7}"
    )
print("-" * 106)
print(
    f"{'Average':<10} {avg_psnr:5.2f}  {avg_ssim:5.3f}  {avg_lpips:5.3f}  {avg_time_formatted}  {avg_fps:6.2f} {avg_start_gaussians:8.0f} {avg_final_gaussians:8.0f}  {avg_init_scale_str:<15} {avg_final_scale_str:<15}  {avg_skipped_pct_str:>7}"
)
