#!/usr/bin/env python3
"""Universal, parallelized LBM renderer.
Works with arbitrary domain dimensions (nx, ny) and arranges panels closest to a square.
Pure data rendering — no artificial geometric shapes or overlays.
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image, ImageDraw


# ---------------- Layout Optimizer ----------------
def find_best_grid(n_panels, panel_w, panel_h, border=2):
    """
    Finds (rows, cols) to arrange n_panels of size (panel_w, panel_h)
    such that the overall canvas aspect ratio is closest to 1.0 (a square).
    """
    best_rows, best_cols = 1, n_panels
    best_score = float("inf")

    for r in range(1, n_panels + 1):
        c = (n_panels + r - 1) // r  # ceil(n_panels / r)
        total_w = c * panel_w + (c - 1) * border
        total_h = r * panel_h + (r - 1) * border
        aspect = total_w / total_h
        # Distance from 1.0 in logarithmic scale
        score = abs(np.log(aspect))
        if score < best_score:
            best_score = score
            best_rows, best_cols = r, c

    return best_rows, best_cols


def extract_step(path):
    m = re.search(r"frame_(\d+)\.bin", path)
    return int(m.group(1)) if m else 0


def get_ffmpeg_binary():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def render_streamline_image(nx, ny, ux, uy, spd, max_spd):
    """Renders raw streamlines on top of velocity magnitude."""
    fig, ax = plt.subplots(figsize=(nx / 80.0, ny / 80.0), dpi=80)
    fig.patch.set_facecolor("#111111")

    ax.imshow(spd, cmap="turbo", vmin=0, vmax=max(max_spd, 1e-4), origin="lower", aspect="equal")

    # Downsample vector field for fast streamline integration
    step_s = max(1, min(nx, ny) // 100)
    xs = np.arange(0, nx, step_s)
    ys = np.arange(0, ny, step_s)
    ux_s = ux[::step_s, ::step_s]
    uy_s = uy[::step_s, ::step_s]

    max_vel = max(np.max(np.abs(ux_s)), np.max(np.abs(uy_s)))
    if max_vel > 1e-6:
        try:
            ax.streamplot(
                xs, ys, ux_s, uy_s,
                color="white", density=0.8, linewidth=0.7, arrowsize=0.6,
                integration_direction="both"
            )
        except Exception:
            pass

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[:, :, :3].copy()
    plt.close(fig)

    if img.shape[:2] != (ny, nx):
        img = np.array(Image.fromarray(img).resize((nx, ny)))
    return img


def draw_label(img_arr, text):
    """Adds a small dark HUD pill with label text."""
    pil_im = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_im)
    draw.rectangle([6, 6, 12 + len(text) * 6, 20], fill=(0, 0, 0, 200))
    draw.text((10, 8), text, fill=(255, 255, 255))
    return np.array(pil_im)


def render_single_frame(args_tuple):
    (filepath, nx, ny, field_type, max_spd, max_vort, drho_max, grid_layout) = args_tuple

    arr = np.fromfile(filepath, dtype=np.float32).reshape(3, ny, nx)
    rho, ux, uy = arr[0], arr[1], arr[2]
    step = extract_step(filepath)

    spd = np.sqrt(ux**2 + uy**2)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    pres = rho - 1.0

    cmap_spd = colormaps["turbo"]
    cmap_vort = colormaps["RdBu_r"]
    cmap_pres = colormaps["coolwarm"]

    def colorize(data, vmin, vmax, cmap):
        norm = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        return rgb[::-1]

    if field_type == "combo":
        p_spd = draw_label(colorize(spd, 0.0, max_spd, cmap_spd), f"Speed |u| (Step {step:06d})")
        p_vort = draw_label(colorize(vort, -max_vort, max_vort, cmap_vort), "Vorticity w")
        p_pres = draw_label(colorize(pres, -drho_max, drho_max, cmap_pres), "Pressure Δρ")
        p_strm = draw_label(
            render_streamline_image(nx, ny, ux, uy, spd, max_spd),
            "Streamlines"
        )
        panels = [p_spd, p_vort, p_pres, p_strm]

        rows, cols = grid_layout
        border = 2
        div_v = np.zeros((ny, border, 3), dtype=np.uint8) + 30
        grid_rows = []
        idx = 0
        for r_idx in range(rows):
            row_imgs = []
            for c_idx in range(cols):
                if idx < len(panels):
                    row_imgs.append(panels[idx])
                else:
                    row_imgs.append(np.zeros((ny, nx, 3), dtype=np.uint8))
                idx += 1
            row_combined = row_imgs[0]
            for im in row_imgs[1:]:
                row_combined = np.hstack([row_combined, div_v, im])
            grid_rows.append(row_combined)

        div_h = np.zeros((border, grid_rows[0].shape[1], 3), dtype=np.uint8) + 30
        final_img = grid_rows[0]
        for r_im in grid_rows[1:]:
            final_img = np.vstack([final_img, div_h, r_im])

        return final_img

    elif field_type == "streamlines":
        img_arr = render_streamline_image(nx, ny, ux, uy, spd, max_spd)
        return draw_label(img_arr, f"Streamlines — Step {step:06d}")

    else:
        if field_type == "speed":
            img_arr = colorize(spd, 0.0, max_spd, cmap_spd)
        elif field_type == "vorticity":
            img_arr = colorize(vort, -max_vort, max_vort, cmap_vort)
        elif field_type == "pressure":
            img_arr = colorize(pres, -drho_max, drho_max, cmap_pres)
        return draw_label(img_arr, f"{field_type.upper()} — Step {step:06d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data")
    ap.add_argument("--field", choices=["combo", "speed", "vorticity", "streamlines", "pressure"],
                    default="combo")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    info_path = os.path.join(args.dir, "info.txt")
    if not os.path.exists(info_path):
        raise SystemExit(f"Missing {info_path}. Run the solver first.")

    with open(info_path) as fh:
        parts = fh.read().split()
        nx, ny = int(parts[0]), int(parts[1])

    all_files = sorted(glob.glob(os.path.join(args.dir, "frame_*.bin")), key=extract_step)
    if not all_files:
        raise SystemExit("No frames found!")

    # Guard: Filter out stale leftover frames from runs with different resolutions
    expected_bytes = 3 * nx * ny * 4
    files = [f for f in all_files if os.path.getsize(f) == expected_bytes]
    stale_count = len(all_files) - len(files)
    if stale_count > 0:
        print(f" Ignored {stale_count} stale frames from previous runs.")

    if not files:
        raise SystemExit(f"No frames matching grid size {nx}x{ny} ({expected_bytes} bytes) found. Did you re-run the solver?")

    # Auto-optimize layout closest to square
    grid_layout = find_best_grid(4, nx, ny)
    rows, cols = grid_layout
    tot_w = cols * nx + (cols - 1) * 2
    tot_h = rows * ny + (rows - 1) * 2
    print(f" Domain {nx}x{ny} | Combo layout: {rows}x{cols} grid ({tot_w}x{tot_h}, aspect: {tot_w/tot_h:.2f})")

    # Dynamic calibration across samples
    sample_files = [files[i] for i in np.linspace(0, len(files)-1, min(10, len(files)), dtype=int)]
    max_spds, max_vorts, max_drhos = [], [], []
    for f in sample_files:
        arr = np.fromfile(f, dtype=np.float32).reshape(3, ny, nx)
        max_spds.append(np.percentile(np.sqrt(arr[1]**2 + arr[2]**2), 99.5))
        w = np.gradient(arr[2], axis=1) - np.gradient(arr[1], axis=0)
        max_vorts.append(max(abs(np.percentile(w, 1)), abs(np.percentile(w, 99))))
        max_drhos.append(np.percentile(np.abs(arr[0] - 1.0), 99.5))

    max_spd = float(np.mean(max_spds)) if max_spds else 0.1
    max_vort = float(np.mean(max_vorts)) if max_vorts else 0.05
    drho_max = float(np.mean(max_drhos)) if max_drhos else 0.01
    if max_spd < 1e-4: max_spd = 0.01
    if max_vort < 1e-4: max_vort = 0.01
    if drho_max < 1e-5: drho_max = 0.01

    tasks = [
        (f, nx, ny, args.field, max_spd, max_vort, drho_max, grid_layout)
        for f in files
    ]

    cpu_count = mp.cpu_count()
    print(f" Rendering {len(files)} frames on {cpu_count} CPU cores...")
    t0 = time.time()
    with mp.Pool(cpu_count) as pool:
        frames = pool.map(render_single_frame, tasks)
    t_render = time.time() - t0
    print(f" Rendered in {t_render:.2f}s ({len(frames)/t_render:.1f} fps)")

    h, w, _ = frames[0].shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    if (h, w) != (h_even, w_even):
        frames = [f[:h_even, :w_even] for f in frames]
        h, w = h_even, w_even

    ffmpeg_exe = get_ffmpeg_binary()
    if args.out:
        out_file = args.out
    else:
        out_file = f"animation_{args.field}.mp4" if ffmpeg_exe else f"animation_{args.field}.webp"

    # Encoding
    if ffmpeg_exe and out_file.endswith(".mp4"):
        print(f" Streaming to ffmpeg ({out_file})...")
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(args.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "20",
            out_file
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
    elif out_file.endswith(".gif"):
        print(f" Generating optimized GIF ({out_file})...")
        pil_images = [
            Image.fromarray(f).quantize(colors=128, method=Image.Quantize.FASTOCTREE)
            for f in frames
        ]
        pil_images[0].save(
            out_file, save_all=True, append_images=pil_images[1:],
            duration=int(1000 / args.fps), loop=0
        )
    else:
        if not out_file.endswith(".webp"):
            out_file = os.path.splitext(out_file)[0] + ".webp"
        print(f" Saving animated WebP ({out_file})...")
        pil_images = [Image.fromarray(f) for f in frames]
        pil_images[0].save(
            out_file, "WEBP", save_all=True, append_images=pil_images[1:],
            duration=int(1000 / args.fps), loop=0, quality=90
        )

    print(f" Done in {time.time() - t0:.2f}s! Saved to: {out_file}")


if __name__ == "__main__":
    main()