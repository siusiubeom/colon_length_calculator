import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import networkx as nx


def pick_points_matplotlib(img_rgb, n_points, title):
    points = []
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(img_rgb)
    ax.set_title(
        f"{title}\n"
        f"Left-click: add point ({n_points} total) | Enter: confirm | r: reset | q/Esc: cancel"
    )
    ax.axis("off")

    scat = ax.scatter([], [], s=90, marker="x")
    line, = ax.plot([], [], linewidth=2)
    txt = ax.text(
        0.01, 0.01, "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    def redraw():
        if len(points) == 0:
            scat.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
            txt.set_text("No points yet.")
        else:
            scat.set_offsets(np.array(points))
            if len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                line.set_data(xs, ys)
            else:
                line.set_data([], [])

            msg = "\n".join([f"P{i+1}: ({p[0]:.1f}, {p[1]:.1f})" for i, p in enumerate(points)])
            if len(points) == n_points:
                msg += "\nPress Enter to confirm, or r to reset."
            else:
                msg += f"\nClick {n_points - len(points)} more point(s)."
            txt.set_text(msg)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button != 1:
            return
        if len(points) >= n_points:
            return
        points.append((float(event.xdata), float(event.ydata)))
        redraw()

    def on_key(event):
        if event.key in ("r", "R"):
            points.clear()
            redraw()
        elif event.key in ("enter", "return"):
            if len(points) == n_points:
                plt.close(fig)
        elif event.key in ("q", "escape"):
            points.clear()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    if len(points) != n_points:
        raise RuntimeError("Point selection cancelled or incomplete.")
    return points


def mm_per_pixel_from_points(p1, p2, known_mm: float) -> float:
    (x1, y1), (x2, y2) = p1, p2
    px_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    if px_dist <= 0:
        raise ValueError("Pixel distance between points is zero.")
    if known_mm <= 0:
        raise ValueError("known_mm must be > 0.")
    return known_mm / px_dist


def get_mm_per_pixel(img_bgr, known_mm=None, p1=None, p2=None, interactive=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if p1 is not None and p2 is not None:
        if known_mm is None:
            raise ValueError("--known-mm is required when using --p1/--p2.")
        return mm_per_pixel_from_points(p1, p2, float(known_mm))

    if not interactive:
        raise RuntimeError("Non-interactive mode requires --p1 --p2 and --known-mm.")

    pts = pick_points_matplotlib(img_rgb, 2, "Scale: click TWO points on the ruler (e.g., 0mm and 100mm)")
    if known_mm is None:
        known_mm = float(input("Enter real distance between the two points (mm), e.g., 100: ").strip())
    return mm_per_pixel_from_points(pts[0], pts[1], float(known_mm))


def segment_colon(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    L_blur = cv2.GaussianBlur(L, (7, 7), 0)
    _, mask = cv2.threshold(L_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    lbl = label(mask > 0)
    regions = regionprops(lbl)
    H, W = mask.shape

    best_label = None
    best_score = -1.0

    for r in regions:
        area = r.area
        if area < 800:
            continue

        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        ar = h / max(w, 1)  # aspect ratio (tallness)

        cy, cx = r.centroid
        touches_left = (minc <= 2)

        if touches_left and ar > 6.0 and area > 20000:
            continue

        if cy < H * 0.25 and area < 40000:
            continue

        dist_from_left = cx / W  # 0..1
        ar_penalty = 1.0 / (1.0 + abs(ar - 4.0))  # peak around ar~4 in your example
        score = area * (0.5 + dist_from_left) * ar_penalty

        if score > best_score:
            best_score = score
            best_label = r.label

    if best_label is None:
        raise RuntimeError("Failed to find colon component. Try different lighting or use sam.py later.")

    colon_mask = (lbl == best_label).astype(np.uint8) * 255
    return colon_mask


def skeleton_to_graph(skel_uint8):
    ys, xs = np.where(skel_uint8 > 0)
    coords = list(zip(ys, xs))
    coord_set = set(coords)

    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in coord_set:
                    w = (dy * dy + dx * dx) ** 0.5  # 1 or sqrt(2)
                    G.add_edge((y, x), (ny, nx_), weight=w)
    return G


def nearest_skel_node(skel_uint8, x, y):
    ys, xs = np.where(skel_uint8 > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty skeleton.")
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    i = np.argmin(dx * dx + dy * dy)
    return (int(ys[i]), int(xs[i]))


def skeleton_length_longest_path_px(skel_uint8):
    G = skeleton_to_graph(skel_uint8)
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if len(endpoints) < 2:
        endpoints = list(G.nodes)

    max_len = 0.0
    max_pair = None
    for i, s in enumerate(endpoints):
        lengths = nx.single_source_dijkstra_path_length(G, s, weight="weight")
        for t in endpoints[i + 1:]:
            d = lengths.get(t, None)
            if d is not None and d > max_len:
                max_len = d
                max_pair = (s, t)

    if max_pair is None:
        raise RuntimeError("Could not compute a valid skeleton path length.")
    return max_len, max_pair


def skeleton_length_between_points_px(skel_uint8, p_start_xy, p_end_xy):
    G = skeleton_to_graph(skel_uint8)

    s_node = nearest_skel_node(skel_uint8, p_start_xy[0], p_start_xy[1])
    t_node = nearest_skel_node(skel_uint8, p_end_xy[0], p_end_xy[1])

    length = nx.dijkstra_path_length(G, s_node, t_node, weight="weight")
    path = nx.dijkstra_path(G, s_node, t_node, weight="weight")
    return float(length), (s_node, t_node), path


def overlay_debug(image_bgr, colon_mask, skel_uint8, ends_rc=None, path_rc=None):
    out = image_bgr.copy()

    green = np.zeros_like(out)
    green[:, :, 1] = 180
    m = (colon_mask > 0)
    out[m] = cv2.addWeighted(out[m], 0.6, green[m], 0.4, 0)

    ys, xs = np.where(skel_uint8 > 0)
    out[ys, xs] = (0, 0, 255)

    if ends_rc is not None:
        for (r, c) in ends_rc:
            cv2.circle(out, (int(c), int(r)), 6, (255, 0, 0), -1)

    if path_rc is not None:
        for (r, c) in path_rc:
            out[int(r), int(c)] = (0, 255, 255)

    return out


def measure_colon_length(
    image_path,
    known_mm=None,
    p1=None,
    p2=None,
    interactive_scale=True,
    pick_ends=False,
    debug=False,
    save_dir=None
):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mm_per_px = get_mm_per_pixel(
        img_bgr,
        known_mm=known_mm,
        p1=p1,
        p2=p2,
        interactive=interactive_scale
    )

    colon_mask = segment_colon(img_bgr)

    bw = (colon_mask > 0)
    skel = skeletonize(bw).astype(np.uint8)

    if np.count_nonzero(skel) < 20:
        raise RuntimeError("Skeleton too small; segmentation likely failed.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ends_rc = None
    path_rc = None

    if pick_ends:
        pts = pick_points_matplotlib(
            img_rgb, 2,
            "Ends: click TWO points (1) cecum-side end, (2) rectum-side end"
        )
        length_px, (s_node, t_node), path = skeleton_length_between_points_px(skel, pts[0], pts[1])
        ends_rc = [s_node, t_node]
        path_rc = path
        method = "between_clicked_ends"
    else:
        length_px, (s_node, t_node) = skeleton_length_longest_path_px(skel)
        ends_rc = [s_node, t_node]
        method = "longest_endpoints"

    length_mm = float(length_px * mm_per_px)

    result = {
        "image": os.path.basename(image_path),
        "mm_per_pixel": float(mm_per_px),
        "length_px": float(length_px),
        "length_mm": length_mm,
        "length_cm": length_mm / 10.0,
        "method": method,
    }

    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(colon_mask, cmap="gray"); plt.title("Colon mask"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(skel, cmap="gray"); plt.title("Skeleton"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]

        mask_path = os.path.join(save_dir, f"{base}_mask.png")
        skel_path = os.path.join(save_dir, f"{base}_skel.png")
        overlay_path = os.path.join(save_dir, f"{base}_overlay.png")
        json_path = os.path.join(save_dir, f"{base}_result.json")

        cv2.imwrite(mask_path, colon_mask)
        cv2.imwrite(skel_path, (skel * 255).astype(np.uint8))

        overlay = overlay_debug(img_bgr, colon_mask, skel, ends_rc=ends_rc, path_rc=path_rc)
        cv2.imwrite(overlay_path, overlay)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def parse_point(s):
    try:
        x_str, y_str = s.split(",")
        return (float(x_str), float(y_str))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid point '{s}'. Use format x,y") from e


def main():
    parser = argparse.ArgumentParser(
        description="Colon length from image using classic CV + skeleton (no SAM)."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--known-mm", type=float, default=None,
                        help="Real mm distance between the two scale points.")
    parser.add_argument("--p1", type=parse_point, default=None, help="First scale point as x,y (pixels).")
    parser.add_argument("--p2", type=parse_point, default=None, help="Second scale point as x,y (pixels).")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Disable GUI scale picking; requires --p1 --p2 --known-mm.")
    parser.add_argument("--pick-ends", action="store_true",
                        help="Also click cecum-end and rectum-end to measure between them (recommended).")
    parser.add_argument("--debug", action="store_true", help="Show debug plots.")
    parser.add_argument("--save-dir", default=None,
                        help="If set, saves mask/skeleton/overlay + result.json into this directory.")
    args = parser.parse_args()

    interactive = not args.non_interactive

    try:
        res = measure_colon_length(
            image_path=args.image,
            known_mm=args.known_mm,
            p1=args.p1,
            p2=args.p2,
            interactive_scale=interactive,
            pick_ends=args.pick_ends,
            debug=args.debug,
            save_dir=args.save_dir
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"mm_per_pixel={res['mm_per_pixel']:.6f} | length_mm={res['length_mm']:.2f} | length_cm={res['length_cm']:.2f} | method={res['method']}")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
