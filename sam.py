import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
import networkx as nx

from segment_anything import sam_model_registry, SamPredictor


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
            txt.set_text("No points yet.")
        else:
            scat.set_offsets(np.array(points))
            msg = "\n".join([f"P{i+1}: ({p[0]:.1f}, {p[1]:.1f})" for i, p in enumerate(points)])
            if len(points) == n_points:
                msg += "\nPress Enter to confirm, or r to reset."
            else:
                msg += f"\nClick {n_points - len(points)} more point(s)."
            txt.set_text(msg)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
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


def nearest_skel_node_from_coords(ys, xs, x, y):
    if len(xs) == 0:
        raise RuntimeError("Empty skeleton.")
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    i = int(np.argmin(dx * dx + dy * dy))
    return (int(ys[i]), int(xs[i]))  # (row, col)


def skeleton_length_between_points_px(skel_uint8, p_start_xy, p_end_xy):
    G = skeleton_to_graph(skel_uint8)
    ys, xs = np.where(skel_uint8 > 0)

    s_node = nearest_skel_node_from_coords(ys, xs, p_start_xy[0], p_start_xy[1])
    t_node = nearest_skel_node_from_coords(ys, xs, p_end_xy[0], p_end_xy[1])

    length = nx.dijkstra_path_length(G, s_node, t_node, weight="weight")
    path = nx.dijkstra_path(G, s_node, t_node, weight="weight")
    return float(length), (s_node, t_node), path


def sam_segment_from_points(image_rgb, checkpoint_path, model_type, points_xy, labels):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    point_coords = np.array(points_xy, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_i = int(np.argmax(scores))
    return masks[best_i].astype(bool), float(scores[best_i])


def overlay_debug(image_bgr, mask_bool, skel_uint8=None, ends_rc=None, path_rc=None):
    out = image_bgr.copy()

    # mask overlay (green)
    green = np.zeros_like(out)
    green[:, :, 1] = 180
    out[mask_bool] = cv2.addWeighted(out[mask_bool], 0.6, green[mask_bool], 0.4, 0)

    # skeleton (red)
    if skel_uint8 is not None:
        ys, xs = np.where(skel_uint8 > 0)
        out[ys, xs] = (0, 0, 255)

    # endpoints (blue)
    if ends_rc is not None:
        for (r, c) in ends_rc:
            cv2.circle(out, (int(c), int(r)), 6, (255, 0, 0), -1)

    # measured path (yellow)
    if path_rc is not None:
        for (r, c) in path_rc:
            out[int(r), int(c)] = (0, 255, 255)

    return out


def measure_colon_length_sam_between_ends(
    image_path,
    known_mm,
    checkpoint,
    model_type="vit_b",
    save_dir=None,
    debug=False,
    min_object_size=1000
):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pts = pick_points_matplotlib(
        img_rgb,
        4,
        "Click 4 points: (1) scale P1, (2) scale P2, (3) cecum start, (4) rectum end"
    )
    scale_p1, scale_p2, end1, end2 = pts

    mm_per_px = mm_per_pixel_from_points(scale_p1, scale_p2, float(known_mm))

    mask_bool, sam_score = sam_segment_from_points(
        img_rgb,
        checkpoint_path=checkpoint,
        model_type=model_type,
        points_xy=[end1, end2],
        labels=[1, 1]
    )

    # clean + skeleton
    mask_bool = remove_small_objects(mask_bool, min_size=int(min_object_size))
    skel = skeletonize(mask_bool).astype(np.uint8)
    if np.count_nonzero(skel) < 20:
        raise RuntimeError("Skeleton too small; SAM mask may be wrong. Try clicking ends more on the tissue.")

    length_px, (s_node, t_node), path = skeleton_length_between_points_px(skel, end1, end2)
    length_mm = float(length_px * mm_per_px)

    result = {
        "image": os.path.basename(image_path),
        "mm_per_pixel": float(mm_per_px),
        "sam_model_type": model_type,
        "sam_checkpoint": os.path.basename(checkpoint),
        "sam_score": float(sam_score),
        "length_px": float(length_px),
        "length_mm": float(length_mm),
        "length_cm": float(length_mm / 10.0),
        "method": "sam_mask_skeleton_between_clicked_ends",
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]

        mask_path = os.path.join(save_dir, f"{base}_sam_mask.png")
        skel_path = os.path.join(save_dir, f"{base}_sam_skel.png")
        overlay_path = os.path.join(save_dir, f"{base}_sam_overlay.png")
        json_path = os.path.join(save_dir, f"{base}_sam_result.json")

        cv2.imwrite(mask_path, (mask_bool.astype(np.uint8) * 255))
        cv2.imwrite(skel_path, (skel.astype(np.uint8) * 255))

        overlay = overlay_debug(img_bgr, mask_bool, skel_uint8=skel, ends_rc=[s_node, t_node], path_rc=path)
        cv2.imwrite(overlay_path, overlay)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if debug:
        overlay = overlay_debug(img_bgr, mask_bool, skel_uint8=skel, ends_rc=[s_node, t_node], path_rc=path)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(mask_bool, cmap="gray"); plt.title(f"SAM mask (score={sam_score:.3f})"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("Measured path overlay"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Colon length using SAM segmentation + skeleton distance between clicked ends."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--known-mm", type=float, required=True,
                        help="Real mm distance between the two scale points.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SAM checkpoint (.pth).")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type (must match the checkpoint).")
    parser.add_argument("--min-object-size", type=int, default=1000,
                        help="Remove tiny mask blobs smaller than this (pixels).")
    parser.add_argument("--debug", action="store_true", help="Show debug plots.")
    parser.add_argument("--save-dir", default=None,
                        help="If set, saves mask/skeleton/overlay + result.json into this directory.")
    args = parser.parse_args()

    try:
        res = measure_colon_length_sam_between_ends(
            image_path=args.image,
            known_mm=args.known_mm,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            save_dir=args.save_dir,
            debug=args.debug,
            min_object_size=args.min_object_size
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"mm_per_pixel={res['mm_per_pixel']:.6f} | "
        f"length_mm={res['length_mm']:.2f} | length_cm={res['length_cm']:.2f} | "
        f"method={res['method']}"
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
