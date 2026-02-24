from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib.colors import black
from reportlab.pdfgen import canvas
from scipy import ndimage
from skimage.morphology import skeletonize

Point = tuple[int, int]


def _otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))

    sum_bg = 0.0
    weight_bg = 0.0
    max_var = -1.0
    threshold = 127

    for level in range(256):
        weight_bg += hist[level]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += level * hist[level]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_var:
            max_var = between
            threshold = level

    return int(threshold)


def _load_single_page_tiff(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    with Image.open(path) as image:
        n_frames = int(getattr(image, "n_frames", 1))
        if n_frames != 1:
            raise ValueError(
                f"Input TIFF has {n_frames} pages. This script supports single-page TIFF only."
            )

        dpi = image.info.get("dpi", (300, 300))
        if not isinstance(dpi, tuple) or len(dpi) != 2:
            dpi = (300, 300)
        x_dpi = float(dpi[0]) if dpi[0] else 300.0
        y_dpi = float(dpi[1]) if dpi[1] else 300.0

        gray = np.array(image.convert("L"), dtype=np.uint8)

    return gray, (x_dpi, y_dpi)


def _binarize(gray: np.ndarray, threshold: int | None, invert: bool) -> np.ndarray:
    use_threshold = _otsu_threshold(gray) if threshold is None else int(threshold)
    if use_threshold < 0 or use_threshold > 255:
        raise ValueError("threshold must be in [0, 255].")

    if invert:
        ink = gray > use_threshold
    else:
        ink = gray <= use_threshold
    return ink


NEIGHBOR_STEPS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _edge_key(a: Point, b: Point) -> tuple[Point, Point]:
    return (a, b) if a <= b else (b, a)


def _build_skeleton_graph(mask: np.ndarray) -> dict[Point, set[Point]]:
    rows, cols = mask.shape
    points = [tuple(int(v) for v in point) for point in np.argwhere(mask)]
    point_set = set(points)
    adjacency: dict[Point, set[Point]] = defaultdict(set)

    for p in points:
        r, c = p
        for dr, dc in NEIGHBOR_STEPS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            q = (nr, nc)
            if q not in point_set:
                continue
            adjacency[p].add(q)
            adjacency[q].add(p)

    return dict(adjacency)


def _extract_chains(adjacency: dict[Point, set[Point]]) -> list[list[Point]]:
    visited_edges: set[tuple[Point, Point]] = set()
    chains: list[list[Point]] = []

    def follow(start: Point, neighbor: Point) -> list[Point]:
        chain: list[Point] = [start]
        prev = start
        curr = neighbor
        visited_edges.add(_edge_key(prev, curr))

        while True:
            chain.append(curr)
            degree = len(adjacency[curr])
            if degree != 2:
                break

            n1, n2 = tuple(adjacency[curr])
            nxt = n1 if n1 != prev else n2
            edge = _edge_key(curr, nxt)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            prev, curr = curr, nxt

        return chain

    for node, neighbors in adjacency.items():
        if len(neighbors) == 2:
            continue
        for neighbor in neighbors:
            edge = _edge_key(node, neighbor)
            if edge in visited_edges:
                continue
            chains.append(follow(node, neighbor))

    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = _edge_key(node, neighbor)
            if edge in visited_edges:
                continue
            chains.append(follow(node, neighbor))

    return [chain for chain in chains if len(chain) >= 2]


def _perp_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    segment_norm = float(np.linalg.norm(segment))
    if segment_norm == 0:
        return float(np.linalg.norm(point - start))
    px = point[0] - start[0]
    py = point[1] - start[1]
    sx = segment[0]
    sy = segment[1]
    cross_z = sx * py - sy * px
    return float(abs(cross_z) / segment_norm)


def _rdp_indices(points: np.ndarray, epsilon: float) -> list[int]:
    if points.shape[0] <= 2:
        return [0, points.shape[0] - 1]

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = -1

    for idx in range(1, points.shape[0] - 1):
        distance = _perp_distance(points[idx], start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = idx

    if max_distance <= epsilon:
        return [0, points.shape[0] - 1]

    left = _rdp_indices(points[: split_index + 1], epsilon)
    right = _rdp_indices(points[split_index:], epsilon)
    return left[:-1] + [split_index + idx for idx in right]


def _simplify_chain(chain: list[Point], epsilon: float) -> list[Point]:
    if epsilon <= 0 or len(chain) <= 2:
        return chain

    coords = np.array([(float(c), float(r)) for r, c in chain], dtype=np.float64)
    keep_indices = sorted(set(_rdp_indices(coords, epsilon)))
    simplified = [chain[index] for index in keep_indices]
    return simplified if len(simplified) >= 2 else chain


def _node_width_px(
    node: Point,
    dist_map: np.ndarray,
    width_scale: float,
    min_width_px: float,
    max_width_px: float,
) -> float:
    radius = float(dist_map[node[0], node[1]])
    width = max(min_width_px, min(max_width_px, 2.0 * radius * width_scale))
    return width


def _split_chain_by_width(
    chain: list[Point],
    dist_map: np.ndarray,
    width_scale: float,
    min_width_px: float,
    max_width_px: float,
    width_delta_px: float,
    min_run_nodes: int,
) -> list[tuple[list[Point], float]]:
    if len(chain) < 2:
        return []

    widths = [
        _node_width_px(
            node=node,
            dist_map=dist_map,
            width_scale=width_scale,
            min_width_px=min_width_px,
            max_width_px=max_width_px,
        )
        for node in chain
    ]

    runs: list[tuple[int, int]] = []
    start = 0
    index = 1

    while index < len(chain):
        should_split = abs(widths[index] - widths[index - 1]) >= width_delta_px
        long_enough = (index - start + 1) >= max(2, min_run_nodes)

        if should_split and long_enough:
            runs.append((start, index))
            start = max(0, index - 1)
        index += 1

    runs.append((start, len(chain) - 1))

    merged_runs: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        if not merged_runs:
            merged_runs.append((run_start, run_end))
            continue

        if run_end - run_start + 1 < 2:
            prev_start, prev_end = merged_runs[-1]
            merged_runs[-1] = (prev_start, run_end)
        else:
            merged_runs.append((run_start, run_end))

    split_runs: list[tuple[list[Point], float]] = []
    for run_start, run_end in merged_runs:
        nodes = chain[run_start : run_end + 1]
        if len(nodes) < 2:
            continue
        run_width = float(np.median(widths[run_start : run_end + 1]))
        split_runs.append((nodes, run_width))

    return split_runs


def _px_to_pt(value_px: float, dpi: float) -> float:
    return (value_px / dpi) * 72.0


def convert_tiff_to_vector_pdf(
    input_tiff: Path,
    output_pdf: Path,
    threshold: int | None,
    invert: bool,
    width_scale: float,
    min_width_px: float,
    max_width_px: float,
    width_delta_px: float,
    min_run_nodes: int,
    simplify_epsilon_px: float,
) -> int:
    gray, (x_dpi, y_dpi) = _load_single_page_tiff(input_tiff)
    ink = _binarize(gray, threshold=threshold, invert=invert)

    if not np.any(ink):
        raise ValueError("No black line pixels detected after thresholding.")

    skel = skeletonize(ink)
    if not np.any(skel):
        raise ValueError("Skeletonization produced no centerlines.")

    dist_map = ndimage.distance_transform_edt(ink)
    adjacency = _build_skeleton_graph(skel)
    chains = _extract_chains(adjacency)

    height_px, width_px = gray.shape
    page_w_pt = _px_to_pt(float(width_px), x_dpi)
    page_h_pt = _px_to_pt(float(height_px), y_dpi)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_pdf), pagesize=(page_w_pt, page_h_pt), pageCompression=1)
    pdf.setStrokeColor(black)
    pdf.setLineCap(1)
    pdf.setLineJoin(1)

    path_count = 0
    avg_dpi = (x_dpi + y_dpi) / 2.0

    for chain in chains:
        simplified_chain = _simplify_chain(chain, epsilon=simplify_epsilon_px)
        runs = _split_chain_by_width(
            chain=simplified_chain,
            dist_map=dist_map,
            width_scale=width_scale,
            min_width_px=min_width_px,
            max_width_px=max_width_px,
            width_delta_px=width_delta_px,
            min_run_nodes=min_run_nodes,
        )

        for run_nodes, run_width in runs:
            if len(run_nodes) < 2:
                continue

            pdf_path = pdf.beginPath()
            first_r, first_c = run_nodes[0]
            first_x = _px_to_pt(float(first_c), x_dpi)
            first_y = page_h_pt - _px_to_pt(float(first_r), y_dpi)
            pdf_path.moveTo(first_x, first_y)

            for r, c in run_nodes[1:]:
                x = _px_to_pt(float(c), x_dpi)
                y = page_h_pt - _px_to_pt(float(r), y_dpi)
                pdf_path.lineTo(x, y)

            pdf.setLineWidth(_px_to_pt(run_width, avg_dpi))
            pdf.drawPath(pdf_path, stroke=1, fill=0)
            path_count += 1

    pdf.showPage()
    pdf.save()

    return path_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a black-and-white raster TIFF into a stroke-only vector PDF "
            "using skeleton centerlines and variable line width."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input single-page TIFF.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output PDF.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Threshold 0..255. If omitted, Otsu threshold is used.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Treat bright pixels as line work if your TIFF has white lines on dark background.",
    )
    parser.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to local line width derived from distance transform.",
    )
    parser.add_argument(
        "--min-width-px",
        type=float,
        default=0.75,
        help="Minimum output line width in source-image pixels.",
    )
    parser.add_argument(
        "--max-width-px",
        type=float,
        default=30.0,
        help="Maximum output line width in source-image pixels.",
    )
    parser.add_argument(
        "--width-delta-px",
        type=float,
        default=2.5,
        help="Split connected paths when local width change between adjacent nodes reaches this value.",
    )
    parser.add_argument(
        "--min-run-nodes",
        type=int,
        default=40,
        help="Minimum nodes in a run before a width-change split is allowed.",
    )
    parser.add_argument(
        "--simplify-epsilon-px",
        type=float,
        default=0.8,
        help="Mild polyline simplification tolerance in pixels; 0 disables simplification.",
    )
    parser.add_argument(
        "--allow-large-images",
        action="store_true",
        help="Disable Pillow decompression bomb limit for very large TIFFs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_large_images:
        Image.MAX_IMAGE_PIXELS = None

    if args.width_scale <= 0:
        raise ValueError("--width-scale must be > 0")
    if args.min_width_px <= 0:
        raise ValueError("--min-width-px must be > 0")
    if args.max_width_px <= 0:
        raise ValueError("--max-width-px must be > 0")
    if args.min_width_px > args.max_width_px:
        raise ValueError("--min-width-px cannot be greater than --max-width-px")
    if args.width_delta_px <= 0:
        raise ValueError("--width-delta-px must be > 0")
    if args.min_run_nodes < 2:
        raise ValueError("--min-run-nodes must be >= 2")
    if args.simplify_epsilon_px < 0:
        raise ValueError("--simplify-epsilon-px must be >= 0")

    path_runs = convert_tiff_to_vector_pdf(
        input_tiff=args.input,
        output_pdf=args.output,
        threshold=args.threshold,
        invert=args.invert,
        width_scale=args.width_scale,
        min_width_px=args.min_width_px,
        max_width_px=args.max_width_px,
        width_delta_px=args.width_delta_px,
        min_run_nodes=args.min_run_nodes,
        simplify_epsilon_px=args.simplify_epsilon_px,
    )
    print(f"Generated {args.output} with {path_runs} connected stroked path runs.")


if __name__ == "__main__":
    main()
