"""
TIFF Raster to Vector PDF Converter
https://github.com/jgrimard/convert_tiff_to_vector_pdf
====================================
This script converts a black-and-white raster TIFF image into a stroke-only
vector PDF. The high-level pipeline is:

  1. Load a single-page TIFF and convert it to grayscale.
  2. Binarize (threshold) the image to separate ink (line work) from background.
  3. Skeletonize the binary image to find 1-pixel-wide centerlines.
  4. Build a graph from the skeleton pixels and extract chains (polylines).
  5. Simplify chains using the Ramer-Douglas-Peucker algorithm to reduce points.
  6. Use a distance-transform map to estimate the original line width at each
     skeleton point, and split chains where the width changes significantly.
  7. Render each chain segment as a stroked path in a PDF, with the line width
     set to match the original raster thickness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

# NumPy: array operations for image data and math
import numpy as np
# Pillow (PIL): reading TIFF images and extracting DPI metadata
from PIL import Image
# ReportLab: generating PDF files with vector drawing commands
from reportlab.lib.colors import black
from reportlab.pdfgen import canvas
# SciPy ndimage: distance transform to measure line thickness
from scipy import ndimage
# scikit-image: morphological skeletonization (thin lines to 1-pixel centerlines)
from skimage.morphology import skeletonize

# Type alias: a point is a (row, column) tuple in pixel coordinates
Point = tuple[int, int]


# ---------------------------------------------------------------------------
# Otsu's Method for Automatic Thresholding
# ---------------------------------------------------------------------------
def _otsu_threshold(gray: np.ndarray) -> int:
    """
    Compute the optimal binarization threshold using Otsu's method.

    Otsu's algorithm finds the threshold that minimizes intra-class variance
    (equivalently, maximizes inter-class / "between-class" variance) between
    two groups of pixels: foreground and background.

    Parameters:
        gray: 2-D uint8 grayscale image array.

    Returns:
        The optimal threshold value (0-255).
    """
    # Build a 256-bin histogram of pixel intensities (0 = black, 255 = white)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size  # total number of pixels in the image
    # Weighted sum of all intensity levels (used to compute class means)
    sum_total = float(np.dot(np.arange(256), hist))

    sum_bg = 0.0       # cumulative intensity sum for the background class
    weight_bg = 0.0     # cumulative pixel count for the background class
    max_var = -1.0      # best between-class variance found so far
    threshold = 127     # default threshold (will be overwritten)

    # Sweep through every possible threshold level 0..255
    for level in range(256):
        weight_bg += hist[level]        # add this level's pixels to background
        if weight_bg == 0:
            continue                    # no background pixels yet, skip
        weight_fg = total - weight_bg   # remaining pixels belong to foreground
        if weight_fg == 0:
            break                       # all pixels are in background, done

        sum_bg += level * hist[level]          # update background intensity sum
        mean_bg = sum_bg / weight_bg           # mean intensity of background
        mean_fg = (sum_total - sum_bg) / weight_fg  # mean intensity of foreground

        # Between-class variance: larger values mean better class separation
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_var:
            max_var = between
            threshold = level  # record the best threshold so far

    return int(threshold)


# ---------------------------------------------------------------------------
# TIFF Loading
# ---------------------------------------------------------------------------
def _load_single_page_tiff(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Open a TIFF file, verify it has exactly one page, read its DPI metadata,
    and convert the image to an 8-bit grayscale NumPy array.

    Parameters:
        path: Filesystem path to the input TIFF file.

    Returns:
        A tuple of (grayscale_array, (x_dpi, y_dpi)).

    Raises:
        ValueError: If the TIFF contains more than one page/frame.
    """
    with Image.open(path) as image:
        # Check that the TIFF has only one page (multi-page TIFFs are not supported)
        n_frames = int(getattr(image, "n_frames", 1))
        if n_frames != 1:
            raise ValueError(
                f"Input TIFF has {n_frames} pages. This script supports single-page TIFF only."
            )

        # Read DPI from TIFF metadata; default to 300 if missing or malformed
        dpi = image.info.get("dpi", (300, 300))
        if not isinstance(dpi, tuple) or len(dpi) != 2:
            dpi = (300, 300)
        x_dpi = float(dpi[0]) if dpi[0] else 300.0
        y_dpi = float(dpi[1]) if dpi[1] else 300.0

        # Convert image to 8-bit grayscale ("L" = luminance mode)
        gray = np.array(image.convert("L"), dtype=np.uint8)

    return gray, (x_dpi, y_dpi)


# ---------------------------------------------------------------------------
# Binarization (converting grayscale to a True/False ink mask)
# ---------------------------------------------------------------------------
def _binarize(gray: np.ndarray, threshold: int | None, invert: bool) -> np.ndarray:
    """
    Convert a grayscale image into a boolean "ink" mask where True = line work.

    Parameters:
        gray:      8-bit grayscale image array.
        threshold: Fixed threshold (0-255). If None, Otsu's method is used to
                   pick an optimal value automatically.
        invert:    When False (default), dark pixels (<=threshold) are treated
                   as ink. When True, bright pixels (>threshold) are ink, which
                   is useful for white-on-black images.

    Returns:
        A boolean 2-D array where True means "ink / line work".
    """
    # Use the supplied threshold, or compute one automatically via Otsu
    use_threshold = _otsu_threshold(gray) if threshold is None else int(threshold)
    if use_threshold < 0 or use_threshold > 255:
        raise ValueError("threshold must be in [0, 255].")

    if invert:
        # For images with white lines on a dark background
        ink = gray > use_threshold
    else:
        # Normal case: dark lines on a light background
        ink = gray <= use_threshold
    return ink


# The 8 directions to check for neighboring skeleton pixels:
# up, down, left, right, and the four diagonals.
NEIGHBOR_STEPS = [
    (-1, 0),   # up
    (1, 0),    # down
    (0, -1),   # left
    (0, 1),    # right
    (-1, -1),  # upper-left
    (-1, 1),   # upper-right
    (1, -1),   # lower-left
    (1, 1),    # lower-right
]


# ---------------------------------------------------------------------------
# Skeleton Graph Construction
# ---------------------------------------------------------------------------
def _edge_key(a: Point, b: Point) -> tuple[Point, Point]:
    """
    Create a canonical (order-independent) key for an edge between two points.
    This ensures that the edge (A, B) and (B, A) map to the same key, which
    prevents processing the same edge twice.
    """
    return (a, b) if a <= b else (b, a)


def _build_skeleton_graph(mask: np.ndarray) -> dict[Point, set[Point]]:
    """
    Convert a boolean skeleton mask into an adjacency graph.

    Every True pixel becomes a node.  Two nodes are connected by an edge if
    they are 8-connected neighbors (horizontally, vertically, or diagonally
    adjacent).

    Parameters:
        mask: Boolean 2-D array where True = skeleton pixel.

    Returns:
        A dict mapping each skeleton point to the set of its neighbor points.
    """
    rows, cols = mask.shape

    # Collect all True pixel positions as (row, col) tuples
    points = [tuple(int(v) for v in point) for point in np.argwhere(mask)]
    # A set copy for O(1) membership checking
    point_set = set(points)
    adjacency: dict[Point, set[Point]] = defaultdict(set)

    # For every skeleton pixel, look at its 8 neighbors and record edges
    for p in points:
        r, c = p
        for dr, dc in NEIGHBOR_STEPS:
            nr, nc = r + dr, c + dc
            # Skip neighbors that fall outside the image bounds
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            q = (nr, nc)
            # Only connect to other skeleton pixels
            if q not in point_set:
                continue
            # Add an undirected edge between p and q
            adjacency[p].add(q)
            adjacency[q].add(p)

    return dict(adjacency)


# ---------------------------------------------------------------------------
# Chain Extraction (traversing the skeleton graph into polylines)
# ---------------------------------------------------------------------------
def _extract_chains(adjacency: dict[Point, set[Point]]) -> list[list[Point]]:
    """
    Walk the skeleton graph and extract ordered chains (polylines) of
    connected pixels.

    Strategy:
      - First, start walks from "junction" nodes (degree != 2), which are
        endpoints (degree 1) or intersections (degree >= 3). These are natural
        starting points because they are where lines end or branch.
      - Then, pick up any remaining un-visited edges. These belong to closed
        loops (all nodes have degree 2, so no junction was found).

    Each chain is an ordered list of (row, col) points representing a
    continuous path through the skeleton.

    Returns:
        A list of chains, where each chain has at least 2 points.
    """
    visited_edges: set[tuple[Point, Point]] = set()
    chains: list[list[Point]] = []

    def follow(start: Point, neighbor: Point) -> list[Point]:
        """
        Follow a path from 'start' through 'neighbor', continuing along
        degree-2 nodes (simple pass-through points) until reaching an
        endpoint, junction, or already-visited edge.
        """
        chain: list[Point] = [start]
        prev = start
        curr = neighbor
        visited_edges.add(_edge_key(prev, curr))

        while True:
            chain.append(curr)
            degree = len(adjacency[curr])
            # Stop if we hit a junction (degree != 2): it's an endpoint or branch
            if degree != 2:
                break

            # Degree-2 node: exactly two neighbors. Pick the one we didn't come from.
            n1, n2 = tuple(adjacency[curr])
            nxt = n1 if n1 != prev else n2
            edge = _edge_key(curr, nxt)
            # Stop if this edge was already traversed (prevents infinite loops)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            prev, curr = curr, nxt

        return chain

    # --- Pass 1: Start from junction/endpoint nodes (degree != 2) ---
    # These are the most natural places to begin chains because they sit at
    # the ends or branching points of the skeleton.
    for node, neighbors in adjacency.items():
        if len(neighbors) == 2:
            continue  # skip simple pass-through nodes for now
        for neighbor in neighbors:
            edge = _edge_key(node, neighbor)
            if edge in visited_edges:
                continue
            chains.append(follow(node, neighbor))

    # --- Pass 2: Collect remaining edges (closed loops with no junctions) ---
    # In a closed loop, every node has degree 2, so Pass 1 would skip them.
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = _edge_key(node, neighbor)
            if edge in visited_edges:
                continue
            chains.append(follow(node, neighbor))

    # Only keep chains that have at least 2 points (a single point can't form a stroke)
    return [chain for chain in chains if len(chain) >= 2]


# ---------------------------------------------------------------------------
# Ramer-Douglas-Peucker (RDP) Polyline Simplification
# ---------------------------------------------------------------------------
def _perp_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """
    Compute the perpendicular distance from 'point' to the line segment
    defined by 'start' and 'end'.

    Uses the cross-product formula:
        distance = |cross(segment, point-start)| / |segment|

    If start == end (zero-length segment), returns the Euclidean distance
    from the point to start.
    """
    segment = end - start
    segment_norm = float(np.linalg.norm(segment))
    if segment_norm == 0:
        return float(np.linalg.norm(point - start))
    # Components of the vector from start to the point
    px = point[0] - start[0]
    py = point[1] - start[1]
    # Components of the segment vector
    sx = segment[0]
    sy = segment[1]
    # Z-component of the 2-D cross product gives signed area of parallelogram
    cross_z = sx * py - sy * px
    return float(abs(cross_z) / segment_norm)


def _rdp_indices(points: np.ndarray, epsilon: float) -> list[int]:
    """
    Recursive implementation of the Ramer-Douglas-Peucker algorithm.

    Given an ordered array of 2-D points, this identifies the indices of
    points to *keep* such that no omitted point is farther than 'epsilon'
    from the simplified polyline.

    The algorithm works by:
      1. Drawing a straight line from the first to the last point.
      2. Finding the interior point with the greatest perpendicular distance
         to that line.
      3. If that distance exceeds epsilon, recursively simplify the left and
         right halves (split at the farthest point).
      4. If it doesn't exceed epsilon, all interior points can be dropped
         (only keep the first and last).

    Parameters:
        points:  (N, 2) float array of coordinates.
        epsilon: Maximum allowed perpendicular deviation in pixels.

    Returns:
        Sorted list of indices into 'points' that should be kept.
    """
    # Base case: two or fewer points can't be simplified further
    if points.shape[0] <= 2:
        return [0, points.shape[0] - 1]

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = -1

    # Find the point with the largest perpendicular distance from the line
    for idx in range(1, points.shape[0] - 1):
        distance = _perp_distance(points[idx], start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = idx

    # If the farthest point is within tolerance, drop all interior points
    if max_distance <= epsilon:
        return [0, points.shape[0] - 1]

    # Otherwise, recurse on the left half (start..split) and right half (split..end)
    left = _rdp_indices(points[: split_index + 1], epsilon)
    right = _rdp_indices(points[split_index:], epsilon)
    # Merge the two halves, adjusting right-half indices back to global positions
    return left[:-1] + [split_index + idx for idx in right]


def _simplify_chain(chain: list[Point], epsilon: float) -> list[Point]:
    """
    Apply Ramer-Douglas-Peucker simplification to a chain of (row, col) points.

    This reduces the number of points in a polyline while preserving its shape
    within 'epsilon' pixels of tolerance, making the output PDF smaller.

    Parameters:
        chain:   Ordered list of (row, col) skeleton points.
        epsilon: Maximum allowed deviation in pixels. 0 = no simplification.

    Returns:
        A simplified list of points (at least 2 points long).
    """
    if epsilon <= 0 or len(chain) <= 2:
        return chain

    # Convert (row, col) to (x, y) = (col, row) for geometric calculation
    coords = np.array([(float(c), float(r)) for r, c in chain], dtype=np.float64)
    keep_indices = sorted(set(_rdp_indices(coords, epsilon)))
    simplified = [chain[index] for index in keep_indices]
    return simplified if len(simplified) >= 2 else chain


# ---------------------------------------------------------------------------
# Line-Width Estimation and Chain Splitting by Width
# ---------------------------------------------------------------------------
def _node_width_px(
    node: Point,
    dist_map: np.ndarray,
    width_scale: float,
    min_width_px: float,
    max_width_px: float,
) -> float:
    """
    Estimate the original raster line width at a given skeleton point.

    The distance transform gives the distance from each ink pixel to the
    nearest background pixel.  At a skeleton (centerline) pixel, that distance
    equals the radius of the inscribed circle—i.e., half the local line
    thickness.  We double it to get the full width, then apply the user's
    scaling factor and clamp to [min_width_px, max_width_px].

    Parameters:
        node:          (row, col) skeleton point.
        dist_map:      Distance-transform array of the binary ink mask.
        width_scale:   User multiplier for the width.
        min_width_px:  Floor for the computed width.
        max_width_px:  Ceiling for the computed width.

    Returns:
        Estimated line width in source-image pixels.
    """
    # Look up the distance-transform value at this skeleton pixel
    radius = float(dist_map[node[0], node[1]])
    # Diameter = 2 * radius, scaled and clamped
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
    """
    Split a single chain into sub-segments ("runs") wherever the estimated
    line width changes abruptly.

    PDF stroked paths have a single uniform width, so we need separate path
    segments for parts of the skeleton that pass through thicker vs. thinner
    areas of the original raster artwork.

    Algorithm:
      1. Compute the estimated width at every node in the chain.
      2. Walk along the chain; when the width delta between adjacent nodes
         reaches 'width_delta_px' and the current run is long enough
         (>= min_run_nodes), split there.
      3. Merge any very short runs (< 2 nodes) back into their predecessor.
      4. For each final run, compute the median width to use as the PDF
         stroke width for that segment.

    Parameters:
        chain:          Ordered list of (row, col) skeleton points.
        dist_map:       Distance-transform of the binary ink mask.
        width_scale:    Multiplier for computed widths.
        min_width_px:   Minimum allowed width (pixels).
        max_width_px:   Maximum allowed width (pixels).
        width_delta_px: Width change threshold to trigger a split.
        min_run_nodes:  Minimum run length before a split is permitted.

    Returns:
        A list of (sub_chain, median_width) tuples, one per run.
    """
    if len(chain) < 2:
        return []

    # Step 1: Compute estimated width at every node in the chain
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

    # Step 2: Walk the chain and identify split points based on width changes
    runs: list[tuple[int, int]] = []  # each run is (start_index, end_index)
    start = 0
    index = 1

    while index < len(chain):
        # Check if the width changed significantly from the previous node
        should_split = abs(widths[index] - widths[index - 1]) >= width_delta_px
        # Only split if the current run has accumulated enough nodes
        long_enough = (index - start + 1) >= max(2, min_run_nodes)

        if should_split and long_enough:
            runs.append((start, index))
            # Overlap by one node so adjacent runs share their boundary point
            start = max(0, index - 1)
        index += 1

    # Final run extends to the end of the chain
    runs.append((start, len(chain) - 1))

    # Step 3: Merge very short runs (< 2 nodes) into the preceding run
    merged_runs: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        if not merged_runs:
            merged_runs.append((run_start, run_end))
            continue

        if run_end - run_start + 1 < 2:
            # Too short to stand alone — extend the previous run to include it
            prev_start, prev_end = merged_runs[-1]
            merged_runs[-1] = (prev_start, run_end)
        else:
            merged_runs.append((run_start, run_end))

    # Step 4: Build the output list with sub-chains and their median widths
    split_runs: list[tuple[list[Point], float]] = []
    for run_start, run_end in merged_runs:
        nodes = chain[run_start : run_end + 1]
        if len(nodes) < 2:
            continue
        # Use the median width of all nodes in this run for a stable estimate
        run_width = float(np.median(widths[run_start : run_end + 1]))
        split_runs.append((nodes, run_width))

    return split_runs


# ---------------------------------------------------------------------------
# Progress Reporting
# ---------------------------------------------------------------------------
def _print_progress(
    current: int,
    total: int,
    start_time: float,
    bar_length: int = 40,
) -> None:
    """
    Print a single-line progress bar with percentage and estimated time
    remaining.  Overwrites itself on the same console line using \\r.
    """
    fraction = current / total if total else 1.0
    filled = int(bar_length * fraction)
    bar = "#" * filled + "-" * (bar_length - filled)

    elapsed = time.time() - start_time
    if fraction > 0:
        eta = elapsed / fraction - elapsed
    else:
        eta = 0.0

    mins, secs = divmod(int(eta), 60)
    eta_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    sys.stdout.write(
        f"\r  [{bar}] {current}/{total} ({fraction:.0%})  ETA {eta_str}   "
    )
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _step(label: str) -> float:
    """Print a step label and return the current time for later timing."""
    print(label, end="", flush=True)
    return time.time()


def _done(start: float) -> None:
    """Print elapsed time since *start*."""
    elapsed = time.time() - start
    print(f" done ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Unit Conversion Helper
# ---------------------------------------------------------------------------
def _px_to_pt(value_px: float, dpi: float) -> float:
    """
    Convert a measurement from source-image pixels to PDF points.

    PDF points are 1/72 of an inch.  Given the image DPI (dots per inch),
    pixels are converted to inches first, then to points:
        points = (pixels / dpi) * 72
    """
    return (value_px / dpi) * 72.0


# ---------------------------------------------------------------------------
# Main Conversion Pipeline
# ---------------------------------------------------------------------------
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
    """
    Full pipeline: load TIFF -> binarize -> skeletonize -> extract chains ->
    simplify -> estimate widths -> render to PDF.

    Parameters:
        input_tiff:           Path to the source TIFF file.
        output_pdf:           Path where the vector PDF will be written.
        threshold:            Binarization threshold (0-255), or None for Otsu.
        invert:               If True, treat bright pixels as ink.
        width_scale:          Multiplier for estimated line widths.
        min_width_px:         Minimum stroke width in pixels.
        max_width_px:         Maximum stroke width in pixels.
        width_delta_px:       Width change threshold for splitting chains.
        min_run_nodes:        Minimum run length before a width-split is allowed.
        simplify_epsilon_px:  RDP simplification tolerance in pixels (0 to disable).

    Returns:
        The total number of stroked path segments written to the PDF.
    """
    pipeline_start = time.time()

    # --- Step 1: Load the TIFF and get its grayscale pixel data and DPI ---
    t = _step("[1/7] Loading TIFF...")
    gray, (x_dpi, y_dpi) = _load_single_page_tiff(input_tiff)
    _done(t)

    # --- Step 2: Binarize the grayscale image into an ink mask ---
    t = _step("[2/7] Binarizing...")
    ink = _binarize(gray, threshold=threshold, invert=invert)
    _done(t)

    if not np.any(ink):
        raise ValueError("No black line pixels detected after thresholding.")

    # --- Step 3: Skeletonize the binary mask to 1-pixel-wide centerlines ---
    # Skeletonization erodes thick lines down to their medial axis.
    t = _step("[3/7] Skeletonizing...")
    skel = skeletonize(ink)
    _done(t)
    if not np.any(skel):
        raise ValueError("Skeletonization produced no centerlines.")

    # --- Step 4: Build a distance-transform map for line-width estimation ---
    # For every ink pixel, dist_map stores the Euclidean distance to the nearest
    # background pixel.  At skeleton pixels this equals the inscribed circle radius.
    t = _step("[4/7] Computing distance transform...")
    dist_map = ndimage.distance_transform_edt(ink)
    _done(t)

    # --- Step 5: Convert the skeleton into a graph and extract chains ---
    t = _step("[5/7] Building skeleton graph...")
    adjacency = _build_skeleton_graph(skel)
    chains = _extract_chains(adjacency)
    _done(t)
    print(f"       Found {len(chains)} chains.")

    # --- Step 6: Set up the PDF canvas with the same physical dimensions ---
    t = _step("[6/7] Setting up PDF canvas...")
    height_px, width_px = gray.shape
    # Convert image dimensions from pixels to PDF points (1 point = 1/72 inch)
    page_w_pt = _px_to_pt(float(width_px), x_dpi)
    page_h_pt = _px_to_pt(float(height_px), y_dpi)

    # Create output directory if it doesn't exist
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Initialize a ReportLab PDF canvas at the computed page size
    pdf = canvas.Canvas(str(output_pdf), pagesize=(page_w_pt, page_h_pt), pageCompression=1)
    pdf.setStrokeColor(black)   # all strokes are black
    pdf.setLineCap(1)           # 1 = round cap (rounded ends on strokes)
    pdf.setLineJoin(1)          # 1 = round join (rounded corners between segments)

    _done(t)

    path_count = 0
    avg_dpi = (x_dpi + y_dpi) / 2.0  # used for width conversion when x/y DPI differ

    # --- Step 7: Process each chain and render it as stroked PDF path(s) ---
    print("[7/7] Rendering chains to PDF...")
    chain_start = time.time()
    total_chains = len(chains)
    for chain_idx, chain in enumerate(chains, 1):
        # Simplify the chain to reduce the number of points (smaller PDF file)
        simplified_chain = _simplify_chain(chain, epsilon=simplify_epsilon_px)

        # Split the chain into runs of consistent width
        runs = _split_chain_by_width(
            chain=simplified_chain,
            dist_map=dist_map,
            width_scale=width_scale,
            min_width_px=min_width_px,
            max_width_px=max_width_px,
            width_delta_px=width_delta_px,
            min_run_nodes=min_run_nodes,
        )

        # Draw each run as a separate PDF path with its own stroke width
        for run_nodes, run_width in runs:
            if len(run_nodes) < 2:
                continue

            # Start a new PDF path and move to the first point
            pdf_path = pdf.beginPath()
            first_r, first_c = run_nodes[0]
            # Convert pixel coordinates to PDF points.
            # Note: PDF origin is bottom-left, but image origin is top-left,
            # so Y is flipped: pdf_y = page_height - pixel_y
            first_x = _px_to_pt(float(first_c), x_dpi)
            first_y = page_h_pt - _px_to_pt(float(first_r), y_dpi)
            pdf_path.moveTo(first_x, first_y)

            # Draw line segments to each subsequent point
            for r, c in run_nodes[1:]:
                x = _px_to_pt(float(c), x_dpi)
                y = page_h_pt - _px_to_pt(float(r), y_dpi)
                pdf_path.lineTo(x, y)

            # Set the stroke width for this run and draw the path
            pdf.setLineWidth(_px_to_pt(run_width, avg_dpi))
            pdf.drawPath(pdf_path, stroke=1, fill=0)  # stroke only, no fill
            path_count += 1

        # Update progress bar every 50 chains or on the last one
        if chain_idx % 50 == 0 or chain_idx == total_chains:
            _print_progress(chain_idx, total_chains, chain_start)

    # Finalize the PDF page and save to disk
    t = _step("       Saving PDF...")
    pdf.showPage()
    pdf.save()
    _done(t)

    total_elapsed = time.time() - pipeline_start
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"       Total time: {mins}m {secs:02d}s — {path_count} path segments written.")

    return path_count


# ---------------------------------------------------------------------------
# Command-Line Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Define and parse all command-line arguments for the script."""
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Parse CLI arguments, validate them, and run the conversion pipeline.
    """
    args = parse_args()

    # Remove Pillow's safety limit so very large TIFFs can be processed
    Image.MAX_IMAGE_PIXELS = None

    # --- Validate argument ranges ---
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

    # Run the full conversion pipeline
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


# Only run main() when the script is executed directly,
# not when it is imported as a module.
if __name__ == "__main__":
    main()
