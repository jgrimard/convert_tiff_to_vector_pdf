# Raster TIFF to Stroke-Only Vector PDF

This project converts a black-and-white raster TIFF into a vector PDF made from connected stroked centerlines with variable line thickness.

Multi-page TIFFs are not currently supported.  However, multi-frame TIFFs (e.g. pyramid TIFFs with multiple resolutions of the same page) are supported — the script automatically selects the largest frame. Low-DPI images are upscaled in memory before processing to ensure quality results, while the output PDF retains the original physical dimensions (inches).

The output is intentionally stroke-based. It does not generate filled outlines/shapes for linework.

## 1) Create and activate a virtual environment (venv)

```powershell
python -m venv .venv
```

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

If PowerShell blocks script execution, run this once in the current session and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2) Install dependencies (inside venv)

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Convert TIFF to vector PDF (inside venv)

```powershell
python convert_tiff_to_vector_pdf.py --input 444924.tif --output 444924_vector.pdf
```

You can also use positional arguments:

```powershell
python convert_tiff_to_vector_pdf.py 444924.tif 444924_vector.pdf
```

If you provide only the input path, output is created in the same folder using
the same base name with `-vector.pdf` appended:

```powershell
python convert_tiff_to_vector_pdf.py 444924.tif
# -> 444924-vector.pdf
```

### Optional tuning parameters

#### `--threshold`

Controls how the grayscale image is split into "ink" vs "background." Every pixel with a brightness at or below this value is treated as ink (line work). When omitted, the script automatically picks the best split point using Otsu's method, which analyzes the image histogram to maximize contrast between the two groups. You usually only need to set this manually if Otsu's auto-detection is picking up too much noise or missing faint lines.

- **Default:** automatic (Otsu's method)
- **Range:** `0` - `255` (integer). `0` = only pure-black pixels are ink; `255` = almost everything is ink.

#### `--invert`

Controls whether the image colors are inverted before processing. Accepts one of three values:

- **`auto`** (default) — The script examines the image and counts the proportion of dark vs light pixels using the binarization threshold. If more than 50% of pixels are dark, it assumes the drawing has white lines on a dark background and automatically inverts. This is ideal for batch processing where input images may be a mix of normal and inverted drawings.
- **`true`** — Forces inversion on. Use this to override auto-detection when you know your source TIFF has **white lines on a dark background**.
- **`false`** — Forces inversion off. Use this to override auto-detection when you know your source TIFF has **dark lines on a light background**.

- **Default:** `auto`

#### `--width-scale`

A multiplier applied to every computed stroke width in the output PDF. The script estimates each line's thickness from the original raster; this factor scales that estimate up or down uniformly across the entire drawing. Use a value greater than 1.0 to make all output strokes thicker, or less than 1.0 to make them thinner.

- **Default:** `1.0` (no scaling — widths match the original raster)
- **Range:** any value `> 0` (e.g., `0.5` = half thickness, `2.0` = double thickness)

#### `--min-width-px`

The minimum allowed stroke width, measured in source-image pixels. Any line thinner than this value in the original image will be drawn at this width instead. This prevents extremely thin lines from disappearing or becoming too faint in the PDF.

- **Default:** `0.75` px
- **Range:** any value `> 0`

#### `--max-width-px`

The maximum allowed stroke width, measured in source-image pixels. Any line thicker than this value in the original image will be capped at this width. This prevents large filled regions from producing unreasonably wide strokes.

- **Default:** `80.0` px
- **Range:** any value `> 0` (must be ≥ `--min-width-px`)

#### `--width-delta-px`

Controls when the script splits a continuous centerline path into separate segments because the line thickness is changing. As the script walks along a skeleton path, it compares the estimated width at each point to the previous point. When the width difference between two adjacent points reaches this value (in source-image pixels), a new segment begins so each piece can have its own uniform stroke width in the PDF. **Lower values** produce more splits and more accurate per-segment widths, but create a larger PDF with more path objects. **Higher values** keep longer connected paths but average out width variations over longer stretches.

- **Default:** `2.5` px
- **Range:** any value `> 0`

#### `--min-run-nodes`

The minimum number of skeleton points a segment must contain before the script is allowed to split it due to a width change. This prevents the script from creating lots of tiny path fragments in areas where the line width fluctuates rapidly over short distances (e.g., at junctions or near text). **Higher values** force longer unbroken segments and reduce over-splitting from local width noise. **Lower values** (down to the minimum of 2) allow the script to split more aggressively for tighter width accuracy.

- **Default:** `40` nodes
- **Range:** any integer `≥ 2`

#### `--simplify-epsilon-px`

Controls how aggressively the script simplifies the geometry of each centerline path using the Ramer-Douglas-Peucker algorithm. This value is the maximum allowed perpendicular deviation, in source-image pixels, between the simplified path and the original skeleton points. **Larger values** remove more points and produce a smaller PDF, but the paths will be less precise. **Smaller values** keep more detail. Setting this to `0` disables simplification entirely, preserving every single skeleton pixel as a path vertex (largest possible PDF).

- **Default:** `0.8` px
- **Range:** any value `≥ 0` (`0` = no simplification)

#### `--min-dpi`

The minimum DPI (dots per inch) for processing. If the input TIFF has a DPI below this value on either axis, the image is upscaled in memory using bilinear interpolation before conversion begins. This ensures that skeletonization and line-width estimation produce good results even on low-resolution source files. The output PDF retains the original physical dimensions (inches) because the DPI and pixel counts are scaled together.

- **Default:** `300.0`
- **Range:** any value `> 0`

Example:

PowerShell command with default options:

```powershell
python convert_tiff_to_vector_pdf.py --input 444924.tif --output 444924_vector.pdf --width-delta-px 2.5 --min-run-nodes 40 --simplify-epsilon-px 0.8
```

Equivalent positional form:

```powershell
python convert_tiff_to_vector_pdf.py 444924.tif 444924_vector.pdf --width-delta-px 2.5 --min-run-nodes 40 --simplify-epsilon-px 0.8
```

## Notes

- Multi-page TIFFs are not supported.
- Multi-frame TIFFs are supported: the script selects the frame with the most pixels (highest resolution).
- Images below the minimum DPI (default 300) are automatically upscaled in memory before processing.
- Page size is derived from source pixel dimensions and TIFF DPI metadata, so the output PDF matches the original physical dimensions.
- Paths split at line junctions and major thickness transitions, producing connected multi-segment centerlines.
