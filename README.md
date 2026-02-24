# Raster TIFF to Stroke-Only Vector PDF

This project converts a black-and-white single-page raster TIFF into a vector PDF made from connected stroked centerlines with variable line thickness.

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
python convert_tiff_to_vector_pdf.py --input 444924.tif --output 444924_vector.pdf --allow-large-images
```

Optional tuning:

- `--threshold 0..255` to force binarization threshold (default uses Otsu)
- `--invert` if your source is white linework on dark background
- `--width-scale` to globally thicken/thin output strokes
- `--min-width-px` and `--max-width-px` to clamp width range
- `--width-delta-px` split connected paths only when width changes materially
- `--min-run-nodes` avoid over-splitting from local width noise
- `--simplify-epsilon-px` mild geometry simplification (0 disables)
- `--allow-large-images` for very large TIFFs that exceed Pillow's safety pixel limit (always add this)

Current defaults are tuned for longer connected chains:

- `--width-delta-px 2.5`
- `--min-run-nodes 40`
- `--simplify-epsilon-px 0.8`

Example:

PowerShell command with default options:

```powershell
python convert_tiff_to_vector_pdf.py --input 444924.tif --output 444924_vector.pdf --allow-large-images --width-delta-px 2.5 --min-run-nodes 40 --simplify-epsilon-px 0.8
```

## Notes

- Current scope is **single-page TIFF**.
- Page size is derived from source pixel dimensions and TIFF DPI metadata.
- Paths split at line junctions and major thickness transitions, producing connected multi-segment centerlines instead of tiny independent strokes.
