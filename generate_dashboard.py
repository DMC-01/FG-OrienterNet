#!/usr/bin/env python3
"""
OrienterNet Results Dashboard Generator

Creates:
- <run_name>_dashboard.html: main overview with summary, interesting IDs, table
- <run_name>_analysis.html: notebook-style analysis page with plots and exports
- <run_name>_detail_<image_id>.html: one static detail page per image

The detail pages use the saved JPG/JSON artifacts directly by relative path,
so the browser fetches the image files instead of embedding huge base64 strings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# Paths
# ============================================================================

BASE_PROJECT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = BASE_PROJECT_DIR / "data"
RUNS_PATH = BASE_DATA_DIR / "runs"

# ============================================================================
# Data helpers
# ============================================================================

NUMERIC_COLUMNS = [
    "image_id",
    "gt_latitude",
    "gt_longitude",
    "gt_yaw",
    "src_latitude",
    "src_longitude",
    "src_yaw",
    "pred_latitude",
    "pred_longitude",
    "pred_x_meters",
    "pred_y_meters",
    "pred_yaw",
    "pred_probability",
    "distance_m",
    "yaw_error",
    "yaw_error_signed",
    "yaw_error_abs",
    "exif_focal_35mm",
    "exif_focal_ratio",
    "exif_orientation",
    "exif_altitude",
]


def get_latest_run_dir(runs_dir: Path = RUNS_PATH) -> Path:
    run_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    return run_dirs[0]


def load_results_csv(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    return enrich_results(df)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371000.0


def normalize_angle(angle: float) -> float:
    if pd.isna(angle):
        return float("nan")
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return float(angle)


def enrich_results(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_numeric_columns(df)

    coord_cols = ["pred_longitude", "pred_latitude", "gt_longitude", "gt_latitude"]
    if all(col in df.columns for col in coord_cols):
        def distance_for_row(row: pd.Series) -> float:
            values = [row.get(col) for col in coord_cols]
            if any(pd.isna(v) for v in values):
                return float("nan")
            return haversine(row["pred_longitude"], row["pred_latitude"], row["gt_longitude"], row["gt_latitude"])
        df["distance_m"] = df.apply(distance_for_row, axis=1)
    elif "distance_m" not in df.columns:
        df["distance_m"] = np.nan

    yaw_cols = ["pred_yaw", "gt_yaw"]
    if all(col in df.columns for col in yaw_cols):
        def yaw_for_row(row: pd.Series) -> float:
            if pd.isna(row.get("pred_yaw")) or pd.isna(row.get("gt_yaw")):
                return float("nan")
            return normalize_angle(float(row["pred_yaw"]) - float(row["gt_yaw"]))
        df["yaw_error_signed"] = df.apply(yaw_for_row, axis=1)
        df["yaw_error"] = df["yaw_error_signed"]
        df["yaw_error_abs"] = df["yaw_error_signed"].abs()
    else:
        if "yaw_error_signed" not in df.columns:
            df["yaw_error_signed"] = np.nan
        if "yaw_error" not in df.columns:
            df["yaw_error"] = df["yaw_error_signed"]
        if "yaw_error_abs" not in df.columns:
            df["yaw_error_abs"] = df["yaw_error_signed"].abs()

    if "pred_probability" not in df.columns:
        df["pred_probability"] = np.nan

    if "error_message" not in df.columns:
        df["error_message"] = ""

    return df


def parse_tile_radius_meters(run_name: str, default: int = 136) -> int:
    match = re.search(r"(?:^|_)process_(\d+)t_", run_name) or re.search(r"(\d+)t_", run_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return default


def valid_prediction_mask(df: pd.DataFrame) -> pd.Series:
    required = ["gt_latitude", "gt_longitude", "pred_latitude", "pred_longitude", "gt_yaw", "pred_yaw"]
    present = [col for col in required if col in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    mask = df[present].notna().all(axis=1)
    if "error_message" in df.columns:
        # Keep rows with the old JSON serialization warning if coordinates exist; those predictions were valid.
        messages = df["error_message"].fillna("").astype(str)
        severe = (messages != "") & ~messages.str.contains("BoundaryBox is not JSON serializable", na=False)
        mask &= ~severe
    return mask


def format_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(value)


def fmt_num(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return escape(str(value))


def fmt_coord(value: Any) -> str:
    return fmt_num(value, decimals=7)


def fmt_prob(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    value_f = float(value)
    # Do not force percent because OrienterNet probabilities can be extremely small.
    if abs(value_f) >= 0.001:
        return f"{value_f:.6f}"
    return f"{value_f:.3e}"


def safe_stat(series: pd.Series, func: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    if func == "mean":
        return float(values.mean())
    if func == "median":
        return float(values.median())
    if func == "max":
        return float(values.max())
    if func == "min":
        return float(values.min())
    raise ValueError(func)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read JSON {path}: {exc}")
        return {}


def relative_url(path: Path, html_dir: Path) -> str:
    try:
        return os.path.relpath(path, html_dir).replace(os.sep, "/")
    except Exception:
        return str(path).replace(os.sep, "/")


def artifact_url(artifact_dir: Path, metadata: Dict[str, Any], key: str, fallback: str, html_dir: Path) -> str:
    files = metadata.get("files") or {}
    filename = files.get(key) or fallback
    if not filename:
        return ""
    path = artifact_dir / filename
    if not path.exists():
        return ""
    return relative_url(path, html_dir)


def google_maps_link(lat: Any, lon: Any) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return ""
    return f"https://www.google.com/maps/search/?api=1&query={float(lat):.8f},{float(lon):.8f}"


def google_streetview_link(lat: Any, lon: Any, yaw: Any = None) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return ""
    heading = ""
    if yaw is not None and not pd.isna(yaw):
        heading = f"&heading={float(yaw):.2f}"
    return f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={float(lat):.8f},{float(lon):.8f}{heading}"


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def detail_link(run_name: str, image_id: Any, text: Optional[str] = None) -> str:
    iid = format_image_id(image_id)
    label = escape(text if text is not None else iid)
    return f'<a href="{escape(run_name)}_detail_{escape(iid)}.html">{label}</a>'

# ============================================================================
# Main dashboard
# ============================================================================


def metric_card(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-title">{escape(title)}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{escape(subtitle)}</div>
    </div>
    """


def generate_stats_html(df: pd.DataFrame) -> str:
    valid = valid_prediction_mask(df)
    failed = df["error_message"].fillna("").astype(str).str.strip() != ""
    success_count = int(valid.sum())
    total = len(df)
    fail_count = int(failed.sum())
    avg_dist = safe_stat(df.loc[valid, "distance_m"], "mean") if "distance_m" in df else np.nan
    avg_yaw = safe_stat(df.loc[valid, "yaw_error_abs"], "mean") if "yaw_error_abs" in df else np.nan
    max_dist = safe_stat(df.loc[valid, "distance_m"], "max") if "distance_m" in df else np.nan
    max_yaw = safe_stat(df.loc[valid, "yaw_error_abs"], "max") if "yaw_error_abs" in df else np.nan
    avg_prob = safe_stat(df.loc[valid, "pred_probability"], "mean") if "pred_probability" in df else np.nan
    max_prob = safe_stat(df.loc[valid, "pred_probability"], "max") if "pred_probability" in df else np.nan

    return f"""
    <section class="metrics-grid">
        {metric_card("Total rows", str(total), "all CSV rows")}
        {metric_card("Valid predictions", str(success_count), "coordinates + yaw available")}
        {metric_card("Rows with error text", str(fail_count), "non-empty error_message")}
        {metric_card("Average distance", fmt_num(avg_dist, 2, " m"), "mean over valid predictions")}
        {metric_card("Max distance", fmt_num(max_dist, 2, " m"), "largest distance error")}
        {metric_card("Average yaw error", fmt_num(avg_yaw, 2, "°"), "mean absolute yaw error")}
        {metric_card("Max yaw error", fmt_num(max_yaw, 2, "°"), "largest absolute yaw error")}
        {metric_card("Avg / max probability", f"{fmt_prob(avg_prob)} / {fmt_prob(max_prob)}", "pred_probability")}
    </section>
    """


def interesting_list(df: pd.DataFrame, run_name: str, title: str, col: str, ascending: bool, n: int = 5, formatter=fmt_num) -> str:
    if col not in df.columns:
        return ""
    subset = df[["image_id", col]].dropna().copy()
    if subset.empty:
        rows = '<li class="muted">No valid rows.</li>'
    else:
        subset = subset.sort_values(col, ascending=ascending).head(n)
        rows = "".join(
            f"<li>{detail_link(run_name, row.image_id)} <span>{formatter(row[col])}</span></li>"
            for _, row in subset.iterrows()
        )
    return f"""
    <div class="interesting-card">
        <h6>{escape(title)}</h6>
        <ol>{rows}</ol>
    </div>
    """


def generate_interesting_html(df: pd.DataFrame, run_name: str) -> str:
    valid = df.loc[valid_prediction_mask(df)].copy()
    if valid.empty:
        return """
        <section class="section-card">
            <h5><i class="fas fa-magnifying-glass-chart"></i> Interesting IDs</h5>
            <p class="muted">No valid predictions available yet.</p>
        </section>
        """

    return f"""
    <section class="section-card">
        <h5><i class="fas fa-magnifying-glass-chart"></i> Interesting IDs to inspect</h5>
        <div class="interesting-grid">
            {interesting_list(valid, run_name, "Top 5 probability", "pred_probability", ascending=False, formatter=fmt_prob)}
            {interesting_list(valid, run_name, "Lowest 5 probability", "pred_probability", ascending=True, formatter=fmt_prob)}
            {interesting_list(valid, run_name, "Largest 5 distance errors", "distance_m", ascending=False, formatter=lambda v: fmt_num(v, 2, " m"))}
            {interesting_list(valid, run_name, "Smallest 5 distance errors", "distance_m", ascending=True, formatter=lambda v: fmt_num(v, 2, " m"))}
            {interesting_list(valid, run_name, "Largest 5 yaw errors", "yaw_error_abs", ascending=False, formatter=lambda v: fmt_num(v, 2, "°"))}
            {interesting_list(valid, run_name, "Smallest 5 yaw errors", "yaw_error_abs", ascending=True, formatter=lambda v: fmt_num(v, 2, "°"))}
        </div>
    </section>
    """


def generate_table_html(df: pd.DataFrame, run_name: str) -> str:
    columns = [
        ("image_id", "image_id", lambda v: escape(format_image_id(v))),
        ("distance_m", "distance_m", lambda v: fmt_num(v, 2, " m")),
        ("yaw_error_abs", "yaw_abs", lambda v: fmt_num(v, 2, "°")),
        ("pred_probability", "pred_prob", fmt_prob),
        ("gt_latitude", "gt_lat", fmt_coord),
        ("gt_longitude", "gt_lon", fmt_coord),
        ("pred_latitude", "pred_lat", fmt_coord),
        ("pred_longitude", "pred_lon", fmt_coord),
        ("gt_yaw", "gt_yaw", lambda v: fmt_num(v, 2, "°")),
        ("pred_yaw", "pred_yaw", lambda v: fmt_num(v, 2, "°")),
        ("error_message", "error", lambda v: escape(str(v)) if str(v) != "nan" else ""),
    ]
    available = [(col, label, formatter) for col, label, formatter in columns if col in df.columns]

    header_html = "".join(f'<th class="col-{escape(label)}">{escape(label)}</th>' for _, label, _ in available)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col, label, formatter in available:
            value = row.get(col)
            if col == "image_id":
                iid = format_image_id(value)
                cells.append(f'<td class="image-id-cell">{detail_link(run_name, iid)}</td>')
            else:
                cells.append(f'<td class="col-{escape(label)}">{formatter(value)}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="table-wrap">
        <table class="results-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def generate_main_html(run_dir: Path, run_name: str, df: pd.DataFrame, output_dir: Path) -> str:
    stats_html = generate_stats_html(df)
    interesting_html = generate_interesting_html(df, run_name)
    table_html = generate_table_html(df, run_name)
    analysis_file = f"{run_name}_analysis.html"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrienterNet Dashboard - {escape(run_name)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ background:#f5f6fb; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif; color:#242833; }}
        .navbar {{ background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:0.35rem 1rem; }}
        .navbar-brand {{ font-size:1rem; font-weight:700; }}
        .nav-link-small {{ color:white; text-decoration:none; padding:0.25rem 0.55rem; border-radius:6px; background:rgba(255,255,255,0.16); font-size:0.85rem; margin-left:0.4rem; }}
        .nav-link-small:hover {{ color:white; background:rgba(255,255,255,0.28); }}
        .page {{ padding:18px; }}
        .hero {{ background:white; border-radius:12px; padding:16px 18px; box-shadow:0 2px 10px rgba(20,20,60,0.08); margin-bottom:16px; }}
        .hero h1 {{ font-size:1.35rem; margin:0 0 6px; }}
        .run-info {{ color:#667; font-size:0.9rem; }}
        .metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:16px; }}
        .metric-card {{ background:white; border-radius:10px; padding:13px 14px; box-shadow:0 2px 8px rgba(20,20,60,0.08); }}
        .metric-title {{ color:#65708a; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; font-weight:700; }}
        .metric-value {{ font-size:1.35rem; font-weight:800; color:#2a3150; margin-top:3px; }}
        .metric-subtitle {{ color:#7b8398; font-size:0.78rem; }}
        .section-card {{ background:white; border-radius:12px; padding:16px; box-shadow:0 2px 10px rgba(20,20,60,0.08); margin-bottom:16px; }}
        .section-card h5 {{ margin-bottom:12px; font-size:1rem; font-weight:800; color:#2b3150; }}
        .interesting-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }}
        .interesting-card {{ border:1px solid #e6e8f0; border-radius:9px; padding:10px 12px; background:#fbfcff; }}
        .interesting-card h6 {{ margin:0 0 8px; font-size:0.86rem; font-weight:800; color:#5968d8; }}
        .interesting-card ol {{ margin:0; padding-left:20px; }}
        .interesting-card li {{ margin:3px 0; font-size:0.86rem; display:list-item; }}
        .interesting-card span {{ float:right; color:#4a5166; font-variant-numeric:tabular-nums; }}
        .muted {{ color:#7f8799; }}
        .table-wrap {{ overflow:auto; max-height:70vh; border-radius:10px; border:1px solid #e3e6ef; }}
        .results-table {{ width:100%; border-collapse:collapse; background:white; font-size:0.84rem; }}
        .results-table th {{ position:sticky; top:0; z-index:1; background:#667eea; color:white; font-weight:800; text-align:left; padding:8px 9px; white-space:nowrap; }}
        .results-table td {{ text-align:left; padding:7px 9px; border-bottom:1px solid #edf0f6; white-space:nowrap; font-variant-numeric:tabular-nums; }}
        .results-table tr:hover {{ background:#f4f7ff; }}
        .image-id-cell, .col-image_id {{ width:74px; min-width:74px; max-width:74px; }}
        .col-error {{ max-width:420px; overflow:hidden; text-overflow:ellipsis; }}
        a {{ color:#3c57d6; text-decoration:none; font-weight:700; }}
        a:hover {{ text-decoration:underline; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <span class="navbar-brand"><i class="fas fa-map-location-dot"></i> OrienterNet Results Dashboard</span>
        <span>
            <a class="nav-link-small" href="{escape(analysis_file)}"><i class="fas fa-chart-line"></i> Analysis</a>
        </span>
    </nav>
    <main class="page">
        <section class="hero">
            <h1>Run: {escape(run_name)}</h1>
            <div class="run-info">
                <i class="fas fa-clock"></i> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <i class="fas fa-folder"></i> Run path: {escape(str(run_dir))}<br>
                <i class="fas fa-file"></i> Dashboard path: {escape(str(output_dir))}
            </div>
        </section>
        {stats_html}
        {interesting_html}
        <section class="section-card">
            <h5><i class="fas fa-table"></i> Results overview</h5>
            {table_html}
        </section>
    </main>
</body>
</html>"""

# ============================================================================
# Detail pages
# ============================================================================


def first_non_nan(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return default


def tile_radius_from_metadata(metadata: Dict[str, Any], run_name: str) -> float:
    # In the batch process this value is the radius / half-size.  The displayed
    # request tile should therefore be 2 * this number on each side.
    value = first_non_nan(
        metadata.get("tile_radius_meters"),
        metadata.get("tile_half_size_meters"),
        metadata.get("tile_size_meters"),
        default=parse_tile_radius_meters(run_name),
    )
    try:
        return float(value)
    except Exception:
        return float(parse_tile_radius_meters(run_name))


def bounds_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
    raw = metadata.get("tile_bounds_latlon") or {}
    if not isinstance(raw, dict):
        return None
    south = first_non_nan(raw.get("south"), raw.get("min_lat"))
    north = first_non_nan(raw.get("north"), raw.get("max_lat"))
    west = first_non_nan(raw.get("west"), raw.get("min_lon"))
    east = first_non_nan(raw.get("east"), raw.get("max_lon"))
    if any(v is None for v in [south, north, west, east]):
        return None
    try:
        return {"south": float(south), "north": float(north), "west": float(west), "east": float(east)}
    except Exception:
        return None


def build_detail_stats_html(row: pd.Series, camera: Dict[str, Any], urls: Dict[str, str]) -> str:
    gt_lat, gt_lon, gt_yaw = row.get("gt_latitude"), row.get("gt_longitude"), row.get("gt_yaw")
    pred_lat, pred_lon, pred_yaw = row.get("pred_latitude"), row.get("pred_longitude"), row.get("pred_yaw")
    distance_m = row.get("distance_m")
    yaw_signed = row.get("yaw_error_signed")
    yaw_abs = row.get("yaw_error_abs")
    pred_prob = row.get("pred_probability")
    gt_maps = google_maps_link(gt_lat, gt_lon)
    gt_sv = google_streetview_link(gt_lat, gt_lon, gt_yaw)
    pred_maps = google_maps_link(pred_lat, pred_lon)
    pred_sv = google_streetview_link(pred_lat, pred_lon, pred_yaw)

    def maybe_link(url: str, label: str, icon: str) -> str:
        if not url:
            return ""
        return f'<a target="_blank" rel="noopener" href="{escape(url)}"><i class="{escape(icon)}"></i> {escape(label)}</a>'

    artifact_figures = []
    for key, label in [
        ("osm_tile", "Model OSM raster JPG"),
        ("prediction_map", "Prediction map"),
        ("prediction_probability", "Prediction probability"),
        ("neural_map", "Neural activation"),
    ]:
        if urls.get(key):
            artifact_figures.append(
                f'<figure><img src="{escape(urls[key])}" alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>'
            )
    artifacts_html = "".join(artifact_figures) or '<p class="muted">No JPG artifacts found for this image.</p>'

    return f"""
    <div class="stats-section">
        <h6><i class="fas fa-camera"></i> Camera / image</h6>
        <dl class="stats-dl">
            <dt>Image ID</dt><dd>{escape(format_image_id(row.get('image_id')))}</dd>
            <dt>Camera size</dt><dd>{escape(str(camera.get('width', '—')))} × {escape(str(camera.get('height', '—')))}</dd>
            <dt>Focal length</dt><dd>fx {fmt_num(camera.get('fx'), 2)}, fy {fmt_num(camera.get('fy'), 2)}</dd>
            <dt>Principal point</dt><dd>cx {fmt_num(camera.get('cx'), 2)}, cy {fmt_num(camera.get('cy'), 2)}</dd>
            <dt>Image path</dt><dd class="path-text">{escape(str(row.get('image_path', '')))}</dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6><i class="fas fa-bullseye"></i> Errors and probability</h6>
        <dl class="stats-dl">
            <dt>Distance error</dt><dd><strong>{fmt_num(distance_m, 2, ' m')}</strong></dd>
            <dt>Yaw error signed</dt><dd>{fmt_num(yaw_signed, 2, '°')}</dd>
            <dt>Yaw error absolute</dt><dd><strong>{fmt_num(yaw_abs, 2, '°')}</strong></dd>
            <dt>Prediction probability</dt><dd>{fmt_prob(pred_prob)}</dd>
        </dl>
    </div>
    <div class="stats-section two-col-stats">
        <div>
            <h6><i class="fas fa-location-dot"></i> Ground truth</h6>
            <dl class="stats-dl">
                <dt>Latitude</dt><dd>{fmt_coord(gt_lat)}</dd>
                <dt>Longitude</dt><dd>{fmt_coord(gt_lon)}</dd>
                <dt>Yaw</dt><dd>{fmt_num(gt_yaw, 2, '°')}</dd>
            </dl>
            <div class="external-links">{maybe_link(gt_maps, 'Google Maps', 'fas fa-map')} {maybe_link(gt_sv, 'Street View', 'fas fa-street-view')}</div>
        </div>
        <div>
            <h6><i class="fas fa-location-crosshairs"></i> Prediction</h6>
            <dl class="stats-dl">
                <dt>Latitude</dt><dd>{fmt_coord(pred_lat)}</dd>
                <dt>Longitude</dt><dd>{fmt_coord(pred_lon)}</dd>
                <dt>Yaw</dt><dd>{fmt_num(pred_yaw, 2, '°')}</dd>
            </dl>
            <div class="external-links">{maybe_link(pred_maps, 'Google Maps', 'fas fa-map')} {maybe_link(pred_sv, 'Street View', 'fas fa-street-view')}</div>
        </div>
    </div>
    <div class="stats-section">
        <h6><i class="fas fa-images"></i> Saved artifacts</h6>
        <div class="artifact-grid">{artifacts_html}</div>
    </div>
    """


def generate_detail_html(row: pd.Series, run_dir: Path, run_name: str, output_dir: Path) -> str:
    image_id = format_image_id(row.get("image_id"))
    artifact_dir = run_dir / "artifacts" / f"image_{image_id}"
    metadata = load_json(artifact_dir / "artifacts.json")
    camera = load_json(artifact_dir / "camera.json")

    urls = {
        "rectified": artifact_url(artifact_dir, metadata, "rectified_image", "rectified_image.jpg", output_dir),
        "osm_tile": artifact_url(artifact_dir, metadata, "osm_tile", "osm_tile.jpg", output_dir),
        "prediction_map": artifact_url(artifact_dir, metadata, "prediction_map", "prediction_map.jpg", output_dir),
        "prediction_probability": artifact_url(artifact_dir, metadata, "prediction_probability", "prediction_probability.jpg", output_dir),
        "neural_map": artifact_url(artifact_dir, metadata, "neural_map_rgb", "neural_map_rgb.jpg", output_dir),
    }

    tile_radius = tile_radius_from_metadata(metadata, run_name)
    tile_side = first_non_nan(metadata.get("tile_side_meters"), metadata.get("tile_diameter_meters"), default=tile_radius * 2.0)
    try:
        tile_side = float(tile_side)
    except Exception:
        tile_side = tile_radius * 2.0

    center_lat = first_non_nan(row.get("gt_latitude"), row.get("pred_latitude"), default=0.0)
    center_lon = first_non_nan(row.get("gt_longitude"), row.get("pred_longitude"), default=0.0)

    detail_data = {
        "image_id": image_id,
        "gt_lat": row.get("gt_latitude"),
        "gt_lon": row.get("gt_longitude"),
        "gt_yaw": row.get("gt_yaw"),
        "pred_lat": row.get("pred_latitude"),
        "pred_lon": row.get("pred_longitude"),
        "pred_yaw": row.get("pred_yaw"),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "tile_radius_meters": tile_radius,
        "tile_side_meters": tile_side,
        "tile_corners_latlon": metadata.get("tile_corners_latlon"),
        "tile_bounds_latlon": bounds_from_metadata(metadata),
        "urls": urls,
        "links": {
            "gt_maps": google_maps_link(row.get("gt_latitude"), row.get("gt_longitude")),
            "gt_streetview": google_streetview_link(row.get("gt_latitude"), row.get("gt_longitude"), row.get("gt_yaw")),
            "pred_maps": google_maps_link(row.get("pred_latitude"), row.get("pred_longitude")),
            "pred_streetview": google_streetview_link(row.get("pred_latitude"), row.get("pred_longitude"), row.get("pred_yaw")),
        },
    }
    detail_json = json.dumps(json_safe(detail_data), ensure_ascii=False, allow_nan=False)
    stats_html = build_detail_stats_html(row, camera, urls)

    rectified_url = urls.get("rectified") or ""
    rectified_img = (
        f'<img id="rectifiedImage" src="{escape(rectified_url)}" alt="Rectified image">'
        if rectified_url
        else '<div class="missing-image"><i class="fas fa-triangle-exclamation"></i><br>No rectified_image.jpg found</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image {escape(image_id)} Detail</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet-imageoverlay-rotated@0.2.2/Leaflet.ImageOverlay.Rotated.min.js"></script>
    <style>
        * {{ box-sizing:border-box; }}
        html, body {{ height:100%; margin:0; }}
        body {{ background:#f5f5f5; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif; overflow:hidden; color:#222; }}
        .navbar {{ height:42px; min-height:42px; padding:0 8px; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); }}
        .navbar-brand {{ font-size:0.95rem; font-weight:800; }}
        .back-btn, .analysis-btn {{ display:inline-block; color:white; text-decoration:none; background:rgba(255,255,255,0.16); border-radius:5px; padding:5px 9px; font-size:0.82rem; margin-right:6px; }}
        .back-btn:hover, .analysis-btn:hover {{ color:white; background:rgba(255,255,255,0.28); text-decoration:none; }}
        .container-detail {{ display:grid; grid-template-columns:40% 60%; grid-template-rows:44% 56%; height:calc(100vh - 42px); }}
        .left-panel {{ grid-column:1; grid-row:1 / 3; display:flex; flex-direction:column; min-height:0; background:white; border-right:1px solid #d9dce5; }}
        .right-top {{ grid-column:2; grid-row:1; display:flex; flex-direction:column; min-height:0; background:white; border-bottom:1px solid #d9dce5; }}
        .right-bottom {{ grid-column:2; grid-row:2; overflow:auto; background:white; }}
        .panel-header {{ flex:0 0 auto; padding:7px 10px; background:#f8f9fb; border-bottom:1px solid #d9dce5; font-weight:800; font-size:0.86rem; }}
        .image-id-header {{ color:white; background:#667eea; }}
        .map-controls {{ flex:0 0 auto; padding:7px 9px; background:#f7f8fc; border-bottom:1px solid #d9dce5; font-size:0.78rem; }}
        .map-controls-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }}
        .control-col-title {{ font-weight:900; color:#5867e8; font-size:0.75rem; text-transform:uppercase; margin-bottom:2px; }}
        .map-controls label {{ display:block; margin:1px 0; cursor:pointer; font-weight:600; }}
        .map-controls input {{ margin-right:5px; }}
        #map {{ flex:1 1 auto; min-height:0; width:100%; }}
        .image-container {{ flex:1; min-height:0; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; background:#eeeeef; }}
        .image-container img {{ max-width:98%; max-height:98%; object-fit:contain; border-radius:4px; }}
        .missing-image {{ color:#6f7584; text-align:center; font-weight:700; }}
        .expand-btn {{ position:absolute; top:8px; right:8px; z-index:20; background:rgba(0,0,0,0.62); color:white; border:0; border-radius:5px; padding:6px 10px; font-size:0.8rem; }}
        .stats-container {{ padding:12px 14px; }}
        .stats-section {{ border-bottom:1px solid #eceff5; padding-bottom:12px; margin-bottom:12px; }}
        .stats-section h6 {{ color:#667eea; font-weight:900; margin-bottom:8px; }}
        .stats-dl {{ display:grid; grid-template-columns:145px minmax(0,1fr); gap:3px 10px; font-size:0.86rem; margin:0; }}
        .stats-dl dt {{ color:#30364a; font-weight:800; }}
        .stats-dl dd {{ margin:0; color:#596071; min-width:0; word-break:break-word; }}
        .path-text {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:0.78rem; }}
        .two-col-stats {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
        .external-links a {{ margin-right:10px; font-size:0.82rem; }}
        .artifact-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:10px; }}
        .artifact-grid figure {{ margin:0; border:1px solid #e3e6ef; border-radius:8px; background:#fafbff; padding:6px; }}
        .artifact-grid img {{ width:100%; max-height:130px; object-fit:contain; display:block; background:#eee; }}
        .artifact-grid figcaption {{ font-size:0.75rem; color:#596071; margin-top:4px; font-weight:700; }}
        .modal-view {{ display:none; position:fixed; inset:0; z-index:2000; background:rgba(0,0,0,0.9); align-items:center; justify-content:center; padding:24px; }}
        .modal-view.active {{ display:flex; }}
        .modal-view img {{ max-width:95vw; max-height:92vh; object-fit:contain; }}
        .close-btn {{ position:absolute; top:8px; right:18px; color:white; font-size:34px; background:none; border:0; }}
        .yaw-label {{ background:rgba(255,255,255,0.92); border:1px solid rgba(0,0,0,0.25); border-radius:4px; padding:1px 4px; font-size:11px; font-weight:900; white-space:nowrap; }}
        .leaflet-popup-content a {{ font-weight:800; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark d-flex align-items-center justify-content-between">
        <div>
            <a href="{escape(run_name)}_dashboard.html" class="back-btn"><i class="fas fa-arrow-left"></i> Back</a>
            <a href="{escape(run_name)}_analysis.html" class="analysis-btn"><i class="fas fa-chart-line"></i> Analysis</a>
        </div>
        <span class="navbar-brand"><i class="fas fa-image"></i> Image {escape(image_id)} Details</span>
    </nav>

    <div class="container-detail">
        <div class="left-panel">
            <div class="panel-header"><i class="fas fa-map-location-dot"></i> Geolocation Context</div>
            <div class="map-controls">
                <div class="map-controls-grid">
                    <div>
                        <div class="control-col-title">Basemap</div>
                        <label><input type="radio" name="basemap" value="osm" checked> OpenStreetMap</label>
                        <label><input type="radio" name="basemap" value="model_osm" id="modelOsmBasemap"> Model OSM raster JPG</label>
                        <label><input type="radio" name="basemap" value="satellite"> Esri Satellite</label>
                    </div>
                    <div>
                        <div class="control-col-title">Layers</div>
                        <label><input type="checkbox" id="boundaryLayer" checked> Request tile (~{fmt_num(tile_side,0,' m')} side)</label>
                        <label><input type="checkbox" id="predictionLayer"> Prediction map</label>
                        <label><input type="checkbox" id="neuralLayer"> Neural activation</label>
                    </div>
                </div>
            </div>
            <div id="map"></div>
        </div>

        <div class="right-top">
            <div class="panel-header image-id-header"><i class="fas fa-photo-film"></i> Rectified Image</div>
            <div class="image-container">
                <button class="expand-btn" onclick="expandImage()"><i class="fas fa-expand"></i> Expand</button>
                {rectified_img}
            </div>
        </div>

        <div class="right-bottom">
            <div class="stats-container">
                {stats_html}
            </div>
        </div>
    </div>

    <div id="imageModal" class="modal-view">
        <button class="close-btn" onclick="closeImageModal()">&times;</button>
        <img id="expandedImage" src="{escape(rectified_url)}" alt="Expanded rectified image">
    </div>

    <script>
        const imageData = {detail_json};

        function isValidNumber(x) {{ return typeof x === 'number' && Number.isFinite(x); }}
        function latLngValid(lat, lon) {{ return isValidNumber(lat) && isValidNumber(lon) && !(lat === 0 && lon === 0); }}
        function htmlEscape(s) {{ return String(s ?? '').replace(/[&<>'"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}}[c])); }}

        const center = [imageData.center_lat || 0, imageData.center_lon || 0];
        const map = L.map('map', {{ preferCanvas: true }}).setView(center, 17);

        const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 20
        }});
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: '© Esri',
            maxZoom: 20
        }});

        function destinationPoint(lat, lon, bearingDeg, distanceM) {{
            const rad = Math.PI / 180;
            const bearing = bearingDeg * rad;
            const north = Math.cos(bearing) * distanceM;
            const east = Math.sin(bearing) * distanceM;
            const dLat = north / 111320.0;
            const dLon = east / (111320.0 * Math.cos((lat || 0) * rad));
            return [lat + dLat, lon + dLon];
        }}

        function fallbackBounds() {{
            const lat = imageData.gt_lat || imageData.center_lat || 0;
            const lon = imageData.gt_lon || imageData.center_lon || 0;
            const r = imageData.tile_radius_meters || 136;
            const sw = destinationPoint(lat, lon, 225, Math.sqrt(2) * r);
            const ne = destinationPoint(lat, lon, 45, Math.sqrt(2) * r);
            return [sw, ne];
        }}

        function axisBounds() {{
            const b = imageData.tile_bounds_latlon;
            if (b && isValidNumber(b.south) && isValidNumber(b.west) && isValidNumber(b.north) && isValidNumber(b.east)) {{
                return [[b.south, b.west], [b.north, b.east]];
            }}
            return fallbackBounds();
        }}

        function cornersAvailable() {{
            const c = imageData.tile_corners_latlon;
            return c && c.top_left && c.top_right && c.bottom_left && c.bottom_right;
        }}

        function corner(name) {{
            const c = imageData.tile_corners_latlon[name];
            return [c.lat, c.lon];
        }}

        function makeGeoImage(url, opacity, zIndex) {{
            if (!url) return null;
            let layer;
            try {{
                if (cornersAvailable() && L.imageOverlay.rotated) {{
                    layer = L.imageOverlay.rotated(url, corner('top_left'), corner('top_right'), corner('bottom_left'), {{ opacity }});
                }} else {{
                    layer = L.imageOverlay(url, axisBounds(), {{ opacity }});
                }}
                if (layer.setZIndex) layer.setZIndex(zIndex);
                return layer;
            }} catch (err) {{
                console.warn('Could not create image overlay:', err);
                return L.imageOverlay(url, axisBounds(), {{ opacity }});
            }}
        }}

        function makeBoundary() {{
            if (cornersAvailable()) {{
                return L.polygon([corner('top_left'), corner('top_right'), corner('bottom_right'), corner('bottom_left')], {{
                    color:'#222', weight:2, opacity:0.85, fill:false, dashArray:'6,5'
                }});
            }}
            return L.rectangle(axisBounds(), {{ color:'#222', weight:2, opacity:0.85, fill:false, dashArray:'6,5' }});
        }}

        const modelOsmOverlay = makeGeoImage(imageData.urls.osm_tile, 1.0, 50);
        const predictionOverlay = makeGeoImage(imageData.urls.prediction_map, 0.64, 80);
        const neuralOverlay = makeGeoImage(imageData.urls.neural_map, 0.64, 90);
        const boundary = makeBoundary();

        let currentBase = null;
        function setBasemap(name) {{
            [osmLayer, satelliteLayer, modelOsmOverlay].forEach(layer => {{ if (layer && map.hasLayer(layer)) map.removeLayer(layer); }});
            if (name === 'satellite') currentBase = satelliteLayer;
            else if (name === 'model_osm' && modelOsmOverlay) currentBase = modelOsmOverlay;
            else currentBase = osmLayer;
            currentBase.addTo(map);
            // Re-add checked overlays so they remain above the selected basemap.
            syncLayer('boundaryLayer', boundary);
            syncLayer('predictionLayer', predictionOverlay);
            syncLayer('neuralLayer', neuralOverlay);
        }}

        function syncLayer(controlId, layer) {{
            const control = document.getElementById(controlId);
            if (!control) return;
            if (!layer) {{ control.checked = false; control.disabled = true; return; }}
            if (control.checked && !map.hasLayer(layer)) layer.addTo(map);
            if (!control.checked && map.hasLayer(layer)) map.removeLayer(layer);
            if (layer.bringToFront) layer.bringToFront();
        }}

        document.querySelectorAll('input[name="basemap"]').forEach(radio => {{
            radio.addEventListener('change', e => setBasemap(e.target.value));
        }});
        ['boundaryLayer','predictionLayer','neuralLayer'].forEach(id => {{
            document.getElementById(id).addEventListener('change', () => {{
                syncLayer('boundaryLayer', boundary);
                syncLayer('predictionLayer', predictionOverlay);
                syncLayer('neuralLayer', neuralOverlay);
            }});
        }});

        if (!modelOsmOverlay) document.getElementById('modelOsmBasemap').disabled = true;
        setBasemap('osm');

        const gtColor = '#003B73';       // dark blue
        const predColor = '#63C7FF';     // light blue
        const featuresForFit = [];

        function popupHtml(label, lat, lon, yaw, mapsUrl, streetUrl) {{
            const maps = mapsUrl ? `<a target="_blank" rel="noopener" href="${{htmlEscape(mapsUrl)}}">Google Maps</a>` : '';
            const street = streetUrl ? ` · <a target="_blank" rel="noopener" href="${{htmlEscape(streetUrl)}}">Street View</a>` : '';
            return `<strong>${{htmlEscape(label)}}</strong><br>${{lat.toFixed(7)}}, ${{lon.toFixed(7)}}<br>Yaw: ${{isValidNumber(yaw) ? yaw.toFixed(2) + '°' : '—'}}<br>${{maps}}${{street}}`;
        }}

        function addMarker(label, lat, lon, yaw, color, mapsUrl, streetUrl) {{
            if (!latLngValid(lat, lon)) return null;
            const marker = L.circleMarker([lat, lon], {{
                radius:8, fillColor:color, color:'#0b1f35', weight:2, opacity:1, fillOpacity:0.9
            }}).bindPopup(popupHtml(label, lat, lon, yaw, mapsUrl, streetUrl)).addTo(map);
            featuresForFit.push(marker);
            drawYawChevron(lat, lon, yaw, color, label);
            return marker;
        }}

        function drawYawChevron(lat, lon, yaw, color, label) {{
            if (!isValidNumber(yaw)) return;
            const armLength = 34;
            const labelOffset = 24;
            const leftBack = destinationPoint(lat, lon, yaw + 180 - 22.5, armLength);
            const rightBack = destinationPoint(lat, lon, yaw + 180 + 22.5, armLength);
            const apex = [lat, lon];
            const arm1 = L.polyline([leftBack, apex], {{ color, weight:5, opacity:0.95 }}).addTo(map);
            const arm2 = L.polyline([rightBack, apex], {{ color, weight:5, opacity:0.95 }}).addTo(map);
            featuresForFit.push(arm1, arm2);
            const labelPos = destinationPoint(lat, lon, yaw, labelOffset);
            L.marker(labelPos, {{
                interactive:false,
                icon:L.divIcon({{ className:'', html:`<div class="yaw-label">${{htmlEscape(label)}} ${{yaw.toFixed(1)}}°</div>` }})
            }}).addTo(map);
        }}

        addMarker('GT', imageData.gt_lat, imageData.gt_lon, imageData.gt_yaw, gtColor, imageData.links.gt_maps, imageData.links.gt_streetview);
        addMarker('Pred', imageData.pred_lat, imageData.pred_lon, imageData.pred_yaw, predColor, imageData.links.pred_maps, imageData.links.pred_streetview);

        try {{
            const fitLayers = [boundary, ...featuresForFit].filter(Boolean);
            if (fitLayers.length) {{
                const group = L.featureGroup(fitLayers);
                map.fitBounds(group.getBounds().pad(0.10));
            }}
        }} catch (err) {{ console.warn('Could not fit bounds:', err); }}

        function expandImage() {{
            if (imageData.urls.rectified) document.getElementById('imageModal').classList.add('active');
        }}
        function closeImageModal() {{ document.getElementById('imageModal').classList.remove('active'); }}
        document.getElementById('imageModal').addEventListener('click', e => {{ if (e.target.id === 'imageModal') closeImageModal(); }});
        document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeImageModal(); }});
    </script>
</body>
</html>"""

# ============================================================================
# Analysis page
# ============================================================================


def save_analysis_plots(df_valid: pd.DataFrame, assets_dir: Path) -> List[Tuple[str, str]]:
    plots: List[Tuple[str, str]] = []
    if df_valid.empty:
        return plots

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Matplotlib unavailable, skipping plots: {exc}")
        return plots

    assets_dir.mkdir(parents=True, exist_ok=True)

    def save_current(filename: str, title: str) -> None:
        path = assets_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=145, bbox_inches="tight")
        plt.close()
        plots.append((title, filename))

    dist = df_valid["distance_m"].dropna()
    yaw_signed = df_valid["yaw_error_signed"].dropna()
    yaw_abs = df_valid["yaw_error_abs"].dropna()
    prob = df_valid["pred_probability"].dropna()

    if not dist.empty:
        plt.figure(figsize=(9, 4.8))
        plt.hist(dist, bins=50, edgecolor="black", alpha=0.75)
        plt.axvline(dist.mean(), linestyle="--", linewidth=2, label=f"Mean {dist.mean():.2f} m")
        plt.axvline(dist.median(), linestyle="--", linewidth=2, label=f"Median {dist.median():.2f} m")
        plt.xlabel("Distance error (m)")
        plt.ylabel("Count")
        plt.title("Distance Error Distribution")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("distance_hist.png", "Distance histogram")

        plt.figure(figsize=(9, 4.8))
        plt.hist(dist, bins=50, edgecolor="black", alpha=0.75)
        plt.yscale("log")
        plt.axvline(dist.mean(), linestyle="--", linewidth=2, label=f"Mean {dist.mean():.2f} m")
        plt.axvline(dist.median(), linestyle="--", linewidth=2, label=f"Median {dist.median():.2f} m")
        plt.xlabel("Distance error (m)")
        plt.ylabel("Count (log scale)")
        plt.title("Distance Error Distribution, Log Count")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("distance_hist_log.png", "Distance histogram, log count")

        sorted_dist = np.sort(dist.to_numpy())
        cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        plt.figure(figsize=(9, 4.8))
        plt.plot(sorted_dist, cdf, linewidth=2)
        plt.axhline(0.5, linestyle="--", alpha=0.65, label="50%")
        plt.axhline(0.9, linestyle="--", alpha=0.65, label="90%")
        plt.xlabel("Distance error (m)")
        plt.ylabel("Cumulative probability")
        plt.title("Distance Error CDF")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("distance_cdf.png", "Distance CDF")

    if not yaw_signed.empty:
        plt.figure(figsize=(9, 4.8))
        plt.hist(yaw_signed, bins=50, edgecolor="black", alpha=0.75)
        plt.axvline(yaw_signed.mean(), linestyle="--", linewidth=2, label=f"Mean {yaw_signed.mean():.2f}°")
        plt.axvline(0, linewidth=1, alpha=0.5)
        plt.xlabel("Signed yaw error (°)")
        plt.ylabel("Count")
        plt.title("Signed Yaw Error Distribution")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("yaw_signed_hist.png", "Signed yaw histogram")

    if not yaw_abs.empty:
        plt.figure(figsize=(9, 4.8))
        plt.hist(yaw_abs, bins=50, edgecolor="black", alpha=0.75)
        plt.axvline(yaw_abs.mean(), linestyle="--", linewidth=2, label=f"Mean {yaw_abs.mean():.2f}°")
        plt.axvline(yaw_abs.median(), linestyle="--", linewidth=2, label=f"Median {yaw_abs.median():.2f}°")
        plt.xlabel("Absolute yaw error (°)")
        plt.ylabel("Count")
        plt.title("Absolute Yaw Error Distribution")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("yaw_abs_hist.png", "Absolute yaw histogram")

        sorted_yaw = np.sort(yaw_abs.to_numpy())
        cdf = np.arange(1, len(sorted_yaw) + 1) / len(sorted_yaw)
        plt.figure(figsize=(9, 4.8))
        plt.plot(sorted_yaw, cdf, linewidth=2)
        plt.axhline(0.5, linestyle="--", alpha=0.65, label="50%")
        plt.axhline(0.9, linestyle="--", alpha=0.65, label="90%")
        plt.xlabel("Absolute yaw error (°)")
        plt.ylabel("Cumulative probability")
        plt.title("Yaw Error CDF")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("yaw_cdf.png", "Yaw CDF")

    if not prob.empty:
        plt.figure(figsize=(9, 4.8))
        plt.hist(prob, bins=50, edgecolor="black", alpha=0.75)
        plt.axvline(prob.mean(), linestyle="--", linewidth=2, label=f"Mean {prob.mean():.3e}")
        plt.axvline(prob.max(), linestyle="--", linewidth=2, label=f"Max {prob.max():.3e}")
        plt.xlabel("pred_probability")
        plt.ylabel("Count")
        plt.title("Prediction Probability Distribution")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("probability_hist.png", "Prediction probability histogram")

    percentiles = [50, 75, 90, 95, 99]
    if not dist.empty:
        vals = [np.percentile(dist, p) for p in percentiles]
        plt.figure(figsize=(8, 4.5))
        plt.bar([str(p) for p in percentiles], vals)
        plt.xlabel("Percentile")
        plt.ylabel("Distance error (m)")
        plt.title("Distance Error Percentiles")
        plt.grid(True, axis="y", alpha=0.25)
        save_current("distance_percentiles.png", "Distance percentiles")
    if not yaw_abs.empty:
        vals = [np.percentile(yaw_abs, p) for p in percentiles]
        plt.figure(figsize=(8, 4.5))
        plt.bar([str(p) for p in percentiles], vals)
        plt.xlabel("Percentile")
        plt.ylabel("Absolute yaw error (°)")
        plt.title("Yaw Error Percentiles")
        plt.grid(True, axis="y", alpha=0.25)
        save_current("yaw_percentiles.png", "Yaw percentiles")

    if "quality_score" in df_valid.columns and df_valid["quality_score"].notna().any():
        qs = df_valid["quality_score"].dropna()
        plt.figure(figsize=(9, 4.8))
        plt.hist(qs, bins=50, edgecolor="black", alpha=0.75)
        plt.axvline(qs.mean(), linestyle="--", linewidth=2, label=f"Mean {qs.mean():.1f}")
        plt.axvline(qs.median(), linestyle="--", linewidth=2, label=f"Median {qs.median():.1f}")
        plt.xlabel("Quality score (0-100)")
        plt.ylabel("Count")
        plt.title("Combined Quality Score Distribution")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_current("quality_score_hist.png", "Quality score histogram")

    if "error_category" in df_valid.columns:
        counts = df_valid["error_category"].value_counts().reindex(["Excellent", "Good", "Acceptable", "Poor"]).dropna()
        if not counts.empty:
            plt.figure(figsize=(8, 4.5))
            plt.bar(counts.index.astype(str), counts.values)
            plt.xlabel("Category")
            plt.ylabel("Count")
            plt.title("Prediction Error Categories")
            plt.grid(True, axis="y", alpha=0.25)
            save_current("category_bar.png", "Error categories")

    # Regional bin heatmaps, skipping folium maps from the notebook.
    if {"gt_latitude", "gt_longitude", "distance_m", "yaw_error_abs"}.issubset(df_valid.columns) and len(df_valid) >= 5:
        regional = df_valid.copy()
        try:
            regional["lat_bin"] = pd.cut(regional["gt_latitude"], bins=min(6, max(2, regional["gt_latitude"].nunique())))
            regional["lon_bin"] = pd.cut(regional["gt_longitude"], bins=min(6, max(2, regional["gt_longitude"].nunique())))
            for metric, filename, title, label in [
                ("distance_m", "regional_distance_heatmap.png", "Mean Distance Error by Lat/Lon Bin", "Mean distance (m)"),
                ("yaw_error_abs", "regional_yaw_heatmap.png", "Mean Yaw Error by Lat/Lon Bin", "Mean yaw (°)"),
            ]:
                pivot = regional.pivot_table(values=metric, index="lat_bin", columns="lon_bin", aggfunc="mean", observed=False)
                if pivot.empty:
                    continue
                plt.figure(figsize=(9, 6))
                img = plt.imshow(pivot.to_numpy(), aspect="auto")
                plt.colorbar(img, label=label)
                plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=7)
                plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index], fontsize=7)
                plt.title(title)
                plt.xlabel("Longitude bin")
                plt.ylabel("Latitude bin")
                save_current(filename, title)
        except Exception as exc:
            print(f"[WARN] Could not create regional heatmaps: {exc}")

    return plots


def add_quality_columns(df_valid: pd.DataFrame) -> pd.DataFrame:
    df_valid = df_valid.copy()
    if df_valid.empty:
        df_valid["quality_score"] = np.nan
        df_valid["error_category"] = np.nan
        df_valid["loss_combined"] = np.nan
        return df_valid

    dist = df_valid["distance_m"].astype(float)
    yaw = df_valid["yaw_error_abs"].astype(float)
    median_dist = max(float(dist.median()), 1e-9)
    median_yaw = max(float(yaw.median()), 1e-9)
    df_valid["loss_combined"] = ((dist / median_dist) + (yaw / median_yaw)) / 2.0

    max_dist = max(float(dist.quantile(0.95)), 1e-9)
    max_yaw = max(float(yaw.quantile(0.95)), 1e-9)
    df_valid["quality_score"] = 100.0 * (
        0.5 * (1 - np.minimum(dist / max_dist, 1)) +
        0.5 * (1 - np.minimum(yaw / max_yaw, 1))
    )

    dist_threshold = median_dist
    yaw_threshold = median_yaw
    def categorize(row: pd.Series) -> str:
        d = row["distance_m"]
        y = row["yaw_error_abs"]
        if d < dist_threshold and y < yaw_threshold:
            return "Excellent"
        if d < dist_threshold * 1.5 and y < yaw_threshold * 1.5:
            return "Good"
        if d < dist_threshold * 2.5 or y < yaw_threshold * 2.5:
            return "Acceptable"
        return "Poor"
    df_valid["error_category"] = df_valid.apply(categorize, axis=1)
    return df_valid


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return '<p class="muted">No rows.</p>'
    return df.head(max_rows).to_html(index=False, escape=True, classes="analysis-table", border=0)


def generate_analysis_page(run_dir: Path, run_name: str, df: pd.DataFrame, output_dir: Path) -> str:
    valid = df.loc[valid_prediction_mask(df)].copy()
    valid = add_quality_columns(valid)

    assets_dir = output_dir / f"{run_name}_analysis_assets"
    plots = save_analysis_plots(valid, assets_dir)

    if not valid.empty:
        summary_rows = [
            ("Total rows loaded", len(df)),
            ("Valid predictions analyzed", len(valid)),
            ("Excluded rows", len(df) - len(valid)),
            ("Distance MAE", fmt_num(valid["distance_m"].mean(), 2, " m")),
            ("Distance RMSE", fmt_num(np.sqrt((valid["distance_m"] ** 2).mean()), 2, " m")),
            ("Distance median", fmt_num(valid["distance_m"].median(), 2, " m")),
            ("Distance 90th percentile", fmt_num(np.percentile(valid["distance_m"], 90), 2, " m")),
            ("Yaw MAE", fmt_num(valid["yaw_error_abs"].mean(), 2, "°")),
            ("Yaw RMSE", fmt_num(np.sqrt((valid["yaw_error_abs"] ** 2).mean()), 2, "°")),
            ("Yaw median", fmt_num(valid["yaw_error_abs"].median(), 2, "°")),
            ("Yaw 90th percentile", fmt_num(np.percentile(valid["yaw_error_abs"], 90), 2, "°")),
            ("Mean pred_probability", fmt_prob(valid["pred_probability"].mean())),
            ("Max pred_probability", fmt_prob(valid["pred_probability"].max())),
            ("Mean quality score", fmt_num(valid["quality_score"].mean(), 2)),
            ("Median combined loss", fmt_num(valid["loss_combined"].median(), 4)),
        ]
    else:
        summary_rows = [("Total rows loaded", len(df)), ("Valid predictions analyzed", 0)]

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    assets_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = assets_dir / "analysis_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    valid_export_cols = [
        col for col in [
            "image_id", "image_path", "h5_id_dataset",
            "gt_latitude", "gt_longitude", "gt_yaw",
            "pred_latitude", "pred_longitude", "pred_x_meters", "pred_y_meters", "pred_yaw",
            "pred_probability", "distance_m", "yaw_error_signed", "yaw_error_abs",
            "loss_combined", "quality_score", "error_category",
        ] if col in valid.columns
    ]
    valid_csv = assets_dir / "valid_results_with_errors.csv"
    if valid_export_cols:
        valid[valid_export_cols].to_csv(valid_csv, index=False)

    summary_table = dataframe_to_html_table(summary_df, max_rows=100)
    categories_table = ""
    if not valid.empty and "error_category" in valid.columns:
        cat = valid["error_category"].value_counts().reindex(["Excellent", "Good", "Acceptable", "Poor"]).fillna(0).astype(int)
        cat_df = pd.DataFrame({"Category": cat.index, "Count": cat.values, "Percent": [f"{100*c/len(valid):.1f}%" for c in cat.values]})
        categories_table = dataframe_to_html_table(cat_df, max_rows=20)

    best_cols = [c for c in ["image_id", "distance_m", "yaw_error_abs", "pred_probability", "quality_score", "error_category"] if c in valid.columns]
    best_table = dataframe_to_html_table(valid.sort_values("quality_score", ascending=False)[best_cols], max_rows=10) if not valid.empty and best_cols else ""
    worst_table = dataframe_to_html_table(valid.sort_values("loss_combined", ascending=False)[best_cols], max_rows=10) if not valid.empty and best_cols else ""

    plot_cards = "".join(
        f"""
        <section class="plot-card">
            <h5>{escape(title)}</h5>
            <img src="{escape(relative_url(assets_dir / filename, output_dir))}" alt="{escape(title)}">
        </section>
        """
        for title, filename in plots
    ) or '<p class="muted">No plots were generated because no valid predictions were available or matplotlib was unavailable.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis - {escape(run_name)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ background:#f5f6fb; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif; color:#252b3c; }}
        .navbar {{ background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:0.35rem 1rem; }}
        .navbar a {{ color:white; text-decoration:none; background:rgba(255,255,255,0.16); border-radius:6px; padding:0.25rem 0.55rem; font-size:0.85rem; margin-right:0.4rem; }}
        .page {{ padding:18px; }}
        .section {{ background:white; border-radius:12px; padding:16px; margin-bottom:16px; box-shadow:0 2px 10px rgba(20,20,60,0.08); }}
        h1 {{ font-size:1.35rem; margin-bottom:6px; }}
        h4 {{ font-size:1.05rem; font-weight:900; color:#2b3150; margin-bottom:12px; }}
        .analysis-table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
        .analysis-table th {{ text-align:left; background:#667eea; color:white; padding:7px 8px; }}
        .analysis-table td {{ text-align:left; border-bottom:1px solid #edf0f6; padding:7px 8px; }}
        .plot-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }}
        .plot-card {{ background:white; border-radius:12px; padding:14px; box-shadow:0 2px 10px rgba(20,20,60,0.08); }}
        .plot-card h5 {{ font-size:0.95rem; font-weight:900; color:#5968d8; }}
        .plot-card img {{ width:100%; height:auto; display:block; }}
        .muted {{ color:#7f8799; }}
        .download-links a {{ margin-right:12px; font-weight:800; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <a href="{escape(run_name)}_dashboard.html"><i class="fas fa-arrow-left"></i> Dashboard</a>
    </nav>
    <main class="page">
        <section class="section">
            <h1>Notebook-style analysis: {escape(run_name)}</h1>
            <p class="muted">This page implements the numerical/statistical parts of the provided notebook and skips the folium map, since the detail pages already contain the interactive map.</p>
            <div class="download-links">
                <a href="{escape(relative_url(summary_csv, output_dir))}"><i class="fas fa-file-csv"></i> analysis_summary.csv</a>
                <a href="{escape(relative_url(valid_csv, output_dir))}"><i class="fas fa-file-csv"></i> valid_results_with_errors.csv</a>
            </div>
        </section>
        <section class="section">
            <h4>Summary statistics</h4>
            {summary_table}
        </section>
        <section class="section">
            <h4>Error categories</h4>
            {categories_table or '<p class="muted">No categories available.</p>'}
        </section>
        <section class="section">
            <h4>Best rows by quality score</h4>
            {best_table or '<p class="muted">No valid rows.</p>'}
        </section>
        <section class="section">
            <h4>Worst rows by combined loss</h4>
            {worst_table or '<p class="muted">No valid rows.</p>'}
        </section>
        <div class="plot-grid">
            {plot_cards}
        </div>
    </main>
</body>
</html>"""

# ============================================================================
# Main execution
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OrienterNet HTML dashboard pages.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing results.csv and artifacts/. Default: latest folder inside --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(RUNS_PATH),
        help="Directory containing run folders. Used only when --run-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where HTML pages are written. Default: parent of run-dir.",
    )
    parser.add_argument(
        "--no-detail-pages",
        action="store_true",
        help="Only write the dashboard and analysis pages, not per-image detail pages.",
    )
    parser.add_argument(
        "--max-detail-pages",
        type=int,
        default=None,
        help="Optional debug limit for the number of per-image detail pages to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else get_latest_run_dir(runs_dir).resolve()
    run_name = run_dir.name
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("OrienterNet Results Dashboard Generator")
    print("=" * 72)
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Output directory: {output_dir}")

    df = load_results_csv(run_dir)
    print(f"[OK] Loaded {len(df)} rows from results.csv")

    dashboard_html = generate_main_html(run_dir, run_name, df, output_dir)
    dashboard_path = output_dir / f"{run_name}_dashboard.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    print(f"[OK] Wrote dashboard: {dashboard_path}")

    analysis_html = generate_analysis_page(run_dir, run_name, df, output_dir)
    analysis_path = output_dir / f"{run_name}_analysis.html"
    analysis_path.write_text(analysis_html, encoding="utf-8")
    print(f"[OK] Wrote analysis: {analysis_path}")

    if not args.no_detail_pages:
        detail_count = 0
        detail_rows = df
        if args.max_detail_pages is not None:
            if args.max_detail_pages < 0:
                raise ValueError("--max-detail-pages must be >= 0")
            detail_rows = df.head(args.max_detail_pages)
        for _, row in detail_rows.iterrows():
            image_id = format_image_id(row.get("image_id"))
            if not image_id:
                continue
            detail_html = generate_detail_html(row, run_dir, run_name, output_dir)
            detail_path = output_dir / f"{run_name}_detail_{image_id}.html"
            detail_path.write_text(detail_html, encoding="utf-8")
            detail_count += 1
        suffix = "" if args.max_detail_pages is None else f" (limited by --max-detail-pages={args.max_detail_pages})"
        print(f"[OK] Wrote {detail_count} detail pages{suffix}")

    print("=" * 72)
    print(f"Open: {dashboard_path}")


if __name__ == "__main__":
    main()
