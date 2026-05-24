#!/usr/bin/env python3
"""
OrienterNet Results Dashboard Generator

Generates an interactive HTML dashboard for viewing OrienterNet prediction results.
The dashboard includes:
- a main overview table with clickable rows,
- a per-image detail template and on-demand detail generator,
- an analysis page based on the supplied analysis notebook, excluding the folium maps.

CONFIGURATION:
- BASE_DATA_DIR: Path to the data directory containing runs
- RUNS_PATH: Path to the runs folder
"""

import os
import sys

# Fix encoding on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

import base64
import io
import json
import re
from datetime import datetime
from html import escape
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# PATH CONFIGURATION - ADJUST THESE IF YOUR STRUCTURE CHANGES
# ============================================================================
BASE_PROJECT_DIR = Path(__file__).parent
BASE_DATA_DIR = BASE_PROJECT_DIR / "data"
RUNS_PATH = BASE_DATA_DIR / "runs"
OUTPUT_DIR = RUNS_PATH  # Where to save the final HTML pages

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_latest_run_dir() -> Path:
    """Get the latest run directory."""
    run_dirs = sorted(
        [d for d in RUNS_PATH.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {RUNS_PATH}")
    return run_dirs[0]


def get_run_name(run_dir: Path) -> str:
    """Extract run name from directory path."""
    return run_dir.name


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known numeric columns to numeric values when present."""
    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_results_csv(run_dir: Path) -> pd.DataFrame:
    """Load the results CSV from a run directory."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    return enrich_results(df)


def load_artifact_json(artifact_dir: Path, filename: str) -> Dict[str, Any]:
    """Safely load a JSON file from an artifact directory."""
    json_path = artifact_dir / filename
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Error reading {json_path}: {exc}")
        return {}


def image_to_base64(image_path: Path) -> str:
    """Convert an image file to a base64 data URI."""
    if not image_path.exists():
        return ""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.suffix.lower()[1:]
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{data}"
    except Exception as exc:
        print(f"Error converting image to base64: {exc}")
        return ""


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance between two WGS84 points in meters."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180]."""
    if pd.isna(angle):
        return np.nan
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def parse_tile_size_meters(run_name: str, default: int = 136) -> int:
    """Extract the tile size from run names like process_136t_256r_og."""
    match = re.search(r"(?:^|_)process_(\d+)t_", run_name)
    if not match:
        match = re.search(r"(\d+)t_", run_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return default


def enrich_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add shared error metrics used by the dashboard, detail pages, and analysis."""
    df = df.copy()
    df = _coerce_numeric_columns(df)

    coord_cols = ["pred_longitude", "pred_latitude", "gt_longitude", "gt_latitude"]
    if all(col in df.columns for col in coord_cols):
        df["distance_m"] = df.apply(
            lambda row: haversine(
                row["pred_longitude"],
                row["pred_latitude"],
                row["gt_longitude"],
                row["gt_latitude"],
            )
            if all(pd.notna([row[c] for c in coord_cols]))
            else np.nan,
            axis=1,
        )
    elif "distance_m" not in df.columns:
        df["distance_m"] = np.nan

    yaw_cols = ["pred_yaw", "gt_yaw"]
    if all(col in df.columns for col in yaw_cols):
        df["yaw_error_signed"] = df.apply(
            lambda row: normalize_angle(row["pred_yaw"] - row["gt_yaw"])
            if all(pd.notna([row[c] for c in yaw_cols]))
            else np.nan,
            axis=1,
        )
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

    return df


def safe_mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if pd.notna(value) else np.nan


def safe_max(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").max()
    return float(value) if pd.notna(value) else np.nan


def format_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
    except Exception:
        pass
    return str(value)


def format_meters(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f} m"


def format_degrees(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}°"


def format_probability(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def detail_filename(run_name: str, image_id: Any) -> str:
    return f"{run_name}_detail_{format_image_id(image_id)}.html"


def detail_link(run_name: str, image_id: Any, label: Optional[str] = None) -> str:
    image_label = label or format_image_id(image_id)
    return f'<a href="{detail_filename(run_name, image_id)}">{escape(image_label)}</a>'


def dataframe_to_html_left(df: pd.DataFrame, classes: str) -> str:
    """Render a compact HTML table. CSS later forces left alignment."""
    html = df.to_html(classes=classes, index=False, escape=False)
    # Pandas adds right-alignment inline styles for numeric data; remove the most common one.
    return html.replace('<tr style="text-align: right;">', "<tr>")


# ============================================================================
# MAIN PAGE HELPERS
# ============================================================================


def generate_summary_html(df: pd.DataFrame) -> str:
    total_images = len(df)
    valid_distance = int(df["distance_m"].notna().sum()) if "distance_m" in df else 0
    dist_avg = safe_mean(df["distance_m"])
    yaw_avg = safe_mean(df["yaw_error_abs"])
    prob_avg = safe_mean(df["pred_probability"])
    prob_max = safe_max(df["pred_probability"])

    max_prob_id = ""
    if "pred_probability" in df.columns and df["pred_probability"].notna().any():
        try:
            max_prob_idx = df["pred_probability"].idxmax()
            max_prob_id = format_image_id(df.loc[max_prob_idx, "image_id"])
        except Exception:
            max_prob_id = ""

    cards = [
        ("Total Images", f"{total_images}"),
        ("Valid Predictions", f"{valid_distance}"),
        ("Avg Distance Error", format_meters(dist_avg)),
        ("Avg Yaw Error", format_degrees(yaw_avg)),
        ("Avg Pred Probability", format_probability(prob_avg)),
        (
            "Max Pred Probability",
            f"{format_probability(prob_max)}" + (f" <span class='muted'>(ID {escape(max_prob_id)})</span>" if max_prob_id else ""),
        ),
    ]

    card_html = "".join(
        f"""
        <div class="summary-card">
            <div class="summary-label">{label}</div>
            <div class="summary-value">{value or 'N/A'}</div>
        </div>
        """
        for label, value in cards
    )
    return f'<div class="summary-grid">{card_html}</div>'


def _metric_rows_html(
    df: pd.DataFrame,
    run_name: str,
    metric_col: str,
    formatter,
    ascending: bool,
    limit: int = 5,
) -> str:
    if metric_col not in df.columns:
        return '<div class="empty-note">Metric not available.</div>'

    subset = df[df[metric_col].notna()].sort_values(metric_col, ascending=ascending).head(limit)
    if subset.empty:
        return '<div class="empty-note">No valid rows.</div>'

    rows = []
    for _, row in subset.iterrows():
        image_id = row.get("image_id", "")
        context = []
        if "distance_m" in row and pd.notna(row["distance_m"]):
            context.append(f"dist {format_meters(row['distance_m'])}")
        if "yaw_error_abs" in row and pd.notna(row["yaw_error_abs"]):
            context.append(f"yaw {format_degrees(row['yaw_error_abs'])}")
        if "pred_probability" in row and pd.notna(row["pred_probability"]):
            context.append(f"prob {format_probability(row['pred_probability'])}")
        rows.append(
            f"""
            <div class="interesting-row" onclick="window.location.href='{detail_filename(run_name, image_id)}'">
                <span class="interesting-id">ID {escape(format_image_id(image_id))}</span>
                <span class="interesting-value">{formatter(row[metric_col])}</span>
                <span class="interesting-context">{' · '.join(context)}</span>
            </div>
            """
        )
    return "".join(rows)


def generate_interesting_ids_html(df: pd.DataFrame, run_name: str) -> str:
    cards = [
        ("Highest prediction probability", "pred_probability", format_probability, False),
        ("Lowest prediction probability", "pred_probability", format_probability, True),
        ("Largest distance error", "distance_m", format_meters, False),
        ("Smallest distance error", "distance_m", format_meters, True),
        ("Largest yaw error", "yaw_error_abs", format_degrees, False),
        ("Smallest yaw error", "yaw_error_abs", format_degrees, True),
    ]

    html_cards = []
    for title, metric_col, formatter, ascending in cards:
        html_cards.append(
            f"""
            <div class="interesting-card">
                <h6>{escape(title)}</h6>
                {_metric_rows_html(df, run_name, metric_col, formatter, ascending)}
            </div>
            """
        )

    return f"""
    <div class="interesting-section">
        <div class="section-title"><i class="fas fa-star"></i> Interesting image IDs to inspect</div>
        <div class="interesting-grid">
            {''.join(html_cards)}
        </div>
    </div>
    """


# ============================================================================
# HTML GENERATION - MAIN PAGE
# ============================================================================


def generate_main_html(run_dir: Path, run_name: str, df: pd.DataFrame) -> str:
    """Generate the main table/overview HTML page."""
    df = enrich_results(df)

    table_columns = [
        "image_id",
        "image_path",
        "h5_id_dataset",
        "gt_yaw",
        "pred_yaw",
        "pred_probability",
        "distance_m",
        "yaw_error_abs",
    ]
    table_columns = [col for col in table_columns if col in df.columns]
    display_df = df[table_columns].copy()

    display_names = {
        "image_id": "Image ID",
        "image_path": "Image Path",
        "h5_id_dataset": "H5 Dataset ID",
        "gt_yaw": "GT Yaw",
        "pred_yaw": "Pred Yaw",
        "pred_probability": "Pred Probability",
        "distance_m": "Distance Error",
        "yaw_error_abs": "Yaw Error",
    }

    if "image_id" in display_df.columns:
        display_df["image_id"] = display_df["image_id"].apply(format_image_id)

    for col in ["gt_yaw", "pred_yaw", "yaw_error_abs"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_degrees)

    if "distance_m" in display_df.columns:
        display_df["distance_m"] = display_df["distance_m"].apply(format_meters)

    if "pred_probability" in display_df.columns:
        display_df["pred_probability"] = display_df["pred_probability"].apply(format_probability)

    display_df = display_df.rename(columns=display_names)
    table_html = dataframe_to_html_left(
        display_df,
        classes="table table-sm table-striped table-hover results-table",
    )

    rows = table_html.split("<tbody>")[1].split("</tbody>")[0]
    new_rows = ""
    for i, row in enumerate(rows.split("</tr>")):
        if row.strip():
            new_rows += f'<tr onclick="viewDetails({i})" class="detail-row">{row}</tr>'

    table_html = table_html.split("<tbody>")[0] + "<tbody>" + new_rows + "</tbody>" + table_html.split("</tbody>")[1]

    summary_html = generate_summary_html(df)
    interesting_html = generate_interesting_ids_html(df, run_name)
    images_json = json.dumps([format_image_id(v) for v in df["image_id"].tolist()])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrienterNet Results Dashboard - {escape(run_name)}</title>

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        body {{
            background-color: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}

        .navbar {{
            min-height: 42px;
            padding: 4px 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }}

        .navbar-brand {{
            font-size: 1rem;
            line-height: 1.1;
        }}

        .nav-actions a {{
            color: white;
            text-decoration: none;
            font-size: 0.86rem;
            margin-left: 12px;
            opacity: 0.92;
        }}

        .nav-actions a:hover {{ opacity: 1; text-decoration: underline; }}

        .page-container {{ padding: 14px; }}

        .header-section,
        .table-container,
        .interesting-section {{
            background: white;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08);
        }}

        h1 {{
            color: #667eea;
            margin-bottom: 6px;
            font-size: 1.55rem;
        }}

        .run-info {{ color: #666; font-size: 0.9rem; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }}

        .summary-card {{
            background: white;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08);
            border-left: 4px solid #667eea;
        }}

        .summary-label {{
            color: #6c757d;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 4px;
        }}

        .summary-value {{ font-size: 1.2rem; font-weight: 700; color: #222; }}
        .muted {{ font-size: 0.82rem; color: #6c757d; font-weight: 500; }}

        .section-title {{
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
        }}

        .interesting-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
        }}

        .interesting-card {{
            border: 1px solid #e7e7e7;
            border-radius: 8px;
            padding: 10px;
            background: #fafafa;
        }}

        .interesting-card h6 {{
            color: #333;
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 0.92rem;
        }}

        .interesting-row {{
            display: grid;
            grid-template-columns: 80px 92px 1fr;
            gap: 6px;
            padding: 5px 6px;
            border-radius: 5px;
            cursor: pointer;
            align-items: baseline;
            font-size: 0.84rem;
        }}

        .interesting-row:hover {{ background: #eef1ff; }}
        .interesting-id {{ font-weight: 700; color: #667eea; }}
        .interesting-value {{ font-variant-numeric: tabular-nums; color: #222; }}
        .interesting-context {{ color: #777; font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .empty-note {{ color: #777; font-size: 0.85rem; }}

        .detail-row {{ cursor: pointer; transition: background-color 0.2s ease; }}
        .detail-row:hover {{ background-color: #e9ecef !important; }}

        .table {{ margin-bottom: 0; }}
        .results-table th,
        .results-table td {{
            text-align: left !important;
            vertical-align: middle;
        }}

        .results-table thead {{ background-color: #667eea; color: white; }}
        .results-table thead th {{ border-color: #667eea; font-weight: 600; white-space: nowrap; }}

        .results-table th:first-child,
        .results-table td:first-child {{
            width: 76px;
            min-width: 76px;
            max-width: 76px;
            white-space: nowrap;
        }}

        .results-table td:nth-child(2) {{
            max-width: 520px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid d-flex justify-content-between align-items-center px-1">
            <span class="navbar-brand mb-0">
                <i class="fas fa-map-location-dot"></i> OrienterNet Results Dashboard
            </span>
            <span class="nav-actions">
                <a href="{run_name}_analysis.html"><i class="fas fa-chart-line"></i> Analysis</a>
            </span>
        </div>
    </nav>

    <div class="container-fluid page-container">
        <div class="header-section">
            <h1>Run: {escape(run_name)}</h1>
            <div class="run-info">
                <i class="fas fa-database"></i> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <i class="fas fa-folder"></i> Path: {escape(str(run_dir))}
            </div>
        </div>

        {summary_html}
        {interesting_html}

        <div class="table-container">
            <div class="section-title"><i class="fas fa-table"></i> Results Overview</div>
            <div style="overflow-x: auto;">
                {table_html}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const images = {images_json};

        function viewDetails(rowIndex) {{
            const imageId = images[rowIndex];
            window.location.href = `{run_name}_detail_${{imageId}}.html`;
        }}
    </script>
</body>
</html>"""
    return html_content


# ============================================================================
# ANALYSIS PAGE
# ============================================================================


def _skew(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        return np.nan
    std = values.std(ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean(((values - values.mean()) / std) ** 3))


def _kurtosis(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        return np.nan
    std = values.std(ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean(((values - values.mean()) / std) ** 4) - 3.0)


def _figure_to_data_uri(fig: Any) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def _safe_percentile(series: pd.Series, percentile: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, percentile))


def _analysis_valid_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = enrich_results(df)
    required = ["gt_latitude", "gt_longitude", "gt_yaw", "pred_latitude", "pred_longitude", "pred_yaw"]
    present = [col for col in required if col in df.columns]
    if len(present) != len(required):
        return df.iloc[0:0].copy(), len(df)

    valid_mask = df[required].notna().all(axis=1)
    valid = df.loc[valid_mask].copy()
    excluded_count = len(df) - len(valid)

    median_dist = valid["distance_m"].median()
    median_yaw = valid["yaw_error_abs"].median()
    dist_norm = median_dist if pd.notna(median_dist) and median_dist > 0 else 1.0
    yaw_norm = median_yaw if pd.notna(median_yaw) and median_yaw > 0 else 1.0
    valid["loss_combined"] = ((valid["distance_m"] / dist_norm) + (valid["yaw_error_abs"] / yaw_norm)) / 2

    max_dist = valid["distance_m"].quantile(0.95)
    max_yaw = valid["yaw_error_abs"].quantile(0.95)
    max_dist = max_dist if pd.notna(max_dist) and max_dist > 0 else max(valid["distance_m"].max(), 1.0)
    max_yaw = max_yaw if pd.notna(max_yaw) and max_yaw > 0 else max(valid["yaw_error_abs"].max(), 1.0)

    valid["quality_score"] = 100 * (
        0.5 * (1 - np.minimum(valid["distance_m"] / max_dist, 1))
        + 0.5 * (1 - np.minimum(valid["yaw_error_abs"] / max_yaw, 1))
    )

    dist_threshold = valid["distance_m"].median()
    yaw_threshold = valid["yaw_error_abs"].median()
    dist_threshold = dist_threshold if pd.notna(dist_threshold) and dist_threshold > 0 else 1.0
    yaw_threshold = yaw_threshold if pd.notna(yaw_threshold) and yaw_threshold > 0 else 1.0

    def categorize_error(row: pd.Series) -> str:
        dist = row["distance_m"]
        yaw = row["yaw_error_abs"]
        if dist < dist_threshold and yaw < yaw_threshold:
            return "Excellent"
        if dist < dist_threshold * 1.5 and yaw < yaw_threshold * 1.5:
            return "Good"
        if dist < dist_threshold * 2.5 or yaw < yaw_threshold * 2.5:
            return "Acceptable"
        return "Poor"

    valid["error_category"] = valid.apply(categorize_error, axis=1)
    return valid, excluded_count


def _metric_table_html(rows: List[Tuple[str, str]]) -> str:
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return dataframe_to_html_left(df, "table table-sm metric-table")


def _chart_section(title: str, img_uri: str, caption: str = "") -> str:
    caption_html = f'<p class="chart-caption">{escape(caption)}</p>' if caption else ""
    return f"""
    <div class="chart-card">
        <h5>{escape(title)}</h5>
        <img src="{img_uri}" alt="{escape(title)}">
        {caption_html}
    </div>
    """


def _try_compute_s2_summary(valid: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    cell_path = BASE_DATA_DIR / "cells" / "bern_14.gpkg"
    if not cell_path.exists():
        return None, f"S2 cell file not found at {cell_path}; S2 cell table skipped."

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as exc:
        return None, f"geopandas/shapely not available ({exc}); S2 cell table skipped."

    try:
        s2_cells = gpd.read_file(cell_path)
        if str(s2_cells.crs) != "EPSG:4326":
            s2_cells = s2_cells.to_crs("EPSG:4326")

        geometry = [Point(xy) for xy in zip(valid["pred_longitude"], valid["pred_latitude"])]
        gdf = gpd.GeoDataFrame(valid, geometry=geometry, crs="EPSG:4326")
        points_in_cells = gpd.sjoin(gdf, s2_cells, how="left", predicate="within")
        cell_stats = points_in_cells.groupby("index_right").agg(
            distance_mean=("distance_m", "mean"),
            distance_std=("distance_m", "std"),
            yaw_mean=("yaw_error_abs", "mean"),
            yaw_std=("yaw_error_abs", "std"),
            yaw_signed_mean=("yaw_error_signed", "mean"),
            count=("image_id", "count"),
        )
        cell_stats = cell_stats.reset_index().rename(columns={"index_right": "cell_idx"})
        cell_stats = cell_stats[cell_stats["count"] > 0].sort_values("distance_mean", ascending=False)
        return cell_stats, f"Loaded {len(s2_cells)} S2 cells from {cell_path}."
    except Exception as exc:
        return None, f"Could not compute S2 cell summary ({exc}); S2 cell table skipped."


def generate_analysis_html(run_dir: Path, run_name: str, df: pd.DataFrame) -> str:
    """Generate an analysis page based on the notebook, excluding folium maps."""
    valid, excluded_count = _analysis_valid_df(df)

    if valid.empty:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrienterNet Analysis - {escape(run_name)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="p-4">
    <a href="{run_name}_dashboard.html">← Back to dashboard</a>
    <h1>Analysis: {escape(run_name)}</h1>
    <div class="alert alert-warning">No valid predictions with ground-truth and predicted coordinates/yaw were found.</div>
</body>
</html>"""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrienterNet Analysis - {escape(run_name)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="p-4">
    <a href="{run_name}_dashboard.html">← Back to dashboard</a>
    <h1>Analysis: {escape(run_name)}</h1>
    <div class="alert alert-danger">matplotlib is required to create the analysis charts: {escape(str(exc))}</div>
</body>
</html>"""

    charts: List[str] = []

    # Distance histogram, linear and log scale.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.hist(valid["distance_m"].dropna(), bins=50, edgecolor="black", alpha=0.75)
    ax1.axvline(valid["distance_m"].mean(), linestyle="--", linewidth=2, label=f"Mean: {valid['distance_m'].mean():.1f} m")
    ax1.axvline(valid["distance_m"].median(), linestyle="--", linewidth=2, label=f"Median: {valid['distance_m'].median():.1f} m")
    ax1.set_xlabel("Distance Error (meters)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distance Error Distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.hist(valid["distance_m"].dropna(), bins=50, edgecolor="black", alpha=0.75)
    ax2.set_yscale("log")
    ax2.axvline(valid["distance_m"].mean(), linestyle="--", linewidth=2, label=f"Mean: {valid['distance_m'].mean():.1f} m")
    ax2.axvline(valid["distance_m"].median(), linestyle="--", linewidth=2, label=f"Median: {valid['distance_m'].median():.1f} m")
    ax2.set_xlabel("Distance Error (meters)")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title("Distance Error Distribution, Log Frequency")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    charts.append(_chart_section("Distance error histogram", _figure_to_data_uri(fig)))

    # Yaw histogram, signed and absolute.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.hist(valid["yaw_error_signed"].dropna(), bins=50, edgecolor="black", alpha=0.75)
    ax1.axvline(valid["yaw_error_signed"].mean(), linestyle="--", linewidth=2, label=f"Mean: {valid['yaw_error_signed'].mean():.1f}°")
    ax1.axvline(0, linestyle="-", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Signed Yaw Error (degrees)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Signed Yaw Error Distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.hist(valid["yaw_error_abs"].dropna(), bins=50, edgecolor="black", alpha=0.75)
    ax2.axvline(valid["yaw_error_abs"].mean(), linestyle="--", linewidth=2, label=f"Mean: {valid['yaw_error_abs'].mean():.1f}°")
    ax2.axvline(valid["yaw_error_abs"].median(), linestyle="--", linewidth=2, label=f"Median: {valid['yaw_error_abs'].median():.1f}°")
    ax2.set_xlabel("Absolute Yaw Error (degrees)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Absolute Yaw Error Distribution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    charts.append(_chart_section("Yaw error histogram", _figure_to_data_uri(fig)))

    # Percentiles.
    percentiles = [50, 75, 90, 95, 99]
    dist_perc = [_safe_percentile(valid["distance_m"], p) for p in percentiles]
    yaw_perc = [_safe_percentile(valid["yaw_error_abs"], p) for p in percentiles]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar([str(p) for p in percentiles], dist_perc)
    ax1.set_xlabel("Percentile")
    ax1.set_ylabel("Distance (meters)")
    ax1.set_title("Distance Error Percentiles")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar([str(p) for p in percentiles], yaw_perc)
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("Yaw Error (degrees)")
    ax2.set_title("Yaw Error Percentiles")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    charts.append(_chart_section("Percentile analysis", _figure_to_data_uri(fig)))

    # CDF.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sorted_dist = np.sort(valid["distance_m"].dropna().to_numpy(dtype=float))
    cdf_dist = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    ax1.plot(sorted_dist, cdf_dist, linewidth=2, label="Model CDF")
    if len(sorted_dist) > 0:
        ax1.plot([sorted_dist.min(), sorted_dist.max()], [0, 1], linestyle="--", linewidth=2, alpha=0.7, label="Linear reference")
    ax1.set_xlabel("Distance Error (meters)")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("Distance Error CDF")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.5, linestyle="--", alpha=0.5, label="50%")
    ax1.axhline(0.9, linestyle="--", alpha=0.5, label="90%")
    ax1.legend()

    sorted_yaw = np.sort(valid["yaw_error_abs"].dropna().to_numpy(dtype=float))
    cdf_yaw = np.arange(1, len(sorted_yaw) + 1) / len(sorted_yaw)
    ax2.plot(sorted_yaw, cdf_yaw, linewidth=2, label="Model CDF")
    if len(sorted_yaw) > 0:
        ax2.plot([sorted_yaw.min(), sorted_yaw.max()], [0, 1], linestyle="--", linewidth=2, alpha=0.7, label="Linear reference")
    ax2.set_xlabel("Yaw Error (degrees)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Yaw Error CDF")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.5, linestyle="--", alpha=0.5, label="50%")
    ax2.axhline(0.9, linestyle="--", alpha=0.5, label="90%")
    ax2.legend()
    fig.tight_layout()
    charts.append(_chart_section("Cumulative distribution functions", _figure_to_data_uri(fig)))

    # Correlation scatter.
    correlation = valid["distance_m"].corr(valid["yaw_error_abs"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid["distance_m"], valid["yaw_error_abs"], alpha=0.55, s=20)
    ax.set_xlabel("Distance Error (meters)")
    ax.set_ylabel("Yaw Error (degrees)")
    ax.set_title(f"Distance vs Yaw Error Correlation (r={correlation:.4f})")
    ax.grid(True, alpha=0.3)
    if len(valid) >= 2 and valid["distance_m"].nunique() > 1:
        z = np.polyfit(valid["distance_m"], valid["yaw_error_abs"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["distance_m"].min(), valid["distance_m"].max(), 100)
        ax.plot(x_line, p(x_line), linestyle="--", linewidth=2, label="Trend")
        ax.legend()
    fig.tight_layout()
    charts.append(_chart_section("Distance/yaw correlation", _figure_to_data_uri(fig)))

    # Box plots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.boxplot([valid["distance_m"].dropna()], vert=True, patch_artist=True)
    ax1.set_ylabel("Distance Error (meters)")
    ax1.set_title("Distance Error Box Plot")
    ax1.set_xticklabels(["Distance"])
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.boxplot([valid["yaw_error_abs"].dropna()], vert=True, patch_artist=True)
    ax2.set_ylabel("Yaw Error (degrees)")
    ax2.set_title("Yaw Error Box Plot")
    ax2.set_xticklabels(["Yaw"])
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    charts.append(_chart_section("Error distribution box plots", _figure_to_data_uri(fig)))

    # Regional bins heatmap.
    regional_note = ""
    try:
        valid["lat_bin"] = pd.cut(valid["gt_latitude"], bins=6)
        valid["lon_bin"] = pd.cut(valid["gt_longitude"], bins=6)
        pivot_dist = valid.pivot_table(values="distance_m", index="lat_bin", columns="lon_bin", aggfunc="mean")
        pivot_yaw = valid.pivot_table(values="yaw_error_abs", index="lat_bin", columns="lon_bin", aggfunc="mean")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        im1 = ax1.imshow(pivot_dist.to_numpy(dtype=float), aspect="auto")
        ax1.set_title("Mean Distance Error by Region")
        ax1.set_xlabel("Longitude Bin")
        ax1.set_ylabel("Latitude Bin")
        ax1.set_xticks(range(len(pivot_dist.columns)))
        ax1.set_xticklabels([str(c) for c in pivot_dist.columns], rotation=45, ha="right", fontsize=7)
        ax1.set_yticks(range(len(pivot_dist.index)))
        ax1.set_yticklabels([str(i) for i in pivot_dist.index], fontsize=7)
        fig.colorbar(im1, ax=ax1, label="Mean Distance (m)")
        for y in range(pivot_dist.shape[0]):
            for x in range(pivot_dist.shape[1]):
                value = pivot_dist.iloc[y, x]
                if pd.notna(value):
                    ax1.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=7)

        im2 = ax2.imshow(pivot_yaw.to_numpy(dtype=float), aspect="auto")
        ax2.set_title("Mean Yaw Error by Region")
        ax2.set_xlabel("Longitude Bin")
        ax2.set_ylabel("Latitude Bin")
        ax2.set_xticks(range(len(pivot_yaw.columns)))
        ax2.set_xticklabels([str(c) for c in pivot_yaw.columns], rotation=45, ha="right", fontsize=7)
        ax2.set_yticks(range(len(pivot_yaw.index)))
        ax2.set_yticklabels([str(i) for i in pivot_yaw.index], fontsize=7)
        fig.colorbar(im2, ax=ax2, label="Mean Yaw (°)")
        for y in range(pivot_yaw.shape[0]):
            for x in range(pivot_yaw.shape[1]):
                value = pivot_yaw.iloc[y, x]
                if pd.notna(value):
                    ax2.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=7)
        fig.tight_layout()
        charts.append(_chart_section("Regional analysis by latitude/longitude bins", _figure_to_data_uri(fig)))
    except Exception as exc:
        regional_note = f"Regional bin heatmap skipped: {exc}"

    # Category pie/bar.
    category_counts = valid["error_category"].value_counts().reindex(["Excellent", "Good", "Acceptable", "Poor"]).fillna(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    non_zero = category_counts[category_counts > 0]
    if not non_zero.empty:
        ax1.pie(non_zero, labels=non_zero.index, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Prediction Error Categories")
    ax2.bar(category_counts.index, category_counts.values)
    ax2.set_title("Count by Error Category")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=35)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    charts.append(_chart_section("Error categories", _figure_to_data_uri(fig)))

    # Quality score.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(valid["quality_score"].dropna(), bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(valid["quality_score"].mean(), linestyle="--", linewidth=2, label=f"Mean: {valid['quality_score'].mean():.1f}")
    ax.axvline(valid["quality_score"].median(), linestyle="--", linewidth=2, label=f"Median: {valid['quality_score'].median():.1f}")
    ax.set_xlabel("Quality Score (0-100)")
    ax.set_ylabel("Frequency")
    ax.set_title("Combined Quality Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts.append(_chart_section("Quality score analysis", _figure_to_data_uri(fig)))

    summary_rows = [
        ("Total rows loaded", f"{len(df)}"),
        ("Valid predictions analyzed", f"{len(valid)}"),
        ("Excluded rows", f"{excluded_count}"),
        ("Distance MAE", format_meters(valid["distance_m"].mean())),
        ("Distance RMSE", format_meters(np.sqrt((valid["distance_m"] ** 2).mean()))),
        ("Distance median", format_meters(valid["distance_m"].median())),
        ("Distance 90th percentile", format_meters(_safe_percentile(valid["distance_m"], 90))),
        ("Yaw MAE", format_degrees(valid["yaw_error_abs"].mean())),
        ("Yaw RMSE", format_degrees(np.sqrt((valid["yaw_error_abs"] ** 2).mean()))),
        ("Yaw median", format_degrees(valid["yaw_error_abs"].median())),
        ("Yaw 90th percentile", format_degrees(_safe_percentile(valid["yaw_error_abs"], 90))),
        ("Mean prediction probability", format_probability(valid["pred_probability"].mean())),
        ("Max prediction probability", format_probability(valid["pred_probability"].max())),
        ("Mean quality score", f"{valid['quality_score'].mean():.2f}/100"),
        ("Distance/yaw correlation", f"{correlation:.4f}"),
    ]

    statistical_rows = [
        ("Distance std dev", format_meters(valid["distance_m"].std())),
        ("Distance min", format_meters(valid["distance_m"].min())),
        ("Distance max", format_meters(valid["distance_m"].max())),
        ("Distance Q1", format_meters(valid["distance_m"].quantile(0.25))),
        ("Distance Q3", format_meters(valid["distance_m"].quantile(0.75))),
        ("Distance skewness", f"{_skew(valid['distance_m']):.4f}"),
        ("Distance kurtosis", f"{_kurtosis(valid['distance_m']):.4f}"),
        ("Yaw std dev", format_degrees(valid["yaw_error_abs"].std())),
        ("Yaw min", format_degrees(valid["yaw_error_abs"].min())),
        ("Yaw max", format_degrees(valid["yaw_error_abs"].max())),
        ("Yaw Q1", format_degrees(valid["yaw_error_abs"].quantile(0.25))),
        ("Yaw Q3", format_degrees(valid["yaw_error_abs"].quantile(0.75))),
        ("Yaw skewness", f"{_skew(valid['yaw_error_abs']):.4f}"),
        ("Yaw kurtosis", f"{_kurtosis(valid['yaw_error_abs']):.4f}"),
        ("Combined loss mean", f"{valid['loss_combined'].mean():.4f}"),
        ("Combined loss median", f"{valid['loss_combined'].median():.4f}"),
    ]

    category_rows = []
    for category in ["Excellent", "Good", "Acceptable", "Poor"]:
        count = int((valid["error_category"] == category).sum())
        pct = 100 * count / len(valid) if len(valid) else 0.0
        category_rows.append((category, f"{count} ({pct:.1f}%)"))

    s2_stats, s2_note = _try_compute_s2_summary(valid)
    s2_html = ""
    if s2_stats is not None and not s2_stats.empty:
        s2_display = s2_stats.head(20).copy()
        for col in ["distance_mean", "distance_std"]:
            if col in s2_display.columns:
                s2_display[col] = s2_display[col].apply(format_meters)
        for col in ["yaw_mean", "yaw_std", "yaw_signed_mean"]:
            if col in s2_display.columns:
                s2_display[col] = s2_display[col].apply(format_degrees)
        s2_html = f"""
        <div class="analysis-card">
            <h4>S2 cell statistics</h4>
            <p class="note">{escape(s2_note)} Showing up to 20 cells sorted by worst mean distance error.</p>
            {dataframe_to_html_left(s2_display, 'table table-sm metric-table')}
        </div>
        """
    else:
        s2_html = f"""
        <div class="analysis-card">
            <h4>S2 cell statistics</h4>
            <p class="note">{escape(s2_note)}</p>
        </div>
        """

    recommendations = []
    poor_pct = 100 * (valid["error_category"] == "Poor").sum() / len(valid)
    if poor_pct > 10:
        recommendations.append("High proportion of poor predictions. Consider checking the failed cases for recurring scene or tile issues.")
    if _safe_percentile(valid["distance_m"], 90) > 1000:
        recommendations.append("The 90th percentile distance error is above 1000 m, so there are large localization outliers.")
    if abs(valid["yaw_error_signed"].mean()) > 5:
        recommendations.append("Orientation bias detected: the signed yaw error mean is more than 5° away from zero.")
    if pd.notna(correlation) and correlation > 0.3:
        recommendations.append(f"Distance and yaw errors have a moderate positive correlation (r={correlation:.3f}).")
    if not recommendations:
        recommendations.append("No strong warning thresholds were triggered by the notebook-style checks.")

    recommendation_html = "".join(f"<li>{escape(item)}</li>" for item in recommendations)
    regional_note_html = f'<div class="alert alert-warning">{escape(regional_note)}</div>' if regional_note else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrienterNet Analysis - {escape(run_name)}</title>

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        body {{
            background: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}
        .navbar {{
            min-height: 42px;
            padding: 4px 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }}
        .navbar a, .navbar-brand {{ color: white; text-decoration: none; font-size: 1rem; }}
        .page {{ padding: 14px; }}
        .hero, .analysis-card, .chart-card {{
            background: white;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08);
        }}
        h1 {{ color: #667eea; font-size: 1.55rem; margin-bottom: 6px; }}
        h4 {{ color: #667eea; font-size: 1.05rem; margin-bottom: 10px; }}
        h5 {{ color: #333; font-size: 1rem; margin-bottom: 10px; }}
        .summary-layout {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }}
        .chart-card img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }}
        .chart-caption, .note {{ color: #666; font-size: 0.9rem; margin-bottom: 8px; }}
        .metric-table th, .metric-table td {{ text-align: left !important; vertical-align: middle; }}
        .metric-table thead {{ background: #667eea; color: white; }}
        ul {{ margin-bottom: 0; }}
        @media (max-width: 700px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid d-flex justify-content-between align-items-center px-1">
            <a href="{run_name}_dashboard.html"><i class="fas fa-arrow-left"></i> Back to dashboard</a>
            <span class="navbar-brand"><i class="fas fa-chart-line"></i> Notebook Analysis</span>
        </div>
    </nav>

    <div class="container-fluid page">
        <div class="hero">
            <h1>Analysis: {escape(run_name)}</h1>
            <div class="note">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Based on the provided notebook computations, excluding the folium map pages.
            </div>
        </div>

        <div class="summary-layout">
            <div class="analysis-card">
                <h4>Key metrics</h4>
                {_metric_table_html(summary_rows)}
            </div>
            <div class="analysis-card">
                <h4>Statistical summary</h4>
                {_metric_table_html(statistical_rows)}
            </div>
            <div class="analysis-card">
                <h4>Error categories</h4>
                {_metric_table_html(category_rows)}
            </div>
            <div class="analysis-card">
                <h4>Recommendations</h4>
                <ul>{recommendation_html}</ul>
            </div>
        </div>

        {regional_note_html}
        {s2_html}

        <div class="charts-grid">
            {''.join(charts)}
        </div>
    </div>
</body>
</html>"""
    return html


# ============================================================================
# HTML GENERATION - DETAIL PAGE TEMPLATE
# ============================================================================


def generate_detail_html_template(run_name: str) -> str:
    """Generate a detail page template that will be filled with image data."""
    html_template = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Detail</title>

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>

    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background-color: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            height: 100vh;
            overflow: hidden;
        }

        .navbar {
            min-height: 42px;
            padding: 4px 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .navbar .container-fluid { padding: 0 4px; }
        .navbar-brand { font-size: 0.95rem; line-height: 1.1; }

        .back-btn,
        .analysis-btn {
            display: inline-block;
            padding: 4px 8px;
            background: rgba(255,255,255,0.14);
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.78rem;
            transition: background 0.2s;
            margin-right: 6px;
        }

        .back-btn:hover,
        .analysis-btn:hover {
            background: rgba(255,255,255,0.25);
            text-decoration: none;
            color: white;
        }

        .container-detail {
            display: grid;
            grid-template-columns: minmax(0, 1.12fr) minmax(0, 0.88fr);
            grid-template-rows: minmax(260px, 44%) minmax(240px, 56%);
            height: calc(100vh - 42px);
            gap: 0;
        }

        .left-panel {
            grid-column: 1;
            grid-row: 1 / 3;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #ddd;
            background: white;
            overflow: hidden;
        }

        .right-top {
            grid-column: 2;
            grid-row: 1;
            display: flex;
            flex-direction: column;
            border-bottom: 1px solid #ddd;
            background: white;
            min-height: 0;
        }

        .right-bottom {
            grid-column: 2;
            grid-row: 2;
            overflow-y: auto;
            background: white;
            min-height: 0;
        }

        #map {
            width: 100%;
            flex: 1;
            min-height: 0;
        }

        .map-header {
            padding: 7px 10px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            font-weight: 700;
            font-size: 0.88rem;
            color: #333;
        }

        .map-controls {
            padding: 6px 8px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            font-size: 0.78rem;
        }

        .map-control-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            align-items: start;
        }

        .control-group-title {
            display: block;
            color: #667eea;
            font-weight: 800;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 3px;
        }

        .map-controls label {
            display: block;
            margin-bottom: 2px;
            cursor: pointer;
            font-weight: 500;
            white-space: nowrap;
        }

        .map-controls input { margin-right: 4px; }

        .image-id-header {
            padding: 8px 10px;
            background: #667eea;
            color: white;
            font-weight: 700;
            border-bottom: 1px solid #ddd;
            font-size: 0.88rem;
        }

        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            background: #f0f0f0;
            min-height: 0;
        }

        .image-container img {
            max-width: 96%;
            max-height: 96%;
            object-fit: contain;
            border-radius: 4px;
        }

        .expand-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            z-index: 100;
            background: rgba(0,0,0,0.6);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 8px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 0.78rem;
        }

        .expand-btn:hover { background: rgba(0,0,0,0.8); }

        .stats-container {
            padding: 12px;
            overflow-y: auto;
        }

        .stats-section {
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #eee;
        }

        .stats-section:last-child { border-bottom: none; }

        .stats-section h6 {
            color: #667eea;
            margin-bottom: 8px;
            font-weight: 700;
        }

        dl { margin-bottom: 0; font-size: 0.86rem; }
        dt { font-weight: 700; color: #333; }
        dd { color: #555; margin-left: 0; }

        .external-link {
            display: inline-block;
            margin-top: 3px;
            color: #0d6efd;
            text-decoration: none;
            font-size: 0.82rem;
        }
        .external-link:hover { text-decoration: underline; }

        .yaw-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 999px;
            padding: 2px 8px;
            color: white;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .yaw-pill.pred { background: #4bbff5; color: #05384f; }
        .yaw-pill.gt { background: #003b8e; }

        .modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .modal.active { display: flex; }

        .modal-content {
            max-width: 88vw;
            max-height: 88vh;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: transparent;
        }

        .modal-content img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .close-btn {
            position: absolute;
            top: -40px;
            right: 0;
            color: white;
            font-size: 34px;
            font-weight: bold;
            cursor: pointer;
            background: none;
            border: none;
            padding: 0;
        }

        .close-btn:hover { color: #ccc; }

        .leaflet-yaw-label {
            background: transparent;
            border: 0;
        }

        .yaw-label-inner {
            display: inline-block;
            padding: 2px 5px;
            border-radius: 4px;
            background: rgba(255,255,255,0.9);
            font-size: 11px;
            font-weight: 800;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25);
            white-space: nowrap;
        }

        .map-legend {
            background: rgba(255,255,255,0.94);
            border-radius: 6px;
            padding: 6px 8px;
            font-size: 0.75rem;
            box-shadow: 0 1px 5px rgba(0,0,0,0.25);
            line-height: 1.3;
        }
        .legend-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid d-flex justify-content-between align-items-center">
            <div>
                <a href="__RUN_NAME___dashboard.html" class="back-btn"><i class="fas fa-arrow-left"></i> Back</a>
                <a href="__RUN_NAME___analysis.html" class="analysis-btn"><i class="fas fa-chart-line"></i> Analysis</a>
            </div>
            <span class="navbar-brand mb-0">
                <i class="fas fa-image"></i> <span id="imageTitle">Image Details</span>
            </span>
        </div>
    </nav>

    <div class="container-detail">
        <div class="left-panel">
            <div class="map-header">
                <i class="fas fa-map-location-dot"></i> Geolocation Context
            </div>
            <div class="map-controls">
                <div class="map-control-grid">
                    <div class="control-group">
                        <span class="control-group-title">Basemap</span>
                        <label><input type="radio" name="basemap" value="osm" checked> OpenStreetMap</label>
                        <label><input type="radio" name="basemap" value="model" id="modelTileRadio"> Model OSM raster JPG</label>
                        <label><input type="radio" name="basemap" value="satellite"> Esri Satellite</label>
                    </div>
                    <div class="control-group">
                        <span class="control-group-title">Layers</span>
                        <label><input type="checkbox" id="boundaryLayer" checked> <span id="boundaryLabel">Request tile</span></label>
                        <label><input type="checkbox" id="predictionLayer"> Prediction map</label>
                        <label><input type="checkbox" id="neuralLayer"> Neural activation</label>
                    </div>
                </div>
            </div>
            <div id="map"></div>
        </div>

        <div class="right-top">
            <div class="image-id-header">
                <i class="fas fa-photo-film"></i> Rectified Image
            </div>
            <div class="image-container">
                <button class="expand-btn" onclick="expandImage()"><i class="fas fa-expand"></i> Expand</button>
                <img id="rectifiedImage" src="" alt="Rectified Image">
            </div>
        </div>

        <div class="right-bottom">
            <div class="stats-container" id="statsContainer"></div>
        </div>
    </div>

    <div id="imageModal" class="modal">
        <div class="modal-content">
            <button class="close-btn" onclick="closeImageModal()">&times;</button>
            <img id="expandedImage" src="" alt="Expanded Image">
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const imageData = DATA_PLACEHOLDER;

        // Minimal rotated image overlay. Leaflet's normal imageOverlay can only handle
        // axis-aligned bounds; this version maps the image onto top-left/top-right/bottom-left
        // corners so saved model rasters, prediction maps, and neural maps share one geometry.
        L.ImageOverlay.Rotated = L.Layer.extend({
            options: { opacity: 1, interactive: false, className: '' },
            initialize: function(url, topLeft, topRight, bottomLeft, options) {
                this._url = url;
                this._topLeft = L.latLng(topLeft);
                this._topRight = L.latLng(topRight);
                this._bottomLeft = L.latLng(bottomLeft);
                L.setOptions(this, options);
            },
            onAdd: function(map) {
                this._map = map;
                if (!this._image) this._initImage();
                map.getPanes().overlayPane.appendChild(this._image);
                map.on('zoomend viewreset moveend resize', this._reset, this);
                this._reset();
            },
            onRemove: function(map) {
                if (this._image && this._image.parentNode) this._image.parentNode.removeChild(this._image);
                map.off('zoomend viewreset moveend resize', this._reset, this);
            },
            _initImage: function() {
                const img = this._image = L.DomUtil.create('img', 'leaflet-image-layer ' + (this.options.className || ''));
                img.src = this._url;
                img.style.position = 'absolute';
                img.style.transformOrigin = '0 0';
                img.style.willChange = 'transform';
                img.style.pointerEvents = this.options.interactive ? 'auto' : 'none';
                L.DomUtil.setOpacity(img, this.options.opacity);
                img.onload = () => this._reset();
            },
            _reset: function() {
                if (!this._map || !this._image) return;
                const w = this._image.naturalWidth || this._image.width;
                const h = this._image.naturalHeight || this._image.height;
                if (!w || !h) return;
                const p0 = this._map.latLngToLayerPoint(this._topLeft);
                const p1 = this._map.latLngToLayerPoint(this._topRight);
                const p2 = this._map.latLngToLayerPoint(this._bottomLeft);
                const vx = p1.subtract(p0);
                const vy = p2.subtract(p0);
                const a = vx.x / w;
                const b = vx.y / w;
                const c = vy.x / h;
                const d = vy.y / h;
                this._image.style.width = w + 'px';
                this._image.style.height = h + 'px';
                const transform = `translate3d(${p0.x}px, ${p0.y}px, 0) matrix(${a}, ${b}, ${c}, ${d}, 0, 0)`;
                this._image.style[L.DomUtil.TRANSFORM || 'transform'] = transform;
            },
            setOpacity: function(opacity) {
                this.options.opacity = opacity;
                if (this._image) L.DomUtil.setOpacity(this._image, opacity);
                return this;
            }
        });
        L.imageOverlay.rotated = function(url, topLeft, topRight, bottomLeft, options) {
            return new L.ImageOverlay.Rotated(url, topLeft, topRight, bottomLeft, options);
        };

        function pairToLatLng(pair) {
            return L.latLng(pair[0], pair[1]);
        }

        function getCornerLatLngs() {
            if (!imageData.tile_corners) return null;
            const c = imageData.tile_corners;
            if (!(c.top_left && c.top_right && c.bottom_left && c.bottom_right)) return null;
            return {
                topLeft: pairToLatLng(c.top_left),
                topRight: pairToLatLng(c.top_right),
                bottomLeft: pairToLatLng(c.bottom_left),
                bottomRight: pairToLatLng(c.bottom_right),
            };
        }

        function getTileBounds() {
            const corners = getCornerLatLngs();
            if (corners) {
                return L.latLngBounds([corners.topLeft, corners.topRight, corners.bottomLeft, corners.bottomRight]);
            }
            if (imageData.tile_bounds && imageData.tile_bounds.length === 2) {
                return L.latLngBounds(imageData.tile_bounds);
            }
            return L.latLngBounds([
                [imageData.gt_lat - 0.0005, imageData.gt_lon - 0.0005],
                [imageData.gt_lat + 0.0005, imageData.gt_lon + 0.0005]
            ]);
        }

        function createGeorefImageOverlay(imageUrl, options) {
            if (!imageUrl) return null;
            const corners = getCornerLatLngs();
            if (corners) {
                return L.imageOverlay.rotated(imageUrl, corners.topLeft, corners.topRight, corners.bottomLeft, options || {});
            }
            return L.imageOverlay(imageUrl, getTileBounds(), options || {});
        }

        function createBoundaryLayer() {
            const corners = getCornerLatLngs();
            const style = {
                color: '#222222',
                weight: 2,
                opacity: 0.9,
                fill: false,
                dashArray: '6, 5'
            };
            if (corners) {
                return L.polygon([corners.topLeft, corners.topRight, corners.bottomRight, corners.bottomLeft], style);
            }
            return L.rectangle(getTileBounds(), style);
        }

        function destinationPoint(lat, lon, bearingDeg, distanceMeters) {
            const R = 6378137;
            const bearing = bearingDeg * Math.PI / 180;
            const lat1 = lat * Math.PI / 180;
            const lon1 = lon * Math.PI / 180;
            const d = distanceMeters / R;
            const lat2 = Math.asin(Math.sin(lat1) * Math.cos(d) + Math.cos(lat1) * Math.sin(d) * Math.cos(bearing));
            const lon2 = lon1 + Math.atan2(
                Math.sin(bearing) * Math.sin(d) * Math.cos(lat1),
                Math.cos(d) - Math.sin(lat1) * Math.sin(lat2)
            );
            return [lat2 * 180 / Math.PI, lon2 * 180 / Math.PI];
        }

        function addYawChevron(targetGroup, lat, lon, yawDeg, color, labelText) {
            const armLength = Math.max(18, (imageData.tile_size_meters || 136) * 0.17);
            const halfAngle = 22.5;
            const center = [lat, lon];
            const left = destinationPoint(lat, lon, yawDeg - halfAngle, armLength);
            const right = destinationPoint(lat, lon, yawDeg + halfAngle, armLength);
            const labelPos = destinationPoint(lat, lon, yawDeg, armLength * 0.72);
            L.polyline([center, left], { color, weight: 4, opacity: 0.95, lineCap: 'round' }).addTo(targetGroup);
            L.polyline([center, right], { color, weight: 4, opacity: 0.95, lineCap: 'round' }).addTo(targetGroup);
            L.marker(labelPos, {
                interactive: false,
                icon: L.divIcon({
                    className: 'leaflet-yaw-label',
                    html: `<span class="yaw-label-inner" style="color:${color}">${labelText} ${yawDeg.toFixed(1)}°</span>`,
                    iconSize: [70, 18],
                    iconAnchor: [35, 9]
                })
            }).addTo(targetGroup);
        }

        function googleMapsUrl(lat, lon) {
            return `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
        }

        const modelBounds = getTileBounds();
        const map = L.map('map', { preferCanvas: true });
        map.fitBounds(modelBounds.pad(0.28));

        const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 20
        });
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri',
            maxZoom: 20
        });
        const modelTileLayer = createGeorefImageOverlay(imageData.osm_tile_img, { opacity: 1.0, interactive: false, className: 'model-tile-layer' });

        let activeBasemap = null;
        function setBasemap(name) {
            if (activeBasemap && map.hasLayer(activeBasemap)) map.removeLayer(activeBasemap);
            if (name === 'satellite') {
                activeBasemap = satelliteLayer;
            } else if (name === 'model' && modelTileLayer) {
                activeBasemap = modelTileLayer;
            } else {
                activeBasemap = osmLayer;
            }
            activeBasemap.addTo(map);
            boundaryLayer.bringToFront();
            markerGroup.bringToFront();
        }

        if (!modelTileLayer) {
            const modelRadio = document.getElementById('modelTileRadio');
            modelRadio.disabled = true;
            modelRadio.parentElement.title = 'No saved osm_tile.jpg artifact found for this image.';
        }

        const boundaryLayer = createBoundaryLayer().addTo(map);
        const predictionOverlay = L.layerGroup();
        const neuralOverlay = L.layerGroup();
        const predictionImg = createGeorefImageOverlay(imageData.prediction_map_img, { opacity: 0.65, interactive: false });
        if (predictionImg) predictionOverlay.addLayer(predictionImg);
        const neuralImg = createGeorefImageOverlay(imageData.neural_map_img, { opacity: 0.65, interactive: false });
        if (neuralImg) neuralOverlay.addLayer(neuralImg);

        const markerGroup = L.layerGroup().addTo(map);
        const darkBlue = '#003b8e';
        const lightBlue = '#61c9ff';

        const gtMarker = L.circleMarker([imageData.gt_lat, imageData.gt_lon], {
            radius: 8,
            fillColor: darkBlue,
            color: '#001b42',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9,
            title: 'Ground Truth'
        }).bindPopup(`<b>Ground Truth Position</b><br><a href="${googleMapsUrl(imageData.gt_lat, imageData.gt_lon)}" target="_blank" rel="noopener">Open in Google Maps</a>`);
        markerGroup.addLayer(gtMarker);

        const predMarker = L.circleMarker([imageData.pred_lat, imageData.pred_lon], {
            radius: 8,
            fillColor: lightBlue,
            color: '#066184',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9,
            title: 'Prediction'
        }).bindPopup(`<b>Predicted Position</b><br><a href="${googleMapsUrl(imageData.pred_lat, imageData.pred_lon)}" target="_blank" rel="noopener">Open in Google Maps</a>`);
        markerGroup.addLayer(predMarker);

        addYawChevron(markerGroup, imageData.gt_lat, imageData.gt_lon, imageData.gt_yaw, darkBlue, 'GT');
        addYawChevron(markerGroup, imageData.pred_lat, imageData.pred_lon, imageData.pred_yaw, lightBlue, 'Pred');

        setBasemap('osm');

        const legend = L.control({ position: 'bottomleft' });
        legend.onAdd = function() {
            const div = L.DomUtil.create('div', 'map-legend');
            div.innerHTML = `
                <div><span class="legend-dot" style="background:${lightBlue}"></span>Prediction</div>
                <div><span class="legend-dot" style="background:${darkBlue}"></span>Ground truth</div>
                <div>Yaw shown as 45° chevrons.</div>
            `;
            return div;
        };
        legend.addTo(map);

        document.getElementById('boundaryLabel').textContent = `Request tile (${imageData.tile_size_meters || 136}m × ${imageData.tile_size_meters || 136}m)`;

        document.querySelectorAll('input[name="basemap"]').forEach(radio => {
            radio.addEventListener('change', (e) => setBasemap(e.target.value));
        });

        document.getElementById('boundaryLayer').addEventListener('change', (e) => {
            if (e.target.checked) map.addLayer(boundaryLayer);
            else map.removeLayer(boundaryLayer);
        });

        document.getElementById('predictionLayer').addEventListener('change', (e) => {
            if (e.target.checked && predictionOverlay.getLayers().length > 0) map.addLayer(predictionOverlay);
            else map.removeLayer(predictionOverlay);
            markerGroup.bringToFront();
        });

        document.getElementById('neuralLayer').addEventListener('change', (e) => {
            if (e.target.checked && neuralOverlay.getLayers().length > 0) map.addLayer(neuralOverlay);
            else map.removeLayer(neuralOverlay);
            markerGroup.bringToFront();
        });

        if (predictionOverlay.getLayers().length === 0) {
            document.getElementById('predictionLayer').disabled = true;
            document.getElementById('predictionLayer').parentElement.title = 'No prediction_map.jpg artifact found.';
        }
        if (neuralOverlay.getLayers().length === 0) {
            document.getElementById('neuralLayer').disabled = true;
            document.getElementById('neuralLayer').parentElement.title = 'No neural_map_rgb.jpg artifact found.';
        }

        document.getElementById('rectifiedImage').src = imageData.rectified_img;
        document.getElementById('expandedImage').src = imageData.rectified_img;
        document.getElementById('imageTitle').textContent = 'Image ' + imageData.image_id + ' Details';
        document.getElementById('statsContainer').innerHTML = imageData.stats_html;

        function expandImage() {
            document.getElementById('imageModal').classList.add('active');
        }

        function closeImageModal() {
            document.getElementById('imageModal').classList.remove('active');
        }

        document.getElementById('imageModal').addEventListener('click', (e) => {
            if (e.target.id === 'imageModal') closeImageModal();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeImageModal();
        });
    </script>
</body>
</html>'''
    return html_template.replace("__RUN_NAME__", run_name)


# ============================================================================
# GENERATOR SCRIPT FOR ON-DEMAND DETAIL PAGES
# ============================================================================


def create_detail_generator_script(run_dir: Path, run_name: str, df: pd.DataFrame) -> str:
    """Create a script to generate detail pages on-demand."""
    script_template = r'''#!/usr/bin/env python3
"""
On-demand Detail Page Generator.
Usage: python <run_name>_generate_detail.py <image_id>
"""

import base64
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

RUN_DIR_NAME = __RUN_DIR_NAME_JSON__
RUN_NAME = __RUN_NAME_JSON__
BASE_DIR = Path(__file__).parent
RUN_DIR = BASE_DIR / RUN_DIR_NAME
RESULTS_CSV = RUN_DIR / "results.csv"
TEMPLATE_FILE = BASE_DIR / f"{RUN_NAME}_detail_template.html"


def image_to_base64(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.suffix.lower()[1:]
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return ""


def load_artifact_json(artifact_dir: Path, filename: str) -> dict:
    json_path = artifact_dir / filename
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_float(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371000


def normalize_angle(angle):
    if pd.isna(angle):
        return np.nan
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def parse_tile_size_meters(default: int = 136) -> int:
    match = re.search(r"(?:^|_)process_(\d+)t_", RUN_NAME)
    if not match:
        match = re.search(r"(\d+)t_", RUN_NAME)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return default


def google_maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def fallback_tile_geometry(center_lat: float, center_lon: float, tile_size_meters: float) -> Dict[str, Any]:
    half = float(tile_size_meters) / 2.0
    earth_radius = 6378137.0
    lat_delta = (half / earth_radius) * (180.0 / math.pi)
    lon_delta = (half / (earth_radius * max(math.cos(math.radians(center_lat)), 1e-9))) * (180.0 / math.pi)

    south = center_lat - lat_delta
    north = center_lat + lat_delta
    west = center_lon - lon_delta
    east = center_lon + lon_delta
    corners = {
        "top_left": [north, west],
        "top_right": [north, east],
        "bottom_left": [south, west],
        "bottom_right": [south, east],
    }
    return {
        "corners": corners,
        "bounds": [[south, west], [north, east]],
        "source": "fallback_centered_on_ground_truth",
    }


def _valid_pair(pair: Any) -> Optional[list]:
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None
    try:
        lat = float(pair[0])
        lon = float(pair[1])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        return [lat, lon]
    except Exception:
        return None


def _normalize_corners(value: Any) -> Optional[Dict[str, list]]:
    if not isinstance(value, dict):
        return None
    aliases = {
        "top_left": ["top_left", "north_west", "nw", "upper_left"],
        "top_right": ["top_right", "north_east", "ne", "upper_right"],
        "bottom_left": ["bottom_left", "south_west", "sw", "lower_left"],
        "bottom_right": ["bottom_right", "south_east", "se", "lower_right"],
    }
    out = {}
    for canonical, keys in aliases.items():
        found = None
        for key in keys:
            if key in value:
                found = _valid_pair(value[key])
                break
        if found is None:
            return None
        out[canonical] = found
    return out


def _bounds_from_corners(corners: Dict[str, list]) -> list:
    lats = [pair[0] for pair in corners.values()]
    lons = [pair[1] for pair in corners.values()]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def _geometry_from_artifacts(artifacts_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    corner_keys = [
        "tile_corners_latlon",
        "canvas_corners_latlon",
        "canvas_bbox_latlon_corners",
        "tile_latlon_corners",
    ]
    for key in corner_keys:
        corners = _normalize_corners(artifacts_data.get(key))
        if corners:
            return {"corners": corners, "bounds": _bounds_from_corners(corners), "source": key}

    bounds = artifacts_data.get("tile_bounds_latlon") or artifacts_data.get("canvas_bounds_latlon")
    if isinstance(bounds, dict):
        try:
            south = float(bounds.get("south"))
            west = float(bounds.get("west"))
            north = float(bounds.get("north"))
            east = float(bounds.get("east"))
            corners = {
                "top_left": [north, west],
                "top_right": [north, east],
                "bottom_left": [south, west],
                "bottom_right": [south, east],
            }
            return {"corners": corners, "bounds": [[south, west], [north, east]], "source": "bounds_artifact"}
        except Exception:
            pass
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        sw = _valid_pair(bounds[0])
        ne = _valid_pair(bounds[1])
        if sw and ne:
            south, west = sw
            north, east = ne
            corners = {
                "top_left": [north, west],
                "top_right": [north, east],
                "bottom_left": [south, west],
                "bottom_right": [south, east],
            }
            return {"corners": corners, "bounds": [[south, west], [north, east]], "source": "bounds_artifact"}

    return None


def build_tile_geometry(artifacts_data: Dict[str, Any], gt_lat: float, gt_lon: float) -> Dict[str, Any]:
    tile_size_meters = int(artifacts_data.get("tile_size_meters") or parse_tile_size_meters())
    artifact_geometry = _geometry_from_artifacts(artifacts_data)
    if artifact_geometry:
        artifact_geometry["tile_size_meters"] = tile_size_meters
        return artifact_geometry
    fallback = fallback_tile_geometry(gt_lat, gt_lon, tile_size_meters)
    fallback["tile_size_meters"] = tile_size_meters
    return fallback


def yaw_pill(yaw: float, css_class: str) -> str:
    return f'<span class="yaw-pill {css_class}">{yaw:.2f}°</span>'


def generate_detail_page(image_id: int):
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    df["image_id"] = pd.to_numeric(df["image_id"], errors="coerce")

    row_data = df[df["image_id"] == image_id]
    if row_data.empty:
        return f"<h1>Error: Image {image_id} not found</h1>"

    row_data = row_data.iloc[0]
    artifact_dir = RUN_DIR / "artifacts" / f"image_{image_id}"
    if not artifact_dir.exists():
        return f"<h1>Error: Artifact directory not found: {artifact_dir}</h1>"

    artifacts_data = load_artifact_json(artifact_dir, "artifacts.json")
    camera_data = load_artifact_json(artifact_dir, "camera.json")

    rectified_img = image_to_base64(artifact_dir / "rectified_image.jpg")
    prediction_map_img = image_to_base64(artifact_dir / "prediction_map.jpg")
    neural_map_img = image_to_base64(artifact_dir / "neural_map_rgb.jpg")
    osm_tile_img = image_to_base64(artifact_dir / "osm_tile.jpg")

    gt_lat = safe_float(row_data.get("gt_latitude"))
    gt_lon = safe_float(row_data.get("gt_longitude"))
    pred_lat = safe_float(row_data.get("pred_latitude"))
    pred_lon = safe_float(row_data.get("pred_longitude"))
    gt_yaw = safe_float(row_data.get("gt_yaw"))
    pred_yaw = safe_float(row_data.get("pred_yaw"))
    pred_prob = safe_float(row_data.get("pred_probability"), default=float("nan"))

    distance_error = haversine(pred_lon, pred_lat, gt_lon, gt_lat)
    yaw_error_signed = normalize_angle(pred_yaw - gt_yaw)
    yaw_error_abs = abs(yaw_error_signed)
    tile_geometry = build_tile_geometry(artifacts_data, gt_lat, gt_lon)
    tile_size_meters = tile_geometry.get("tile_size_meters", parse_tile_size_meters())

    width = camera_data.get("width", "N/A")
    height = camera_data.get("height", "N/A")
    fx = fmt_float(camera_data.get("fx"), 2)
    fy = fmt_float(camera_data.get("fy"), 2)
    cx = fmt_float(camera_data.get("cx"), 2)
    cy = fmt_float(camera_data.get("cy"), 2)
    prob_text = f"{pred_prob * 100:.2f}%" if math.isfinite(pred_prob) else "N/A"

    geometry_note = "exact artifact corners" if tile_geometry.get("source") != "fallback_centered_on_ground_truth" else "fallback square centered on GT"

    stats_html = f"""
    <div class="stats-section">
        <h6>Camera Information</h6>
        <dl class="row">
            <dt class="col-6">Resolution:</dt>
            <dd class="col-6">{width}x{height}</dd>
            <dt class="col-6">Focal Length:</dt>
            <dd class="col-6">fx {fx}, fy {fy}</dd>
            <dt class="col-6">Principal Point:</dt>
            <dd class="col-6">({cx}, {cy})</dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6>Prediction Results</h6>
        <dl class="row">
            <dt class="col-6">Distance Error:</dt>
            <dd class="col-6"><strong>{distance_error:.2f} m</strong></dd>
            <dt class="col-6">Yaw Error:</dt>
            <dd class="col-6"><strong>{yaw_error_signed:.2f}°</strong> ({yaw_error_abs:.2f}° abs)</dd>
            <dt class="col-6">Confidence:</dt>
            <dd class="col-6">{prob_text}</dd>
            <dt class="col-6">Model Tile:</dt>
            <dd class="col-6">{tile_size_meters}m × {tile_size_meters}m<br><small>{geometry_note}</small></dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6>Ground Truth</h6>
        <dl class="row">
            <dt class="col-6">Position:</dt>
            <dd class="col-6">{gt_lat:.6f}<br>{gt_lon:.6f}<br><a class="external-link" href="{google_maps_url(gt_lat, gt_lon)}" target="_blank" rel="noopener"><i class="fas fa-up-right-from-square"></i> Google Maps GT</a></dd>
            <dt class="col-6">Yaw:</dt>
            <dd class="col-6">{yaw_pill(gt_yaw, 'gt')}</dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6>Prediction</h6>
        <dl class="row">
            <dt class="col-6">Position:</dt>
            <dd class="col-6">{pred_lat:.6f}<br>{pred_lon:.6f}<br><a class="external-link" href="{google_maps_url(pred_lat, pred_lon)}" target="_blank" rel="noopener"><i class="fas fa-up-right-from-square"></i> Google Maps prediction</a></dd>
            <dt class="col-6">Yaw:</dt>
            <dd class="col-6">{yaw_pill(pred_yaw, 'pred')}</dd>
        </dl>
    </div>
    """

    image_data = {
        "image_id": image_id,
        "gt_lat": gt_lat,
        "gt_lon": gt_lon,
        "gt_yaw": gt_yaw,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "pred_yaw": pred_yaw,
        "tile_size_meters": tile_size_meters,
        "tile_corners": tile_geometry.get("corners"),
        "tile_bounds": tile_geometry.get("bounds"),
        "tile_geometry_source": tile_geometry.get("source"),
        "rectified_img": rectified_img,
        "osm_tile_img": osm_tile_img,
        "prediction_map_img": prediction_map_img,
        "neural_map_img": neural_map_img,
        "stats_html": stats_html,
    }

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        template = f.read()

    html = template.replace(
        "const imageData = DATA_PLACEHOLDER;",
        "const imageData = " + json.dumps(image_data, allow_nan=False) + ";",
    )
    return html


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {RUN_NAME}_generate_detail.py <image_id>")
        sys.exit(1)

    try:
        image_id = int(sys.argv[1])
        html = generate_detail_page(image_id)
        output_file = BASE_DIR / f"{RUN_NAME}_detail_{image_id}.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[OK] Generated: {output_file}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
'''
    return (
        script_template.replace("__RUN_DIR_NAME_JSON__", json.dumps(run_dir.name))
        .replace("__RUN_NAME_JSON__", json.dumps(run_name))
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""
    print("=" * 70)
    print("OrienterNet Results Dashboard Generator")
    print("=" * 70)

    try:
        run_dir = get_latest_run_dir()
        run_name = get_run_name(run_dir)
        print(f"\n[OK] Using run directory: {run_dir}")
        print(f"[OK] Run name: {run_name}")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    try:
        df = load_results_csv(run_dir)
        print(f"[OK] Loaded {len(df)} results")
    except Exception as exc:
        print(f"\n[ERROR] Loading results: {exc}")
        sys.exit(1)

    try:
        print("\n[STEP] Generating main dashboard page...")
        main_html = generate_main_html(run_dir, run_name, df)
        main_output_path = OUTPUT_DIR / f"{run_name}_dashboard.html"
        with open(main_output_path, "w", encoding="utf-8") as f:
            f.write(main_html)
        print(f"[OK] Main page generated: {main_output_path}")
    except Exception as exc:
        print(f"\n[ERROR] Generating main page: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        print("\n[STEP] Generating notebook-style analysis page...")
        analysis_html = generate_analysis_html(run_dir, run_name, df)
        analysis_output_path = OUTPUT_DIR / f"{run_name}_analysis.html"
        with open(analysis_output_path, "w", encoding="utf-8") as f:
            f.write(analysis_html)
        print(f"[OK] Analysis page generated: {analysis_output_path}")
    except Exception as exc:
        print(f"\n[ERROR] Generating analysis page: {exc}")
        import traceback
        traceback.print_exc()
        analysis_output_path = None

    try:
        print("\n[STEP] Generating detail page template...")
        template_html = generate_detail_html_template(run_name)
        template_output_path = OUTPUT_DIR / f"{run_name}_detail_template.html"
        with open(template_output_path, "w", encoding="utf-8") as f:
            f.write(template_html)
        print(f"[OK] Template generated: {template_output_path}")
    except Exception as exc:
        print(f"\n[ERROR] Generating template: {exc}")
        import traceback
        traceback.print_exc()
        template_output_path = None

    try:
        print("\n[STEP] Creating detail page generator script...")
        generator_script = create_detail_generator_script(run_dir, run_name, df)
        gen_script_path = OUTPUT_DIR / f"{run_name}_generate_detail.py"
        with open(gen_script_path, "w", encoding="utf-8") as f:
            f.write(generator_script)
        print(f"[OK] Generator script created: {gen_script_path}")
    except Exception as exc:
        print(f"\n[ERROR] Creating generator script: {exc}")
        import traceback
        traceback.print_exc()
        gen_script_path = None

    print("\n" + "=" * 70)
    print("[OK] Dashboard generation complete!")
    print(f"\n[OUTPUT] Main dashboard: {main_output_path}")
    if analysis_output_path:
        print(f"[OUTPUT] Analysis page: {analysis_output_path}")
    if template_output_path:
        print(f"[OUTPUT] Template: {template_output_path}")
    if gen_script_path:
        print(f"[OUTPUT] Generator: {gen_script_path}")
        print("\nTo generate individual detail pages on-demand:")
        print(f"   python {gen_script_path} <image_id>")
    print("=" * 70)


if __name__ == "__main__":
    main()
