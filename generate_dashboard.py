#!/usr/bin/env python3
"""
OrienterNet Results Dashboard Generator

Generates an interactive HTML dashboard for viewing OrienterNet prediction results.
The dashboard includes a main table view with clickable rows linking to detailed artifact pages.

CONFIGURATION:
- BASE_DATA_DIR: Path to the data directory containing runs
- RUNS_PATH: Path to the runs folder
"""

import os
import sys

# Fix encoding on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from math import radians, cos, sin, asin, sqrt

# ============================================================================
# PATH CONFIGURATION - ADJUST THESE IF YOUR STRUCTURE CHANGES
# ============================================================================
BASE_PROJECT_DIR = Path(__file__).parent
BASE_DATA_DIR = BASE_PROJECT_DIR / "data"
RUNS_PATH = BASE_DATA_DIR / "runs"
OUTPUT_DIR = RUNS_PATH  # Where to save the final HTML

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_latest_run_dir() -> Path:
    """Get the latest run directory."""
    run_dirs = sorted([d for d in RUNS_PATH.iterdir() if d.is_dir()],
                     key=lambda x: x.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {RUNS_PATH}")
    return run_dirs[0]

def get_run_name(run_dir: Path) -> str:
    """Extract run name from directory path."""
    return run_dir.name

def load_results_csv(run_dir: Path) -> pd.DataFrame:
    """Load the results CSV from a run directory."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Convert coordinate and angle columns to numeric
    numeric_cols = ['gt_latitude', 'gt_longitude', 'gt_yaw',
                   'pred_latitude', 'pred_longitude', 'pred_yaw', 'pred_probability',
                   'exif_focal_35mm', 'exif_focal_ratio', 'exif_altitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def load_artifact_json(artifact_dir: Path, filename: str) -> Dict:
    """Safely load a JSON file from an artifact directory."""
    json_path = artifact_dir / filename
    if not json_path.exists():
        return {}
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}

def image_to_base64(image_path: Path) -> str:
    """Convert an image file to base64 data URI."""
    if not image_path.exists():
        return ""
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = image_path.suffix.lower()[1:]
        if ext == 'jpg':
            ext = 'jpeg'
        return f"data:image/{ext};base64,{data}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def haversine(lon1, lat1, lon2, lat2):
    """Calculate great circle distance between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def normalize_angle(angle):
    """Normalize angle to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

# ============================================================================
# HTML GENERATION - MAIN PAGE
# ============================================================================

def generate_main_html(run_dir: Path, run_name: str, df: pd.DataFrame) -> str:
    """Generate the main table/overview HTML page."""

    # Select relevant columns for the table
    table_columns = [
        'image_id', 'image_path', 'h5_id_dataset',
        'gt_yaw', 'pred_yaw', 'pred_probability'
    ]

    # Only include columns that exist
    table_columns = [col for col in table_columns if col in df.columns]

    # Create display dataframe
    display_df = df[table_columns].copy()

    # Add distance and yaw error calculations
    display_df['distance_m'] = df.apply(
        lambda row: haversine(
            row['pred_longitude'], row['pred_latitude'],
            row['gt_longitude'], row['gt_latitude']
        ) if all(pd.notna([row['pred_longitude'], row['pred_latitude'],
                          row['gt_longitude'], row['gt_latitude']])) else np.nan,
        axis=1
    )

    display_df['yaw_error'] = df.apply(
        lambda row: normalize_angle(row['pred_yaw'] - row['gt_yaw'])
        if all(pd.notna([row['pred_yaw'], row['gt_yaw']])) else np.nan,
        axis=1
    )

    # Format numeric columns
    for col in ['gt_yaw', 'pred_yaw', 'yaw_error']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}°" if pd.notna(x) else "")

    for col in ['distance_m']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}m" if pd.notna(x) else "")

    # Format probability as percentage
    if 'pred_probability' in display_df.columns:
        display_df['pred_probability'] = display_df['pred_probability'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
        )

    # Generate table HTML
    table_html = display_df.to_html(classes='table table-striped table-hover', index=False, escape=False)

    # Wrap rows with onclick handlers
    rows = table_html.split('<tbody>')[1].split('</tbody>')[0]
    new_rows = ""
    for i, row in enumerate(rows.split('</tr>')):
        if row.strip():
            new_rows += f'<tr onclick="viewDetails({i})" style="cursor: pointer;" class="detail-row">{row}</tr>'

    table_html = table_html.split('<tbody>')[0] + '<tbody>' + new_rows + '</tbody>' + table_html.split('</tbody>')[1]

    # Generate summary statistics
    dist_avg = df['distance_m'].mean() if 'distance_m' in df.columns else 0
    yaw_avg = np.abs(df['yaw_error']).mean() if 'yaw_error' in df.columns else 0

    stats_html = f"""
    <div class="alert alert-info" role="alert">
        <strong>Dataset Summary:</strong>
        Total Images: {len(df)} | 
        Avg Distance Error: {dist_avg:.2f}m |
        Avg Yaw Error: {yaw_avg:.2f}°
    </div>
    """

    images_json = json.dumps(df['image_id'].tolist())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrienterNet Results Dashboard - {run_name}</title>
    
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        body {{
            background-color: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}
        
        .navbar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }}
        
        .detail-row {{
            transition: background-color 0.2s ease;
        }}
        
        .detail-row:hover {{
            background-color: #e9ecef !important;
        }}
        
        .table-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,.1);
        }}
        
        .header-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,.1);
        }}
        
        h1 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .run-info {{
            color: #666;
            font-size: 0.95rem;
        }}
        
        .table {{
            margin-bottom: 0;
        }}
        
        .table thead {{
            background-color: #667eea;
            color: white;
        }}
        
        .table thead th {{
            border-color: #667eea;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-map-location-dot"></i> OrienterNet Results Dashboard
            </span>
        </div>
    </nav>
    
    <div class="container-fluid" style="padding: 20px;">
        <div class="header-section">
            <h1>Run: {run_name}</h1>
            <div class="run-info">
                <i class="fas fa-database"></i> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <i class="fas fa-folder"></i> Path: {run_dir}
            </div>
        </div>
        
        {stats_html}
        
        <div class="table-container">
            <h5 class="mb-3"><i class="fas fa-table"></i> Results Overview</h5>
            <div style="overflow-x: auto;">
                {table_html}
            </div>
        </div>
    </div>
    
    <!-- Scripts -->
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
# HTML GENERATION - DETAIL PAGE TEMPLATE
# ============================================================================

def generate_detail_html_template(run_name: str) -> str:
    """Generate a detail page template that will be filled with data."""

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Detail</title>
    
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            background-color: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;
            height: 100vh;
            overflow: hidden;
        }}
        
        .navbar {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        
        .container-detail {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr;
            height: calc(100vh - 56px);
            gap: 0;
        }}
        
        .left-panel {{
            grid-column: 1;
            grid-row: 1 / 3;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #ddd;
            background: white;
            overflow: auto;
        }}
        
        .right-top {{
            grid-column: 2;
            grid-row: 1;
            display: flex;
            flex-direction: column;
            border-bottom: 1px solid #ddd;
            background: white;
        }}
        
        .right-bottom {{
            grid-column: 2;
            grid-row: 2;
            overflow-y: auto;
            background: white;
        }}
        
        #map {{
            width: 100%;
            height: 100%;
            min-height: 300px;
        }}
        
        .map-header {{
            padding: 10px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            font-weight: 600;
            font-size: 0.95rem;
        }}
        
        .map-controls {{
            padding: 10px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            font-size: 0.85rem;
            max-height: 150px;
            overflow-y: auto;
        }}
        
        .map-controls label {{
            margin-bottom: 3px;
            cursor: pointer;
            font-weight: 500;
        }}
        
        .image-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            background: #f0f0f0;
        }}
        
        .image-container img {{
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            border-radius: 4px;
        }}
        
        .expand-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            background: rgba(0,0,0,0.6);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 0.85rem;
        }}
        
        .expand-btn:hover {{
            background: rgba(0,0,0,0.8);
        }}
        
        .stats-container {{
            padding: 15px;
            overflow-y: auto;
            max-height: calc(100vh - 150px);
        }}
        
        .stats-section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .stats-section:last-child {{
            border-bottom: none;
        }}
        
        .stats-section h6 {{
            color: #667eea;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        dl {{
            margin-bottom: 0;
            font-size: 0.9rem;
        }}
        
        dt {{
            font-weight: 600;
            color: #333;
        }}
        
        dd {{
            color: #666;
            margin-left: 0;
        }}
        
        .modal {{
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
        }}
        
        .modal.active {{
            display: flex;
        }}
        
        .modal-content {{
            max-width: 80vw;
            max-height: 80vh;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .modal-content img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        
        .close-btn {{
            position: absolute;
            top: -40px;
            right: 0;
            color: white;
            font-size: 36px;
            font-weight: bold;
            cursor: pointer;
            background: none;
            border: none;
            padding: 0;
        }}
        
        .close-btn:hover {{
            color: #ccc;
        }}
        
        .back-btn {{
            display: inline-block;
            padding: 10px 15px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin: 10px;
            font-size: 0.9rem;
            transition: background 0.2s;
        }}
        
        .back-btn:hover {{
            background: #764ba2;
            text-decoration: none;
            color: white;
        }}
        
        .image-id-header {{
            padding: 15px;
            background: #667eea;
            color: white;
            font-weight: 600;
            border-bottom: 1px solid #ddd;
        }}
        
        .yaw-indicator {{
            display: inline-block;
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid">
            <a href="{run_name}_dashboard.html" class="back-btn"><i class="fas fa-arrow-left"></i> Back</a>
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-image"></i> <span id="imageTitle">Image Details</span>
            </span>
        </div>
    </nav>
    
    <div class="container-detail">
        <!-- Left Panel: Map -->
        <div class="left-panel">
            <div class="map-header">
                <i class="fas fa-map-location-dot"></i> Geolocation Context
            </div>
            <div class="map-controls">
                <div>
                    <label><input type="checkbox" id="boundaryLayer" checked> Request Tile (136x136)</label><br>
                    <label><input type="checkbox" id="predictionLayer"> Prediction Map</label><br>
                    <label><input type="checkbox" id="neuralLayer"> Neural Activation</label>
                </div>
                <div style="margin-top: 8px;">
                    <small><strong>Basemap:</strong></small><br>
                    <label style="font-weight: normal;"><input type="radio" name="basemap" value="osm" checked> OpenStreetMap</label><br>
                    <label style="font-weight: normal;"><input type="radio" name="basemap" value="fetched"> Fetched Aerial</label><br>
                    <label style="font-weight: normal;"><input type="radio" name="basemap" value="satellite"> Satellite</label>
                </div>
            </div>
            <div id="map"></div>
        </div>
        
        <!-- Right Top Panel: Rectified Image -->
        <div class="right-top">
            <div class="image-id-header">
                <i class="fas fa-photo-film"></i> Rectified Image
            </div>
            <div class="image-container">
                <button class="expand-btn" onclick="expandImage()"><i class="fas fa-expand"></i> Expand</button>
                <img id="rectifiedImage" src="" alt="Rectified Image">
            </div>
        </div>
        
        <!-- Right Bottom Panel: Statistics -->
        <div class="right-bottom">
            <div class="stats-container" id="statsContainer">
                <!-- Stats will be injected here -->
            </div>
        </div>
    </div>
    
    <!-- Image Expansion Modal -->
    <div id="imageModal" class="modal">
        <div class="modal-content">
            <button class="close-btn" onclick="closeImageModal()">&times;</button>
            <img id="expandedImage" src="" alt="Expanded Image">
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Data will be embedded by the generator script
        const imageData = DATA_PLACEHOLDER;
        
        // Initialize map
        const map = L.map('map').setView(
            [imageData.gt_lat, imageData.gt_lon], 
            15
        );
        
        // Basemap layers
        const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }});
        
        const fetchedLayer = L.tileLayer('file:///path/placeholder', {{
            attribution: 'Fetched Tiles',
            maxZoom: 19,
            errorTile: ''
        }});
        
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: '© Esri',
            maxZoom: 19
        }});
        
        osmLayer.addTo(map);
        
        // Boundary rectangle (136x136 request tile)
        const boundaryRadius = 0.001; // Approximately 111 meters
        const boundaryRect = L.rectangle([
            [imageData.gt_lat - boundaryRadius, imageData.gt_lon - boundaryRadius],
            [imageData.gt_lat + boundaryRadius, imageData.gt_lon + boundaryRadius]
        ], {{
            color: '#555555',
            weight: 2,
            opacity: 0.8,
            fill: false,
            dashArray: '5, 5'
        }}).addTo(map);
        
        // Ground Truth Marker (Orange)
        const gtMarker = L.circleMarker([imageData.gt_lat, imageData.gt_lon], {{
            radius: 8,
            fillColor: '#FF8C00',
            color: '#000',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8,
            title: 'Ground Truth'
        }}).bindPopup('Ground Truth Position').addTo(map);
        
        // Prediction Marker (Yellow) with yaw angle
        const predMarker = L.circleMarker([imageData.pred_lat, imageData.pred_lon], {{
            radius: 8,
            fillColor: '#FFD700',
            color: '#000',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8,
            title: 'Prediction'
        }}).bindPopup('Predicted Position').addTo(map);
        
        // Add yaw angle indicator
        const yawRad = imageData.pred_yaw * Math.PI / 180;
        const scale = 0.00005; // Scale for the indicator line
        const endLat = imageData.pred_lat + Math.cos(yawRad) * scale;
        const endLon = imageData.pred_lon + Math.sin(yawRad) * scale;
        
        L.polyline([
            [imageData.pred_lat, imageData.pred_lon],
            [endLat, endLon]
        ], {{
            color: '#FFD700',
            weight: 3,
            opacity: 0.8,
            dashArray: 'none'
        }}).addTo(map);
        
        // Overlay layers (initially empty, can be populated with images)
        const predictionOverlay = L.layerGroup();
        const neuralOverlay = L.layerGroup();
        
        // Add prediction map if available
        if (imageData.prediction_map_img) {{
            const predImg = L.imageOverlay(imageData.prediction_map_img, [
                [imageData.gt_lat - 0.001, imageData.gt_lon - 0.001],
                [imageData.gt_lat + 0.001, imageData.gt_lon + 0.001]
            ], {{ opacity: 0.6 }});
            predictionOverlay.addLayer(predImg);
        }}
        
        // Add neural map if available
        if (imageData.neural_map_img) {{
            const neuralImg = L.imageOverlay(imageData.neural_map_img, [
                [imageData.gt_lat - 0.001, imageData.gt_lon - 0.001],
                [imageData.gt_lat + 0.001, imageData.gt_lon + 0.001]
            ], {{ opacity: 0.6 }});
            neuralOverlay.addLayer(neuralImg);
        }}
        
        // Basemap switching
        document.querySelectorAll('input[name="basemap"]').forEach(radio => {{
            radio.addEventListener('change', (e) => {{
                map.eachLayer((layer) => {{
                    if (layer instanceof L.TileLayer) map.removeLayer(layer);
                }});
                
                if (e.target.value === 'osm') {{
                    osmLayer.addTo(map);
                }} else if (e.target.value === 'fetched') {{
                    osmLayer.addTo(map); // Fallback to OSM if fetched not available
                }} else {{
                    satelliteLayer.addTo(map);
                }}
            }});
        }});
        
        // Layer toggle
        document.getElementById('boundaryLayer').addEventListener('change', (e) => {{
            if (e.target.checked) {{
                map.addLayer(boundaryRect);
            }} else {{
                map.removeLayer(boundaryRect);
            }}
        }});
        
        document.getElementById('predictionLayer').addEventListener('change', (e) => {{
            if (e.target.checked && predictionOverlay.getLayers().length > 0) {{
                map.addLayer(predictionOverlay);
            }} else {{
                map.removeLayer(predictionOverlay);
            }}
        }});
        
        document.getElementById('neuralLayer').addEventListener('change', (e) => {{
            if (e.target.checked && neuralOverlay.getLayers().length > 0) {{
                map.addLayer(neuralOverlay);
            }} else {{
                map.removeLayer(neuralOverlay);
            }}
        }});
        
        // Setup image
        document.getElementById('rectifiedImage').src = imageData.rectified_img;
        document.getElementById('expandedImage').src = imageData.rectified_img;
        document.getElementById('imageTitle').textContent = 'Image ' + imageData.image_id + ' Details';
        
        // Setup stats
        document.getElementById('statsContainer').innerHTML = imageData.stats_html;
        
        // Modal functions
        function expandImage() {{
            document.getElementById('imageModal').classList.add('active');
        }}
        
        function closeImageModal() {{
            document.getElementById('imageModal').classList.remove('active');
        }}
        
        document.getElementById('imageModal').addEventListener('click', (e) => {{
            if (e.target.id === 'imageModal') {{
                closeImageModal();
            }}
        }});
        
        // Keyboard close
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeImageModal();
        }});
    </script>
</body>
</html>"""
    return html_template


# ============================================================================
# GENERATOR SCRIPT FOR ON-DEMAND DETAIL PAGES
# ============================================================================

def create_detail_generator_script(run_dir: Path, run_name: str, df: pd.DataFrame) -> str:
    """Create a script to generate detail pages on-demand."""

    # Use raw string with proper escaping to avoid f-string interpolation issues
    script_content = f'''#!/usr/bin/env python3
"""
On-demand Detail Page Generator for {run_name}
Usage: python {run_name}_generate_detail.py <image_id>
"""

import sys
import json
import pandas as pd
import base64
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

# Configuration
BASE_DIR = Path(__file__).parent
RUN_DIR = BASE_DIR / "{run_dir.name}"
RESULTS_CSV = RUN_DIR / "results.csv"

# Load template
TEMPLATE_FILE = BASE_DIR / "{run_name}_detail_template.html"

def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URI."""
    if not image_path.exists():
        return ""
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = image_path.suffix.lower()[1:]
        if ext == 'jpg':
            ext = 'jpeg'
        return f"data:image/{{ext}};base64,{{data}}"
    except:
        return ""

def load_artifact_json(artifact_dir: Path, filename: str) -> dict:
    """Load JSON artifact file."""
    json_path = artifact_dir / filename
    if not json_path.exists():
        return {{}}
    try:
        with open(json_path) as f:
            return json.load(f)
    except:
        return {{}}

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371000

def normalize_angle(angle):
    """Normalize angle to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def get_yaw_arrow(yaw: float) -> str:
    """Get directional arrow based on yaw angle."""
    # Map yaw angles to directional indicators
    yaw_norm = normalize_angle(yaw)
    
    if -22.5 <= yaw_norm < 22.5:
        return "→"  # Right
    elif 22.5 <= yaw_norm < 67.5:
        return "↘"  # Down-Right
    elif 67.5 <= yaw_norm < 112.5:
        return "↓"  # Down
    elif 112.5 <= yaw_norm < 157.5:
        return "↙"  # Down-Left
    elif -157.5 <= yaw_norm < -112.5:
        return "↙"  # Down-Left
    elif -112.5 <= yaw_norm < -67.5:
        return "↓"  # Down
    elif -67.5 <= yaw_norm < -22.5:
        return "↘"  # Down-Right
    else:
        return "→"  # Default Right

def generate_detail_page(image_id: int):
    """Generate detail page for specific image."""
    
    # Load results
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    df['image_id'] = pd.to_numeric(df['image_id'], errors='coerce')
    
    # Find row
    row_data = df[df['image_id'] == image_id]
    if row_data.empty:
        return f"<h1>Error: Image {{image_id}} not found</h1>"
    
    row_data = row_data.iloc[0]
    
    # Load artifacts
    artifact_dir = RUN_DIR / "artifacts" / f"image_{{image_id}}"
    if not artifact_dir.exists():
        return f"<h1>Error: Artifact directory not found</h1>"
    
    artifacts_data = load_artifact_json(artifact_dir, "artifacts.json")
    camera_data = load_artifact_json(artifact_dir, "camera.json")
    
    # Load images
    rectified_img = image_to_base64(artifact_dir / "rectified_image.jpg")
    prediction_map_img = image_to_base64(artifact_dir / "prediction_map.jpg")
    neural_map_img = image_to_base64(artifact_dir / "neural_map_rgb.jpg")
    
    # Extract data
    gt_lat = float(row_data.get('gt_latitude', 0))
    gt_lon = float(row_data.get('gt_longitude', 0))
    pred_lat = float(row_data.get('pred_latitude', 0))
    pred_lon = float(row_data.get('pred_longitude', 0))
    gt_yaw = float(row_data.get('gt_yaw', 0))
    pred_yaw = float(row_data.get('pred_yaw', 0))
    pred_prob = float(row_data.get('pred_probability', 0))
    
    # Calculate errors
    distance_error = haversine(pred_lon, pred_lat, gt_lon, gt_lat)
    yaw_error = normalize_angle(pred_yaw - gt_yaw)
    
    # Get yaw indicators
    gt_yaw_arrow = get_yaw_arrow(gt_yaw)
    pred_yaw_arrow = get_yaw_arrow(pred_yaw)
    
    # Build stats HTML
    exif_make = row_data.get('exif_make', 'Unknown')
    exif_model = row_data.get('exif_model', 'Unknown')
    width = camera_data.get('width', 'N/A')
    height = camera_data.get('height', 'N/A')
    
    stats_html = f"""
    <div class="stats-section">
        <h6>Camera Information</h6>
        <dl class="row">
            <dt class="col-6">Resolution:</dt>
            <dd class="col-6">{{width}}x{{height}}</dd>
            <dt class="col-6">Focal Length X:</dt>
            <dd class="col-6">{{camera_data.get('fx', 'N/A'):.2f}}</dd>
            <dt class="col-6">Principal Point:</dt>
            <dd class="col-6">({{camera_data.get('cx', 'N/A')}}, {{camera_data.get('cy', 'N/A')}})</dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6>Prediction Results</h6>
        <dl class="row">
            <dt class="col-6">Distance Error:</dt>
            <dd class="col-6"><strong>{{distance_error:.2f}} m</strong></dd>
            <dt class="col-6">Yaw Error:</dt>
            <dd class="col-6"><strong>{{yaw_error:.2f}}°</strong></dd>
            <dt class="col-6">Confidence:</dt>
            <dd class="col-6">{{pred_prob*100:.1f}}%</dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6>Ground Truth</h6>
        <dl class="row">
            <dt class="col-6">Position:</dt>
            <dd class="col-6">{{gt_lat:.6f}}<br>{{gt_lon:.6f}}</dd>
            <dt class="col-6">Yaw (Direction):</dt>
            <dd class="col-6">{{gt_yaw:.2f}}° <span class="yaw-indicator">{{gt_yaw_arrow}}</span></dd>
        </dl>
    </div>
    <div class="stats-section">
        <h6>Prediction</h6>
        <dl class="row">
            <dt class="col-6">Position:</dt>
            <dd class="col-6">{{pred_lat:.6f}}<br>{{pred_lon:.6f}}</dd>
            <dt class="col-6">Yaw (Direction):</dt>
            <dd class="col-6">{{pred_yaw:.2f}}° <span class="yaw-indicator">{{pred_yaw_arrow}}</span></dd>
        </dl>
    </div>
    """
    
    # Build data object for JavaScript
    image_data = {{
        'image_id': image_id,
        'gt_lat': gt_lat,
        'gt_lon': gt_lon,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon,
        'pred_yaw': pred_yaw,
        'rectified_img': rectified_img,
        'prediction_map_img': prediction_map_img,
        'neural_map_img': neural_map_img,
        'stats_html': stats_html
    }}
    
    # Load template
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Replace placeholder
    html = template.replace(
        'const imageData = DATA_PLACEHOLDER;',
        'const imageData = ' + json.dumps(image_data) + ';'
    )
    
    return html

def main():
    if len(sys.argv) < 2:
        print("Usage: python {run_name}_generate_detail.py <image_id>")
        sys.exit(1)
    
    try:
        image_id = int(sys.argv[1])
        html = generate_detail_page(image_id)
        
        output_file = BASE_DIR / f"{run_name}_detail_{{image_id}}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[OK] Generated: {{output_file}}")
    except Exception as e:
        print(f"[ERROR] {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    return script_content

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("OrienterNet Results Dashboard Generator")
    print("=" * 70)

    # Get run directory
    try:
        run_dir = get_latest_run_dir()
        run_name = get_run_name(run_dir)
        print(f"\n[OK] Using run directory: {run_dir}")
        print(f"[OK] Run name: {run_name}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # Load results
    try:
        df = load_results_csv(run_dir)
        print(f"[OK] Loaded {len(df)} results")
    except Exception as e:
        print(f"\n[ERROR] Loading results: {e}")
        sys.exit(1)

    # Generate main HTML
    try:
        print("\n[STEP] Generating main dashboard page...")
        main_html = generate_main_html(run_dir, run_name, df)
        main_output_path = OUTPUT_DIR / f"{run_name}_dashboard.html"
        with open(main_output_path, 'w', encoding='utf-8') as f:
            f.write(main_html)
        print(f"[OK] Main page generated: {main_output_path}")
    except Exception as e:
        print(f"\n[ERROR] Generating main page: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate detail template
    try:
        print("\n[STEP] Generating detail page template...")
        template_html = generate_detail_html_template(run_name)
        template_output_path = OUTPUT_DIR / f"{run_name}_detail_template.html"
        with open(template_output_path, 'w', encoding='utf-8') as f:
            f.write(template_html)
        print(f"[OK] Template generated: {template_output_path}")
    except Exception as e:
        print(f"\n[ERROR] Generating template: {e}")
        import traceback
        traceback.print_exc()

    # Create generator script
    try:
        print("\n[STEP] Creating detail page generator script...")
        generator_script = create_detail_generator_script(run_dir, run_name, df)
        gen_script_path = OUTPUT_DIR / f"{run_name}_generate_detail.py"
        with open(gen_script_path, 'w', encoding='utf-8') as f:
            f.write(generator_script)
        print(f"[OK] Generator script created: {gen_script_path}")
    except Exception as e:
        print(f"\n[ERROR] Creating generator script: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("[OK] Dashboard generation complete!")
    print(f"\n[OUTPUT] Main dashboard: {main_output_path}")
    print(f"[OUTPUT] Template: {template_output_path}")
    print(f"[OUTPUT] Generator: {gen_script_path}")
    print(f"\nTo generate individual detail pages on-demand:")
    print(f"   python {gen_script_path} <image_id>")
    print("=" * 70)

if __name__ == "__main__":
    main()





