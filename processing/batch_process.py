#!/usr/bin/env python3

"""
Run OrienterNet localization on images and export predictions to CSV.

Supports two image layouts:

1. Extracted images named like:
       data/extracted_images/image_<id>.jpg
       data/extracted_images/image_<id>.png

2. Original nested images resolved through a CSV manifest with columns:
       id, relative_image_path

   Example manifest row:
       id,relative_image_path
       123,/mapillary_dataset/175/175345001056338.jpg

   With:
       --images-dir-original data/og
       --image-manifest-csv data/og_image_paths.csv

   the image path resolves to:
       data/og/175/175345001056338.jpg

Expected H5 metadata structure:
    metadata/id
    metadata/id_dataset
    metadata/latitude
    metadata/longitude
    metadata/yaw

Run naming:
    By default, outputs are written into a parameter-based run folder:

        data/runs/process_<tile-size>t_<num-rotations>r_<data-version>/

    Example:
        python processing/batch_process.py \
            --tile-size 136 \
            --num-rotations 256 \
            --data-version og

    Writes:
        data/runs/process_136t_256r_og/results.csv
        data/runs/process_136t_256r_og/artifacts/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

# =============================================================================
# Project import setup
# =============================================================================

# batch_process.py is inside:
#   FG-OrienterNet/processing/batch_process.py
# so parents[1] is:
#   FG-OrienterNet/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set CUDA allocator config BEFORE importing torch.
# On Windows, expandable_segments may not be supported, so use max_split_size_mb.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

# =============================================================================
# Imports requiring project / ML dependencies
# =============================================================================

import h5py
import numpy as np
import torch
from PIL import Image

from maploc.demo import Demo
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap
from maploc.utils.exif import EXIF
from maploc.utils.viz_2d import features_to_RGB
from maploc.utils.viz_localization import likelihood_overlay

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTPUT_CSV = "data/orienternet_results.csv"
DEFAULT_OUTPUT_ARTIFACTS_DIR = "data/image_outputs"
DEFAULT_RUNS_DIR = "data/runs"

CSV_HEADERS = [
    "image_id",
    "image_path",
    "h5_id_dataset",
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
    "exif_make",
    "exif_model",
    "exif_focal_35mm",
    "exif_focal_ratio",
    "exif_orientation",
    "exif_altitude",
    "error_message",
]

# OrienterNet OSM raster layer limits.
RASTER_LAYER_MAX_VALUES = [7, 10, 33]

# Prefix in the manifest that should be ignored because --images-dir-original
# already points to the dataset root below it.
DEFAULT_MANIFEST_PREFIX_TO_STRIP = "mapillary_dataset/"

# =============================================================================
# Data models
# =============================================================================


@dataclass
class GroundTruthMetadata:
    id_dataset: str
    latitude: float
    longitude: float
    yaw: float


@dataclass
class LocationPrior:
    """Location prior with source information."""

    latitude: float
    longitude: float
    yaw: Optional[float] = None
    source: str = "unknown"  # "h5" | "csv" | "exif" | "combined"


@dataclass
class ProcessorConfig:
    h5_path: Path
    images_dir: Path
    output_csv: Path
    output_artifacts_dir: Path
    tile_size_meters: int = 64
    num_rotations: int = 128
    device: str = "cuda"
    save_artifacts: bool = False
    images_dir_original: Optional[Path] = None
    image_manifest_csv: Optional[Path] = None
    manifest_prefix_to_strip: str = DEFAULT_MANIFEST_PREFIX_TO_STRIP
    prior_strategy: str = "h5"
    csv_prior_path: Optional[Path] = None


# =============================================================================
# PyTorch configuration
# =============================================================================


def configure_pytorch_memory() -> None:
    """Configure PyTorch runtime behavior."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logger.info(
        "PyTorch CUDA allocator config: %s",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )

    if torch.cuda.is_available():
        logger.info("CUDA available: True")
        logger.info("CUDA device count: %d", torch.cuda.device_count())

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            logger.info(
                "GPU %d: %s, %.2f GB VRAM",
                idx,
                props.name,
                props.total_memory / 1024**3,
            )
    else:
        logger.info("CUDA available: False")


# =============================================================================
# Run naming helpers
# =============================================================================


def sanitize_run_label(value: str) -> str:
    """Return a filesystem-safe run label component."""
    cleaned = value.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "data"


def build_run_name(tile_size: int, num_rotations: int, data_version: str) -> str:
    """Build a readable run name from processing parameters.

    Example:
        process_272t_256r_h5v
    """
    clean_version = sanitize_run_label(data_version)
    return f"process_{tile_size}t_{num_rotations}r_{clean_version}"


# =============================================================================
# Main processor
# =============================================================================


class ImageProcessor:
    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert common Python/numpy/torch/custom objects to JSON-safe values."""
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()

        if isinstance(value, np.ndarray):
            return ImageProcessor._json_safe(value.tolist())

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, dict):
            return {str(k): ImageProcessor._json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [ImageProcessor._json_safe(v) for v in value]

        # Handle custom objects like BoundaryBox.
        if hasattr(value, "__dict__"):
            return {
                str(k): ImageProcessor._json_safe(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }

        # Last-resort fallback so json.dump never crashes.
        return str(value)

    def __init__(self, config: ProcessorConfig):
        self.config = config

        device = torch.device(config.device)

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but torch.cuda.is_available() is False."
                )

            torch.cuda.set_device(device)

            logger.info("Set active CUDA device to %s", device)
            logger.info("CUDA current device index: %d", torch.cuda.current_device())
            logger.info("CUDA device name: %s", torch.cuda.get_device_name(device))

        logger.info("Loading OrienterNet model on %s...", device)

        self.demo = Demo(
            num_rotations=config.num_rotations,
            device=device,
        )

        logger.info("Model loaded successfully on %s", device)
        logger.info("Demo device: %s", self.demo.device)

        self._init_csv()

    # -------------------------------------------------------------------------
    # CSV Prior helpers
    # -------------------------------------------------------------------------

    def load_csv_priors(self) -> Dict[int, LocationPrior]:
        """Load optional location priors from CSV file.

        Expected columns for this optional prior CSV:
            image_id, latitude, longitude, yaw

        This is separate from image_manifest_csv, which is only for resolving
        nested image file paths.
        """
        csv_priors: Dict[int, LocationPrior] = {}

        if self.config.csv_prior_path is None:
            return csv_priors

        if not self.config.csv_prior_path.exists():
            logger.warning("CSV prior file does not exist: %s", self.config.csv_prior_path)
            return csv_priors

        logger.info("Loading CSV priors from %s", self.config.csv_prior_path)

        try:
            with self.config.csv_prior_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_id = self._safe_int(row.get("image_id"))
                    if image_id is None:
                        continue

                    lat = self._safe_float(row.get("latitude"))
                    lon = self._safe_float(row.get("longitude"))
                    yaw = self._safe_float(row.get("yaw"))

                    if lat is not None and lon is not None:
                        csv_priors[image_id] = LocationPrior(
                            latitude=lat,
                            longitude=lon,
                            yaw=yaw,
                            source="csv",
                        )

            logger.info("Loaded %d CSV priors", len(csv_priors))

        except Exception as exc:
            logger.error("Failed to load CSV priors: %s", exc)

        return csv_priors

    # -------------------------------------------------------------------------
    # GPU memory management
    # -------------------------------------------------------------------------

    @staticmethod
    def _cleanup_gpu_memory() -> None:
        """Clear CUDA cache after each image."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("GPU cache cleanup failed: %s", exc)

    # -------------------------------------------------------------------------
    # CSV helpers
    # -------------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Create output CSV if it does not already exist."""
        if self.config.output_csv.exists():
            return

        logger.info("Creating CSV file: %s", self.config.output_csv)
        self.config.output_csv.parent.mkdir(parents=True, exist_ok=True)

        with self.config.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

    def _append_csv_row(self, row: Dict[str, Any]) -> None:
        """Append one result row to CSV."""
        try:
            with self.config.output_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row)
        except Exception as exc:
            logger.error("Failed to append row to CSV: %s", exc)

    def _replace_csv_row(self, image_id: int, new_row: Dict[str, Any]) -> None:
        """Replace existing CSV row for one image id."""
        try:
            rows = []

            with self.config.output_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    current_id = self._safe_int(row.get("image_id"))

                    if current_id == image_id:
                        rows.append(new_row)
                    else:
                        rows.append(row)

            with self.config.output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                writer.writerows(rows)

        except Exception as exc:
            logger.error("Failed to update CSV row for image %s: %s", image_id, exc)

    # -------------------------------------------------------------------------
    # H5 metadata
    # -------------------------------------------------------------------------

    def load_h5_metadata(self) -> Dict[int, GroundTruthMetadata]:
        """Load ground-truth metadata from the H5 file."""
        logger.info("Loading metadata from %s", self.config.h5_path)

        if not self.config.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.config.h5_path}")

        metadata: Dict[int, GroundTruthMetadata] = {}

        with h5py.File(self.config.h5_path, "r") as f:
            h5_ids = f["metadata"]["id"][:]
            id_datasets = f["metadata"]["id_dataset"][:]
            latitudes = f["metadata"]["latitude"][:]
            longitudes = f["metadata"]["longitude"][:]
            yaws = f["metadata"]["yaw"][:]

            for idx, h5_id in enumerate(h5_ids):
                image_id = int(h5_id)
                id_dataset = self._decode_h5_value(id_datasets[idx])

                metadata[image_id] = GroundTruthMetadata(
                    id_dataset=id_dataset,
                    latitude=float(latitudes[idx]),
                    longitude=float(longitudes[idx]),
                    yaw=float(yaws[idx]),
                )

        logger.info("Loaded metadata for %d images", len(metadata))
        return metadata

    # -------------------------------------------------------------------------
    # Processed / retry state
    # -------------------------------------------------------------------------

    def load_processed_image_ids(self) -> Set[int]:
        """Return image IDs already present in the output CSV."""
        processed_ids: Set[int] = set()

        if not self.config.output_csv.exists():
            return processed_ids

        try:
            with self.config.output_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    image_id = self._safe_int(row.get("image_id"))
                    if image_id is not None:
                        processed_ids.add(image_id)

            logger.info("Found %d already processed images", len(processed_ids))

        except Exception as exc:
            logger.warning("Could not read existing CSV: %s", exc)

        return processed_ids

    def load_failed_image_ids(self) -> Set[int]:
        """
        Return image IDs that should be retried.

        Retry criteria:
        - error_message contains KeyError
        - prediction coordinates are missing or NaN
        """
        failed_ids: Set[int] = set()

        if not self.config.output_csv.exists():
            return failed_ids

        try:
            with self.config.output_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    image_id = self._safe_int(row.get("image_id"))
                    if image_id is None:
                        continue

                    if self._row_should_be_retried(row):
                        failed_ids.add(image_id)

            if failed_ids:
                logger.info("Found %d failed images to retry", len(failed_ids))

        except Exception as exc:
            logger.warning("Could not read failed images from CSV: %s", exc)

        return failed_ids

    @staticmethod
    def _row_should_be_retried(row: Dict[str, str]) -> bool:
        error_message = row.get("error_message", "") or ""

        if "KeyError" in error_message:
            return True

        pred_x = row.get("pred_x_meters", "")
        pred_y = row.get("pred_y_meters", "")

        return (
            ImageProcessor._is_missing_number(pred_x)
            or ImageProcessor._is_missing_number(pred_y)
        )

    # -------------------------------------------------------------------------
    # Image discovery: manifest or extracted image_<id> fallback
    # -------------------------------------------------------------------------

    def load_image_manifest(self) -> Dict[str, Path]:
        """
        Load image paths from a CSV with columns:
            id, relative_image_path

        Example relative_image_path:
            /mapillary_dataset/175/175345001056338.jpg

        The configured prefix, by default 'mapillary_dataset/', is stripped,
        and the remaining path is resolved relative to images_dir_original.
        """
        image_paths: Dict[str, Path] = {}

        if self.config.image_manifest_csv is None:
            return image_paths

        if not self.config.image_manifest_csv.exists():
            raise FileNotFoundError(
                f"Image manifest CSV not found: {self.config.image_manifest_csv}"
            )

        base_dir = self.config.images_dir_original or self.config.images_dir

        logger.info("Loading image manifest from %s", self.config.image_manifest_csv)
        logger.info("Resolving manifest image paths relative to %s", base_dir)

        with self.config.image_manifest_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(f"Image manifest has no header row: {self.config.image_manifest_csv}")

            required_columns = {"id", "relative_image_path"}
            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"Image manifest is missing required columns {sorted(missing_columns)}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                dataset_id = str(row.get("id") or "").strip()
                relative_image_path = (row.get("relative_image_path") or "").strip()

                if not dataset_id or not relative_image_path:
                    continue

                normalized_path = self._normalize_manifest_relative_path(relative_image_path)
                image_paths[dataset_id] = base_dir / Path(normalized_path)

        logger.info("Loaded %d image paths from manifest", len(image_paths))
        return image_paths

    def _normalize_manifest_relative_path(self, relative_image_path: str) -> str:
        """Normalize a manifest path and strip the configured dataset prefix."""
        normalized = relative_image_path.replace("\\", "/").lstrip("/")

        prefix = self.config.manifest_prefix_to_strip.replace("\\", "/").strip("/")
        if prefix:
            prefix = prefix + "/"
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        return normalized

    def find_local_images(self, h5_metadata: Optional[Dict[int, GroundTruthMetadata]] = None,) -> Dict[int, Path]:
        """
        Find local images.

        If image_manifest_csv is provided, use:
            manifest id -> resolved image path

        Otherwise fall back to the older extracted-image layout:
            image_<id>.jpg/png/jpeg in --images-dir or --images-dir-original.
        """
        if self.config.image_manifest_csv is not None:
            manifest_paths_by_dataset_id = self.load_image_manifest()

            if h5_metadata is None:
                h5_metadata = self.load_h5_metadata()

            image_paths_by_h5_id: Dict[int, Path] = {}

            for image_id, metadata in h5_metadata.items():
                dataset_id = self._normalize_dataset_id(metadata.id_dataset)

                image_path = manifest_paths_by_dataset_id.get(dataset_id)

                if image_path is not None:
                    image_paths_by_h5_id[image_id] = image_path

            existing_image_paths = {
                image_id: image_path
                for image_id, image_path in image_paths_by_h5_id.items()
                if image_path.exists()
            }

            missing_count = len(image_paths_by_h5_id) - len(existing_image_paths)

            logger.info(
                "Matched %d H5 entries to manifest image paths",
                len(image_paths_by_h5_id),
            )
            logger.info(
                "Found %d existing images from manifest",
                len(existing_image_paths),
            )

            if missing_count:
                logger.warning(
                    "Matched manifest referenced %d images that do not exist on disk",
                    missing_count,
                )

            return existing_image_paths

        images_dir = self.config.images_dir_original or self.config.images_dir

        logger.info("Scanning for images in %s", images_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        patterns = [
            "image_*.jpg",
            "image_*.jpeg",
            "image_*.png",
            "image_*.JPG",
            "image_*.JPEG",
            "image_*.PNG",
        ]

        image_paths: Dict[int, Path] = {}

        for pattern in patterns:
            for image_path in images_dir.glob(pattern):
                try:
                    image_id = self.get_image_id(image_path)
                except ValueError:
                    logger.warning("Skipping image with invalid extracted-image filename: %s", image_path)
                    continue
                image_paths[image_id] = image_path

        logger.info("Found %d images", len(image_paths))

        if self.config.images_dir_original:
            logger.info("Using original images from %s", self.config.images_dir_original)

        return image_paths

    @staticmethod
    def get_image_id(image_path: Path) -> int:
        """Extract image id from filename like image_123.jpg."""
        stem = image_path.stem
        if not stem.startswith("image_"):
            raise ValueError(f"Expected filename like image_<id>.jpg, got: {image_path.name}")
        return int(stem.replace("image_", "", 1))

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------

    def process_all_images(
        self,
        max_images: Optional[int] = None,
        retry_failed: bool = True,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1.")

        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(
                f"worker_id must be between 0 and num_workers - 1. "
                f"Got worker_id={worker_id}, num_workers={num_workers}."
            )

        h5_metadata = self.load_h5_metadata()
        csv_priors = self.load_csv_priors()
        processed_ids = self.load_processed_image_ids()

        failed_ids: Set[int] = set()

        if retry_failed:
            failed_ids = self.load_failed_image_ids()
            processed_ids -= failed_ids

        local_images_by_id = self.find_local_images(h5_metadata=h5_metadata)

        before_h5_filter = len(local_images_by_id)
        local_images_by_id = {
            image_id: image_path
            for image_id, image_path in local_images_by_id.items()
            if image_id in h5_metadata
        }
        missing_h5_count = before_h5_filter - len(local_images_by_id)
        if missing_h5_count:
            logger.warning(
                "Skipped %d images because their IDs were not found in H5 metadata",
                missing_h5_count,
            )

        # Shard images by worker id.
        # Example with num_workers=2:
        #   worker 0 processes IDs where image_id % 2 == 0
        #   worker 1 processes IDs where image_id % 2 == 1
        local_images_by_id = {
            image_id: image_path
            for image_id, image_path in local_images_by_id.items()
            if image_id % num_workers == worker_id
        }

        images_to_process: list[Tuple[int, Path]] = [
            (image_id, image_path)
            for image_id, image_path in sorted(local_images_by_id.items())
            if image_id not in processed_ids
        ]

        skipped_count = len(local_images_by_id) - len(images_to_process)

        if max_images is not None:
            images_to_process = images_to_process[:max_images]

        logger.info(
            "Worker %d/%d processing %d images",
            worker_id,
            num_workers,
            len(images_to_process),
        )

        if skipped_count:
            logger.info("Skipped already processed images: %d", skipped_count)

        if failed_ids:
            logger.info("Retrying failed images: %d", len(failed_ids))

        for index, (image_id, image_path) in enumerate(images_to_process, start=1):
            is_retry = image_id in failed_ids

            logger.info(
                "[%d/%d] Processing id=%s path=%s%s",
                index,
                len(images_to_process),
                image_id,
                image_path,
                " [retry]" if is_retry else "",
            )

            result = self.process_single_image(
                image_id=image_id,
                image_path=image_path,
                h5_metadata=h5_metadata,
                csv_priors=csv_priors,
            )

            if is_retry:
                self._replace_csv_row(image_id, result)
            else:
                self._append_csv_row(result)

        logger.info(
            "Worker %d processing complete. Results saved to %s",
            worker_id,
            self.config.output_csv,
        )

    def process_single_image(
        self,
        image_id: int,
        image_path: Path,
        h5_metadata: Dict[int, GroundTruthMetadata],
        csv_priors: Dict[int, LocationPrior],
    ) -> Dict[str, Any]:
        """Process one image and return one CSV row."""
        result = self._empty_result_row(image_id, image_path)

        # Extract EXIF data from image.
        exif_data = self._extract_exif_metadata(image_path)
        result.update(exif_data)

        ground_truth = h5_metadata.get(image_id)

        if ground_truth is None:
            result["error_message"] = "No ground truth metadata found in H5"
        else:
            self._add_ground_truth_to_result(result, ground_truth)
            result["src_latitude"] = ground_truth.latitude
            result["src_longitude"] = ground_truth.longitude
            result["src_yaw"] = ground_truth.yaw

        # Resolve location prior based on strategy.
        location_prior, prior_source = self._resolve_location_prior(
            image_id,
            image_path,
            ground_truth,
            csv_priors,
        )

        if location_prior is not None:
            prior_latlon = (location_prior.latitude, location_prior.longitude)
            logger.info(
                "Image %s: Using location prior from %s (lat=%.6f, lon=%.6f)",
                image_id,
                prior_source,
                location_prior.latitude,
                location_prior.longitude,
            )
        else:
            prior_latlon = None
            logger.warning("Image %s: No location prior found", image_id)
            result["error_message"] = "No location prior found (no H5, CSV, or EXIF)"

        try:
            image, camera, gravity, proj, bbox = self.demo.read_input_image(
                str(image_path),
                prior_latlon=prior_latlon,
                tile_size_meters=self.config.tile_size_meters,
            )

            logger.info("Image %s: loading OSM canvas", image_id)
            canvas = self._load_osm_canvas(proj, bbox)

            logger.info(
                "Image %s: OSM canvas loaded, raster shape=%s dtype=%s min=%s max=%s",
                image_id,
                getattr(canvas.raster, "shape", None),
                getattr(canvas.raster, "dtype", None),
                np.nanmin(canvas.raster),
                np.nanmax(canvas.raster),
            )

            self._sanitize_raster(canvas.raster)

            logger.info(
                "Image %s: raster sanitized, min=%s max=%s unique-per-layer=%s",
                image_id,
                np.nanmin(canvas.raster),
                np.nanmax(canvas.raster),
                [
                    np.unique(canvas.raster[i]).tolist()[:20]
                    for i in range(min(canvas.raster.shape[0], len(RASTER_LAYER_MAX_VALUES)))
                ],
            )

            logger.info("Image %s: running OrienterNet localize", image_id)

            with torch.inference_mode():
                uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
                    image,
                    camera,
                    canvas,
                    gravity=gravity,
                )

            xy = canvas.to_xy(uv)
            latlon = proj.unproject(xy)

            result.update(
                {
                    "pred_x_meters": float(xy[0]),
                    "pred_y_meters": float(xy[1]),
                    "pred_latitude": float(latlon[0]),
                    "pred_longitude": float(latlon[1]),
                    "pred_yaw": float(yaw.detach().cpu().item()),
                    "pred_probability": float(prob.max().detach().cpu().item()),
                    "error_message": "",
                }
            )

            if self.config.save_artifacts:
                self._save_artifacts(
                    image_id=image_id,
                    neural_map=neural_map,
                    prob=prob,
                    image_rectified=image_rectified,
                    camera=camera,
                    raster=canvas.raster,
                    canvas=canvas,
                    predicted_uv=uv,
                )

            logger.info(
                "Image %s localized: lat=%.6f, lon=%.6f, yaw=%.2f, prob=%.4f",
                image_id,
                result["pred_latitude"],
                result["pred_longitude"],
                result["pred_yaw"],
                result["pred_probability"],
            )


        except Exception as exc:

            error_message = f"{type(exc).__name__}: {exc}"

            result["error_message"] = error_message

            logger.exception("Error processing image %s: %s", image_id, error_message)

        finally:
            self._cleanup_gpu_memory()

        return result

    # -------------------------------------------------------------------------
    # OrienterNet / OSM helpers
    # -------------------------------------------------------------------------

    def _load_osm_canvas(self, proj: Any, bbox: Any) -> Any:
        tiler = TileManager.from_bbox(
            proj,
            bbox + 10,
            self.demo.config.data.pixel_per_meter,
        )

        return tiler.query(bbox)

    @staticmethod
    def _sanitize_raster(raster: np.ndarray) -> None:
        """
        Clamp invalid raster layer values in-place.

        This avoids model failures caused by out-of-range OSM raster values.
        """
        if raster is None:
            return

        if raster.ndim < 3:
            logger.warning("Unexpected raster shape: %s", getattr(raster, "shape", None))
            return

        num_layers = min(raster.shape[0], len(RASTER_LAYER_MAX_VALUES))

        for layer_idx in range(num_layers):
            max_value = RASTER_LAYER_MAX_VALUES[layer_idx]
            layer = raster[layer_idx]

            invalid_mask = (
                    ~np.isfinite(layer)
                    | (layer < 0)
                    | (layer > max_value)
            )

            if invalid_mask.any():
                logger.warning(
                    "Clamping %d invalid values in raster layer %d. "
                    "Allowed range: [0, %d]. Before min=%s max=%s",
                    int(invalid_mask.sum()),
                    layer_idx,
                    max_value,
                    np.nanmin(layer),
                    np.nanmax(layer),
                )

                layer = np.nan_to_num(layer, nan=0, posinf=max_value, neginf=0)
                raster[layer_idx] = np.clip(layer, 0, max_value)

    # -------------------------------------------------------------------------
    # Artifact saving
    # -------------------------------------------------------------------------

    def _save_artifacts(
        self,
        image_id: int,
        neural_map: Any,
        prob: Any,
        image_rectified: Any,
        camera: Any,
        raster: np.ndarray,
        canvas: Any,
        predicted_uv: Any,
    ) -> None:
        """Save demo-style visual artifacts for one prediction.

        The RGB exports intentionally mirror the OrienterNet demo notebook:
        - OSM tile: ``Colormap.apply(canvas.raster)``
        - prediction map: ``likelihood_overlay(prob.max(-1), map_viz.mean(...))``
        - neural map: ``features_to_RGB(neural_map)``

        The raw tensors are still saved as ``.npy`` files for debugging.
        """
        folder = self.config.output_artifacts_dir / f"image_{image_id}"
        folder.mkdir(parents=True, exist_ok=True)

        prob_np = self._to_numpy(prob)
        np.save(folder / "prediction_map.npy", prob_np)

        neural_map_np = self._to_numpy(neural_map)
        np.save(folder / "neural_map.npy", neural_map_np)

        rectified_image = self._image_like_to_uint8(image_rectified)
        Image.fromarray(rectified_image).save(folder / "rectified_image.jpg")

        camera_dict = self._camera_to_dict(camera)
        with (folder / "camera.json").open("w", encoding="utf-8") as f:
            json.dump(camera_dict, f, indent=2)

        # Demo-equivalent OSM visualization. This is the categorical raster
        # rendered through OrienterNet's colormap, not a satellite/base-map tile.
        map_viz = Colormap.apply(raster)
        self._save_float_rgb(folder / "osm_tile.jpg", map_viz)

        # Demo-equivalent prediction map. The localization probability is defined
        # over x/y/rotation, so the notebook visualizes the best rotation per pixel.
        prediction_xy = prob_np.max(axis=-1) if prob_np.ndim >= 3 else prob_np
        np.save(folder / "prediction_map_xy.npy", prediction_xy)

        prediction_overlay = likelihood_overlay(
            prediction_xy,
            map_viz.mean(axis=-1, keepdims=True),
        )
        self._save_float_rgb(folder / "prediction_map.jpg", prediction_overlay)

        # A grayscale probability-only export is useful to check whether the
        # model produced a non-zero spatial likelihood away from the overlay.
        probability_rgb = self._probability_to_uint8_rgb(prediction_xy)
        Image.fromarray(probability_rgb).save(folder / "prediction_probability.jpg")

        neural_map_rgb = self._neural_map_to_rgb(neural_map_np)
        Image.fromarray(neural_map_rgb).save(folder / "neural_map_rgb.jpg")

        # Save the predicted pixel in map/image coordinates for easier inspection.
        predicted_uv_np = self._to_numpy(predicted_uv).reshape(-1).tolist()
        artifact_meta = {
            "predicted_uv": self._json_safe(predicted_uv),
            "canvas_bbox": self._json_safe(canvas.bbox) if hasattr(canvas, "bbox") else None,
            "files": {
                "rectified_image": "rectified_image.jpg",
                "osm_tile": "osm_tile.jpg",
                "neural_map_rgb": "neural_map_rgb.jpg",
                "prediction_map": "prediction_map.jpg",
                "prediction_probability": "prediction_probability.jpg",
                "prediction_map_raw": "prediction_map.npy",
                "prediction_map_xy_raw": "prediction_map_xy.npy",
                "neural_map_raw": "neural_map.npy",
            },
        }

        with (folder / "artifacts.json").open("w", encoding="utf-8") as f:
            json.dump(artifact_meta, f, indent=2, allow_nan=False)

        logger.info("Artifacts saved for image %d to %s", image_id, folder)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert torch tensors or array-like values to a CPU numpy array."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _save_float_rgb(path: Path, image: np.ndarray) -> None:
        """Save an RGB image represented either as float [0, 1] or uint8."""
        image_np = np.asarray(image)
        if image_np.dtype == np.uint8:
            out = image_np
        else:
            out = np.clip(image_np, 0.0, 1.0)
            out = (out * 255).astype(np.uint8)
        Image.fromarray(out).save(path)

    @staticmethod
    def _image_like_to_uint8(image_like: Any) -> np.ndarray:
        """Convert OrienterNet image tensors/arrays to an HxWxC uint8 image."""
        image_np = ImageProcessor._to_numpy(image_like)

        # Drop a singleton batch dimension if present.
        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np[0]

        # Convert CHW to HWC when needed.
        if image_np.ndim == 3 and image_np.shape[0] in (1, 3, 4):
            image_np = np.moveaxis(image_np, 0, -1)

        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=-1)

        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)

        if image_np.dtype == np.uint8:
            return image_np

        # Most OrienterNet/demo tensors are already normalized to [0, 1].
        # If an array appears to be in [0, 255], avoid scaling it a second time.
        finite = image_np[np.isfinite(image_np)]
        max_value = float(finite.max()) if finite.size else 1.0
        if max_value > 2.0:
            return np.clip(image_np, 0, 255).astype(np.uint8)

        return np.clip(image_np * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _neural_map_to_rgb(neural_map_np: np.ndarray) -> np.ndarray:
        """Convert neural-map features to RGB using the demo helper."""
        # The demo helper returns a tuple/list, even for one feature map.
        rgb_items = features_to_RGB(neural_map_np)
        neural_map_rgb = rgb_items[0] if isinstance(rgb_items, (tuple, list)) else rgb_items
        return ImageProcessor._image_like_to_uint8(neural_map_rgb)

    @staticmethod
    def _probability_to_uint8_rgb(probability_xy: np.ndarray) -> np.ndarray:
        """Save a simple probability-only heat image without applying matplotlib."""
        prob = np.asarray(probability_xy, dtype=np.float32)
        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)

        min_value = float(prob.min()) if prob.size else 0.0
        max_value = float(prob.max()) if prob.size else 0.0

        if max_value > min_value:
            prob = (prob - min_value) / (max_value - min_value)
        else:
            prob = np.zeros_like(prob)

        gray = np.clip(prob * 255, 0, 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)

    @staticmethod
    def _camera_to_dict(camera: Any) -> Dict[str, Any]:
        return {
            "width": int(camera.size[0].item()),
            "height": int(camera.size[1].item()),
            "fx": float(camera.f[0].item()),
            "fy": float(camera.f[1].item()),
            "cx": float(camera.c[0].item()),
            "cy": float(camera.c[1].item()),
            "dist": camera.dist.detach().cpu().tolist(),
        }

    @staticmethod
    def _extract_exif_metadata(image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image file."""
        exif_data = {
            "exif_make": "",
            "exif_model": "",
            "exif_focal_35mm": np.nan,
            "exif_focal_ratio": np.nan,
            "exif_orientation": np.nan,
            "exif_altitude": np.nan,
        }

        try:
            with open(image_path, "rb") as fid:
                exif = EXIF(fid, lambda: None)

                exif_data["exif_make"] = exif.extract_make()
                exif_data["exif_model"] = exif.extract_model()

                focal_35mm, focal_ratio = exif.extract_focal()
                exif_data["exif_focal_35mm"] = (
                    float(focal_35mm) if focal_35mm else np.nan
                )
                exif_data["exif_focal_ratio"] = (
                    float(focal_ratio) if focal_ratio else np.nan
                )

                orientation = exif.extract_orientation()
                exif_data["exif_orientation"] = (
                    float(orientation) if orientation is not None else np.nan
                )

                altitude = exif.extract_altitude()
                exif_data["exif_altitude"] = float(altitude) if altitude else np.nan

        except Exception as exc:
            logger.warning("Failed to extract EXIF data from %s: %s", image_path, exc)

        return exif_data

    def _extract_location_prior_from_exif(self, image_path: Path) -> Optional[LocationPrior]:
        """Try to extract location prior from EXIF GPS data."""
        try:
            with open(image_path, "rb") as fid:
                exif = EXIF(fid, lambda: None)
                geo = exif.extract_geo()

                if geo and "latitude" in geo and "longitude" in geo:
                    return LocationPrior(
                        latitude=float(geo["latitude"]),
                        longitude=float(geo["longitude"]),
                        yaw=None,
                        source="exif",
                    )
        except Exception as exc:
            logger.debug("Could not extract EXIF location from %s: %s", image_path, exc)

        return None

    def _resolve_location_prior(
        self,
        image_id: int,
        image_path: Path,
        h5_metadata: Optional[GroundTruthMetadata],
        csv_priors: Dict[int, LocationPrior],
    ) -> Tuple[Optional[LocationPrior], str]:
        """
        Resolve location prior based on strategy.

        Returns:
            Tuple of (LocationPrior, source_description)
            source_description: "csv", "exif", "h5", or "none"
        """
        strategy = self.config.prior_strategy.lower()

        if strategy in ("csv", "fallback") and image_id in csv_priors:
            return csv_priors[image_id], "csv"

        if strategy in ("exif", "fallback"):
            exif_prior = self._extract_location_prior_from_exif(image_path)
            if exif_prior is not None:
                return exif_prior, "exif"

        if h5_metadata is not None:
            prior = LocationPrior(
                latitude=h5_metadata.latitude,
                longitude=h5_metadata.longitude,
                yaw=h5_metadata.yaw,
                source="h5",
            )
            return prior, "h5"

        return None, "none"

    # -------------------------------------------------------------------------
    # Result row helpers
    # -------------------------------------------------------------------------

    def _empty_result_row(self, image_id: int, image_path: Path) -> Dict[str, Any]:
        try:
            relative_path = image_path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_path = image_path

        return {
            "image_id": image_id,
            "image_path": str(relative_path),
            "h5_id_dataset": "",
            "gt_latitude": np.nan,
            "gt_longitude": np.nan,
            "gt_yaw": np.nan,
            "src_latitude": np.nan,
            "src_longitude": np.nan,
            "src_yaw": np.nan,
            "pred_latitude": np.nan,
            "pred_longitude": np.nan,
            "pred_x_meters": np.nan,
            "pred_y_meters": np.nan,
            "pred_yaw": np.nan,
            "pred_probability": np.nan,
            "exif_make": "",
            "exif_model": "",
            "exif_focal_35mm": np.nan,
            "exif_focal_ratio": np.nan,
            "exif_orientation": np.nan,
            "exif_altitude": np.nan,
            "error_message": "",
        }

    @staticmethod
    def _add_ground_truth_to_result(
        result: Dict[str, Any],
        ground_truth: GroundTruthMetadata,
    ) -> None:
        result.update(
            {
                "h5_id_dataset": ground_truth.id_dataset,
                "gt_latitude": ground_truth.latitude,
                "gt_longitude": ground_truth.longitude,
                "gt_yaw": ground_truth.yaw,
            }
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def print_summary_stats(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            logger.info("pandas not available. Cannot print summary statistics.")
            return

        if not self.config.output_csv.exists():
            logger.info("No CSV file found at %s", self.config.output_csv)
            return

        df = pd.read_csv(self.config.output_csv)

        successful = df["pred_x_meters"].notna() if "pred_x_meters" in df else []
        has_ground_truth = df["gt_latitude"].notna() if "gt_latitude" in df else []

        logger.info("")
        logger.info("=== SUMMARY STATISTICS ===")
        logger.info("Total rows: %d", len(df))
        logger.info("Successful predictions: %d", int(successful.sum()) if len(df) else 0)
        logger.info("Failed predictions: %d", int((~successful).sum()) if len(df) else 0)
        logger.info("Rows with ground truth: %d", int(has_ground_truth.sum()) if len(df) else 0)

        if "error_message" in df.columns:
            failed_with_error = df[df["error_message"].fillna("") != ""]
            logger.info("Rows with error message: %d", len(failed_with_error))

    # -------------------------------------------------------------------------
    # Small utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _decode_h5_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _normalize_dataset_id(value: Any) -> str:
        """
        Normalize dataset IDs for matching.

        Example:
            mapillary:475541626839824 -> 475541626839824
            475541626839824           -> 475541626839824
        """
        text = ImageProcessor._decode_h5_value(value).strip()

        if ":" in text:
            text = text.split(":", 1)[1]

        return text.strip()

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _is_missing_number(value: Any) -> bool:
        if value is None:
            return True

        text = str(value).strip().lower()

        if text in {"", "nan", "none", "null"}:
            return True

        try:
            return bool(np.isnan(float(text)))
        except Exception:
            return True


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process images through OrienterNet and save localization results to CSV.",
    )

    parser.add_argument(
        "--h5",
        default="data/bern_ground_all.h5",
        help="Path to H5 file with metadata.",
    )

    parser.add_argument(
        "--images-dir",
        default="data/extracted_images",
        help="Directory containing extracted images named image_<id>.*.",
    )

    parser.add_argument(
        "--images-dir-original",
        default=None,
        help=(
            "Directory containing original nested images. "
            "Use together with --image-manifest-csv when filenames are not image_<id>.*."
        ),
    )

    parser.add_argument(
        "--image-manifest-csv",
        default=None,
        help=(
            "CSV with columns 'id' and 'relative_image_path'. The id is matched "
            "to H5 metadata/id. The relative path is resolved under --images-dir-original."
        ),
    )

    parser.add_argument(
        "--manifest-prefix-to-strip",
        default=DEFAULT_MANIFEST_PREFIX_TO_STRIP,
        help=(
            "Path prefix to strip from relative_image_path before resolving under "
            "--images-dir-original. Default: mapillary_dataset/"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=(
            "Output CSV file path. If left as the default, the file is written "
            "inside the generated run directory as results.csv."
        ),
    )

    parser.add_argument(
        "--output-artifacts-dir",
        default=DEFAULT_OUTPUT_ARTIFACTS_DIR,
        help=(
            "Directory for per-image output artifacts. If left as the default, "
            "artifacts are written inside the generated run directory."
        ),
    )

    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="Base directory for named runs. Default: data/runs",
    )

    parser.add_argument(
        "--data-version",
        default="h5v",
        help=(
            "Short version label for the H5/data source. Used in automatic "
            "run names, e.g. h5v or og."
        ),
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional custom run name. If omitted, a name is generated from "
            "tile size, rotations, and data version, e.g. process_272t_256r_h5v."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Tile size in meters.",
    )

    parser.add_argument(
        "--num-rotations",
        type=int,
        default=128,
        help="Number of rotation hypotheses.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process. Default: all.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use, e.g. cuda, cuda:0, cuda:1, or cpu.",
    )

    parser.add_argument(
        "--prior-strategy",
        default="h5",
        choices=["h5", "csv", "exif", "fallback"],
        help=(
            "Location prior strategy: "
            "h5 - Use H5 metadata only; "
            "csv - Use CSV prior if available, else H5; "
            "exif - Use EXIF GPS if available, else H5; "
            "fallback - Try CSV first, then EXIF, then H5. "
            "Default: h5"
        ),
    )

    parser.add_argument(
        "--csv-prior",
        default=None,
        help=(
            "Optional CSV file with location priors: image_id, latitude, longitude, yaw. "
            "Used with --prior-strategy csv or fallback."
        ),
    )

    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker index, from 0 to num-workers - 1.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Total number of worker shards. "
            "This does not spawn processes; launch one process per worker manually."
        ),
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after processing.",
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=True,
        help="Retry images with KeyError or NaN predictions. Default: enabled.",
    )

    parser.add_argument(
        "--no-retry-failed",
        action="store_false",
        dest="retry_failed",
        help="Do not retry failed images.",
    )

    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help=(
            "Save neural maps, probability maps, rectified image, camera JSON, "
            "and OSM image. Disabled by default to save memory and disk space."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ProcessorConfig:
    run_name = args.run_name or build_run_name(
        tile_size=args.tile_size,
        num_rotations=args.num_rotations,
        data_version=args.data_version,
    )

    run_dir = Path(args.runs_dir) / run_name

    output_csv = Path(args.output_csv)
    output_artifacts_dir = Path(args.output_artifacts_dir)

    # If the user left output paths at the defaults, place outputs inside
    # the parameter-based run folder. If the user explicitly passed custom
    # paths, respect those paths exactly.
    if args.output_csv == DEFAULT_OUTPUT_CSV:
        output_csv = run_dir / "results.csv"

    if args.output_artifacts_dir == DEFAULT_OUTPUT_ARTIFACTS_DIR:
        output_artifacts_dir = run_dir / "artifacts"

    logger.info("Run name: %s", run_name)
    logger.info("Run directory: %s", run_dir)
    logger.info("Output CSV: %s", output_csv)
    logger.info("Output artifacts directory: %s", output_artifacts_dir)

    return ProcessorConfig(
        h5_path=Path(args.h5),
        images_dir=Path(args.images_dir),
        output_csv=output_csv,
        output_artifacts_dir=output_artifacts_dir,
        tile_size_meters=args.tile_size,
        num_rotations=args.num_rotations,
        device=args.device,
        save_artifacts=args.save_artifacts,
        images_dir_original=(
            Path(args.images_dir_original) if args.images_dir_original else None
        ),
        image_manifest_csv=(
            Path(args.image_manifest_csv) if args.image_manifest_csv else None
        ),
        manifest_prefix_to_strip=args.manifest_prefix_to_strip,
        prior_strategy=args.prior_strategy,
        csv_prior_path=Path(args.csv_prior) if args.csv_prior else None,
    )


def main() -> None:
    configure_pytorch_memory()

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("maploc exists: %s", (PROJECT_ROOT / "maploc").exists())
    logger.info("Python executable: %s", sys.executable)

    config = build_config(args)

    try:
        processor = ImageProcessor(config)

        processor.process_all_images(
            max_images=args.max_images,
            retry_failed=args.retry_failed,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
        )

        if args.stats:
            processor.print_summary_stats()

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        sys.exit(1)

    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
