#!/usr/bin/env python3

"""
Run OrienterNet localization on extracted images and export predictions to CSV.

Expected project structure:

    FG-OrienterNet/
    ├── maploc/
    ├── processing/
    │   └── batch_process.py
    └── data/
        ├── bern_ground_all.h5
        └── extracted_images/
            ├── image_0.jpg
            ├── image_1.jpg
            └── ...

Expected image names:
    image_<id>.jpg
    image_<id>.png

Expected H5 metadata structure:
    metadata/id
    metadata/id_dataset
    metadata/latitude
    metadata/longitude
    metadata/yaw
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
# Imports
# =============================================================================

import argparse
import csv
import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import h5py
import numpy as np
import torch
from PIL import Image

from maploc.demo import Demo
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


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


configure_pytorch_memory()


# =============================================================================
# Constants
# =============================================================================

CSV_HEADERS = [
    "image_id",
    "image_path",
    "h5_id_dataset",
    "gt_latitude",
    "gt_longitude",
    "gt_yaw",
    "pred_latitude",
    "pred_longitude",
    "pred_x_meters",
    "pred_y_meters",
    "pred_yaw",
    "pred_probability",
    "error_message",
]

# OrienterNet OSM raster layer limits.
RASTER_LAYER_MAX_VALUES = [7, 10, 33]


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
class ProcessorConfig:
    h5_path: Path
    images_dir: Path
    output_csv: Path
    output_artifacts_dir: Path
    tile_size_meters: int = 64
    num_rotations: int = 128
    device: str = "cuda"
    save_artifacts: bool = False


# =============================================================================
# Main processor
# =============================================================================

class ImageProcessor:
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
    # Image discovery
    # -------------------------------------------------------------------------

    def find_local_images(self) -> list[Path]:
        """Find local image files matching image_*.jpg or image_*.png."""
        logger.info("Scanning for images in %s", self.config.images_dir)

        if not self.config.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.config.images_dir}")

        jpg_images = sorted(self.config.images_dir.glob("image_*.jpg"))
        png_images = sorted(self.config.images_dir.glob("image_*.png"))

        images = jpg_images + png_images

        logger.info("Found %d images", len(images))
        return images

    @staticmethod
    def get_image_id(image_path: Path) -> int:
        """Extract image id from filename like image_123.jpg."""
        return int(image_path.stem.replace("image_", ""))

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
        processed_ids = self.load_processed_image_ids()

        failed_ids: Set[int] = set()

        if retry_failed:
            failed_ids = self.load_failed_image_ids()
            processed_ids -= failed_ids

        local_images = self.find_local_images()

        # Shard images by worker id.
        # Example with num_workers=2:
        #   worker 0 processes IDs where image_id % 2 == 0
        #   worker 1 processes IDs where image_id % 2 == 1
        local_images = [
            image_path
            for image_path in local_images
            if self.get_image_id(image_path) % num_workers == worker_id
        ]

        images_to_process = [
            image_path
            for image_path in local_images
            if self.get_image_id(image_path) not in processed_ids
        ]

        skipped_count = len(local_images) - len(images_to_process)

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

        for index, image_path in enumerate(images_to_process, start=1):
            image_id = self.get_image_id(image_path)
            is_retry = image_id in failed_ids

            logger.info(
                "[%d/%d] Processing %s%s",
                index,
                len(images_to_process),
                image_path.name,
                " [retry]" if is_retry else "",
            )

            result = self.process_single_image(
                image_id=image_id,
                image_path=image_path,
                h5_metadata=h5_metadata,
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
    ) -> Dict[str, Any]:
        """Process one image and return one CSV row."""
        result = self._empty_result_row(image_id, image_path)

        ground_truth = h5_metadata.get(image_id)

        if ground_truth is None:
            result["error_message"] = "No ground truth metadata found in H5"
            prior_latlon = None
        else:
            self._add_ground_truth_to_result(result, ground_truth)
            prior_latlon = (ground_truth.latitude, ground_truth.longitude)

        try:
            image, camera, gravity, proj, bbox = self.demo.read_input_image(
                str(image_path),
                prior_latlon=prior_latlon,
                tile_size_meters=self.config.tile_size_meters,
            )

            canvas = self._load_osm_canvas(proj, bbox)
            self._sanitize_raster(canvas.raster)

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

            logger.error("Error processing image %s: %s", image_id, error_message)
            logger.debug(traceback.format_exc())

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
        for layer_idx, max_value in enumerate(RASTER_LAYER_MAX_VALUES):
            invalid_mask = raster[layer_idx] > max_value

            if invalid_mask.any():
                logger.warning(
                    "Clamping %d invalid values in raster layer %d. Max allowed: %d",
                    int(invalid_mask.sum()),
                    layer_idx,
                    max_value,
                )

                raster[layer_idx] = np.clip(raster[layer_idx], 0, max_value)

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
    ) -> None:
        folder = self.config.output_artifacts_dir / f"image_{image_id}"
        folder.mkdir(parents=True, exist_ok=True)

        np.save(folder / "neural_map.npy", neural_map.detach().cpu().numpy())
        np.save(folder / "prediction_prob.npy", prob.detach().cpu().numpy())

        rectified_image = self._tensor_to_uint8_image(image_rectified)
        Image.fromarray(rectified_image).save(folder / "rectified_image.png")

        camera_dict = self._camera_to_dict(camera)
        with (folder / "camera.json").open("w", encoding="utf-8") as f:
            json.dump(camera_dict, f, indent=2)

        rgb = Colormap.apply(raster)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(folder / "osm.png")

    @staticmethod
    def _tensor_to_uint8_image(image_tensor: Any) -> np.ndarray:
        image = image_tensor.detach().cpu()

        if image.ndim == 3:
            image = image.permute(1, 2, 0)

        image_np = image.numpy()
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        return image_np

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

    # -------------------------------------------------------------------------
    # Result row helpers
    # -------------------------------------------------------------------------

    def _empty_result_row(self, image_id: int, image_path: Path) -> Dict[str, Any]:
        try:
            relative_path = image_path.relative_to(self.config.images_dir.parent)
        except ValueError:
            relative_path = image_path

        return {
            "image_id": image_id,
            "image_path": str(relative_path),
            "h5_id_dataset": "",
            "gt_latitude": np.nan,
            "gt_longitude": np.nan,
            "gt_yaw": np.nan,
            "pred_latitude": np.nan,
            "pred_longitude": np.nan,
            "pred_x_meters": np.nan,
            "pred_y_meters": np.nan,
            "pred_yaw": np.nan,
            "pred_probability": np.nan,
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

        successful = df["pred_x_meters"].notna()
        has_ground_truth = df["gt_latitude"].notna()

        logger.info("")
        logger.info("=== SUMMARY STATISTICS ===")
        logger.info("Total rows: %d", len(df))
        logger.info("Successful predictions: %d", int(successful.sum()))
        logger.info("Failed predictions: %d", int((~successful).sum()))
        logger.info("Rows with ground truth: %d", int(has_ground_truth.sum()))

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
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
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
        help="Directory containing local images.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/orienternet_results.csv",
        help="Output CSV file path.",
    )

    parser.add_argument(
        "--output-artifacts-dir",
        default="data/image_outputs",
        help="Directory for per-image output artifacts.",
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
    return ProcessorConfig(
        h5_path=Path(args.h5),
        images_dir=Path(args.images_dir),
        output_csv=Path(args.output_csv),
        output_artifacts_dir=Path(args.output_artifacts_dir),
        tile_size_meters=args.tile_size,
        num_rotations=args.num_rotations,
        device=args.device,
        save_artifacts=args.save_artifacts,
    )


def main() -> None:
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