#!/usr/bin/env python3
"""
Process images from extracted local files and run through OrienterNet pipeline.
Maps images to H5 metadata and saves results to CSV incrementally.
"""

import argparse
import csv
import h5py
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from maploc.demo import Demo
from maploc.osm.tiling import TileManager
from maploc.utils.geo import BoundaryBox

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images through OrienterNet and save results to CSV."""

    def __init__(
        self,
        h5_path: str,
        images_dir: str,
        output_csv: str,
        tile_size_meters: int = 64,
        num_rotations: int = 256,
    ):
        """
        Initialize processor.

        Args:
            h5_path: Path to H5 file with metadata
            images_dir: Directory containing local images
            output_csv: Path to output CSV file
            tile_size_meters: Tile size for map queries
            num_rotations: Number of rotation hypotheses
        """
        self.h5_path = h5_path
        self.images_dir = Path(images_dir)
        self.output_csv = output_csv
        self.tile_size_meters = tile_size_meters
        self.num_rotations = num_rotations

        # Load demo model
        logger.info("Loading OrienterNet model...")
        self.demo = Demo(num_rotations=num_rotations)
        logger.info("Model loaded successfully")

        # CSV headers
        self.csv_headers = [
            'image_id',
            'image_path',
            'h5_id_dataset',
            'gt_latitude',
            'gt_longitude',
            'gt_yaw',
            'pred_latitude',
            'pred_longitude',
            'pred_x_meters',
            'pred_y_meters',
            'pred_yaw',
            'error_message',
        ]

        # Initialize CSV file if it doesn't exist
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not os.path.exists(self.output_csv):
            logger.info(f"Creating new CSV file: {self.output_csv}")
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()

    def _append_to_csv(self, row: Dict):
        """Append a single row to the CSV file."""
        try:
            with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")

    def _update_csv_row(self, image_id: int, new_row: Dict):
        """Update an existing CSV row by image_id."""
        try:
            # Read all rows
            rows = []
            with open(self.output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get('image_id', -1)) == image_id:
                        # Replace this row with the new data
                        rows.append(new_row)
                    else:
                        rows.append(row)

            # Write all rows back
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            logger.error(f"Failed to update CSV row for image {image_id}: {e}")

    def _load_h5_metadata(self) -> Dict:
        """Load all metadata from H5 file indexed by image ID."""
        logger.info(f"Loading metadata from {self.h5_path}...")
        metadata = {}

        with h5py.File(self.h5_path, 'r') as f:
            h5_ids = f['metadata']['id'][:]
            id_datasets = f['metadata']['id_dataset'][:]
            latitudes = f['metadata']['latitude'][:]
            longitudes = f['metadata']['longitude'][:]
            yaws = f['metadata']['yaw'][:]

            for idx, h5_id in enumerate(h5_ids):
                metadata[int(h5_id)] = {
                    'id_dataset': id_datasets[idx].decode('utf-8') if isinstance(id_datasets[idx], bytes) else id_datasets[idx],
                    'latitude': float(latitudes[idx]),
                    'longitude': float(longitudes[idx]),
                    'yaw': float(yaws[idx]),
                }

        logger.info(f"Loaded metadata for {len(metadata)} images from H5")
        return metadata

    def _load_processed_image_ids(self) -> set:
        """Load image IDs already processed from results CSV file."""
        processed_ids = set()

        if not os.path.exists(self.output_csv):
            return processed_ids

        try:
            with open(self.output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('image_id'):
                        processed_ids.add(int(row['image_id']))
            logger.info(f"Found {len(processed_ids)} already processed images in {self.output_csv}")
        except Exception as e:
            logger.warning(f"Could not read existing CSV file: {e}")

        return processed_ids

    def _get_failed_image_ids(self) -> set:
        """Identify image IDs with errors (KeyError or NaN predictions) for retry."""
        failed_ids = set()

        if not os.path.exists(self.output_csv):
            return failed_ids

        try:
            with open(self.output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get('image_id'):
                        continue

                    image_id = int(row['image_id'])
                    error_msg = row.get('error_message', '')

                    # Retry if KeyError in message or if prediction values are NaN
                    if 'KeyError' in error_msg:
                        failed_ids.add(image_id)
                        logger.debug(f"Marking image {image_id} for retry due to KeyError: {error_msg}")
                    elif not error_msg or error_msg == '':
                        # Check if prediction values are NaN (successful computation)
                        try:
                            pred_x = row.get('pred_x_meters', '')
                            pred_y = row.get('pred_y_meters', '')
                            if pred_x in ('nan', '') or pred_y in ('nan', ''):
                                # Try to parse - if it's "nan" string, mark for retry
                                if pred_x in ('nan', '') and pred_y in ('nan', ''):
                                    failed_ids.add(image_id)
                                    logger.debug(f"Marking image {image_id} for retry due to NaN predictions")
                        except Exception:
                            pass

            if failed_ids:
                logger.info(f"Found {len(failed_ids)} failed images to retry (KeyError or NaN)")
        except Exception as e:
            logger.warning(f"Could not read failed images from CSV: {e}")

        return failed_ids

    def _find_local_image(self, image_id: int) -> Optional[Path]:
        """Find local image file for given image ID."""
        # Try different naming patterns
        patterns = [
            f"image_{image_id}.jpg",
            f"image_{image_id}.png",
            f"{image_id}.jpg",
            f"{image_id}.png",
        ]

        for pattern in patterns:
            image_path = self.images_dir / pattern
            if image_path.exists():
                return image_path

        return None

    def _process_single_image(
        self,
        image_id: int,
        image_path: Path,
        h5_metadata: Dict,
    ) -> Dict:
        """
        Process a single image through the pipeline.

        Returns:
            Dictionary with results or error message
        """
        result = {
            'image_id': image_id,
            'image_path': str(image_path.relative_to(self.images_dir.parent)),
            'h5_id_dataset': '',
            'gt_latitude': np.nan,
            'gt_longitude': np.nan,
            'gt_yaw': np.nan,
            'pred_latitude': np.nan,
            'pred_longitude': np.nan,
            'pred_x_meters': np.nan,
            'pred_y_meters': np.nan,
            'pred_yaw': np.nan,
            'error_message': '',
        }

        # Add ground truth from H5 if available
        if image_id in h5_metadata:
            result['h5_id_dataset'] = h5_metadata[image_id]['id_dataset']
            result['gt_latitude'] = h5_metadata[image_id]['latitude']
            result['gt_longitude'] = h5_metadata[image_id]['longitude']
            result['gt_yaw'] = h5_metadata[image_id]['yaw']
            prior_latlon = (
                h5_metadata[image_id]['latitude'],
                h5_metadata[image_id]['longitude']
            )
        else:
            result['error_message'] = "No ground truth metadata found in H5"
            prior_latlon = None

        try:
            # Read and calibrate input image
            logger.info(f"Processing image {image_id}: {image_path.name}")
            image, camera, gravity, proj, bbox = self.demo.read_input_image(
                str(image_path),
                prior_latlon=prior_latlon,
                tile_size_meters=self.tile_size_meters,
            )

            # Query OSM map tile
            tiler = TileManager.from_bbox(
                proj,
                bbox + 10,
                self.demo.config.data.pixel_per_meter
            )
            canvas = tiler.query(bbox)

            # Debug: Check raster values
            raster = canvas.raster
            logger.debug(f"Raster shape: {raster.shape}, dtype: {raster.dtype}")
            logger.debug(f"Raster value ranges: areas={raster[0].min()}-{raster[0].max()}, ways={raster[1].min()}-{raster[1].max()}, nodes={raster[2].min()}-{raster[2].max()}")

            # Clamp raster values to valid ranges to prevent KeyError
            # areas: 0-7, ways: 0-10, nodes: 0-33
            max_values = [7, 10, 33]
            for i, max_val in enumerate(max_values):
                invalid_mask = raster[i] > max_val
                if invalid_mask.any():
                    logger.warning(f"Clamping {invalid_mask.sum()} invalid values in layer {i} (max allowed: {max_val})")
                    raster[i] = np.clip(raster[i], 0, max_val)

            # Run localization
            uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
                image, camera, canvas, gravity=gravity
            )

            # Convert UV to XY coordinates
            xy = canvas.to_xy(uv)  # This gives projected coordinates (x, y in meters)
            latlon = proj.unproject(xy)  # This gives geographic coordinates (lat, lon)

            # Store both projected and geographic coordinates
            result['pred_x_meters'] = float(xy[0])
            result['pred_y_meters'] = float(xy[1])
            result['pred_latitude'] = float(latlon[0])
            result['pred_longitude'] = float(latlon[1])
            result['pred_yaw'] = float(yaw.numpy())

            logger.info(
                f"Successfully processed image {image_id}: "
                f"xy_meters=({result['pred_x_meters']:.2f}, {result['pred_y_meters']:.2f}), "
                f"latlon=({result['pred_latitude']:.6f}, {result['pred_longitude']:.6f}), "
                f"yaw={result['pred_yaw']:.2f}°"
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result['error_message'] = error_msg
            logger.error(f"Error processing image {image_id}: {error_msg}")
            logger.debug(traceback.format_exc())

        return result

    def process_all_images(self, max_images: Optional[int] = None, retry_failed: bool = True):
        """
        Process all images and save results.

        Args:
            max_images: Maximum number of images to process (None for all)
            retry_failed: Whether to retry images with KeyError or NaN predictions
        """
        # Load H5 metadata
        h5_metadata = self._load_h5_metadata()

        # Load already processed image IDs
        processed_ids = self._load_processed_image_ids()

        # Get failed image IDs if retry is enabled
        failed_ids = set()
        if retry_failed:
            failed_ids = self._get_failed_image_ids()
            # Remove failed images from processed set so they get reprocessed
            processed_ids = processed_ids - failed_ids

        # Find all local images
        logger.info(f"Scanning for local images in {self.images_dir}...")
        local_images = sorted(self.images_dir.glob("image_*.jpg"))
        if not local_images:
            local_images = sorted(self.images_dir.glob("image_*.png"))

        logger.info(f"Found {len(local_images)} local images")

        # Filter out already processed images
        images_to_process = []
        for image_path in local_images:
            image_id = int(image_path.stem.replace('image_', ''))
            if image_id not in processed_ids:
                images_to_process.append(image_path)

        skipped_count = len(local_images) - len(images_to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed images")

        logger.info(f"Processing {len(images_to_process)} new images")
        if failed_ids:
            logger.info(f"  Including {len(failed_ids)} failed images for retry")

        if max_images is not None:
            images_to_process = images_to_process[:max_images]

        # Process each image
        for idx, image_path in enumerate(images_to_process, 1):
            # Extract image ID from filename
            image_id = int(image_path.stem.replace('image_', ''))

            is_retry = image_id in failed_ids
            retry_label = " (RETRY)" if is_retry else ""
            logger.info(f"\n[{idx}/{len(images_to_process)}] Processing {image_path.name}{retry_label}...")

            # Process image
            result = self._process_single_image(image_id, image_path, h5_metadata)

            # For retries, update the existing row instead of appending
            if is_retry:
                self._update_csv_row(image_id, result)
                logger.info(f"Updated result for retry image {image_id} in {self.output_csv}")
            else:
                # Save to CSV (incremental)
                self._append_to_csv(result)
                logger.info(f"Saved result to {self.output_csv}")

        logger.info(f"\nProcessing complete! Results saved to {self.output_csv}")
        if skipped_count > 0:
            logger.info(f"Summary: {len(images_to_process)} new images processed, {skipped_count} images skipped")
        if failed_ids:
            logger.info(f"         {len(failed_ids)} failed images were retried")

    def get_summary_stats(self):
        """Print summary statistics from CSV file."""
        try:
            import pandas as pd

            df = pd.read_csv(self.output_csv)

            logger.info("\n=== SUMMARY STATISTICS ===")
            logger.info(f"Total images processed: {len(df)}")
            logger.info(f"Successful predictions: {(~df['pred_x_meters'].isna()).sum()}")
            logger.info(f"Failed predictions: {(df['pred_x_meters'].isna()).sum()}")

            if (~df['pred_x_meters'].isna()).sum() > 0:
                # Compute errors if ground truth is available
                valid_gt = ~df['gt_latitude'].isna()
                if valid_gt.sum() > 0:
                    # Simple distance calculation (not accurate, for reference only)
                    logger.info(f"Images with ground truth: {valid_gt.sum()}")
        except ImportError:
            logger.info("pandas not available for summary stats")


def main():
    parser = argparse.ArgumentParser(
        description="Process images through OrienterNet and save results to CSV"
    )
    parser.add_argument(
        "--h5",
        default="../data/bern_ground_all.h5",
        help="Path to H5 file with metadata"
    )
    parser.add_argument(
        "--images-dir",
        default="../data/extracted_images",
        help="Directory containing local images"
    )
    parser.add_argument(
        "--output-csv",
        default="../data/orienternet_test.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Tile size in meters"
    )
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=256,
        help="Number of rotation hypotheses"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after processing"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=True,
        help="Retry images with KeyError or NaN predictions (default: True)"
    )
    parser.add_argument(
        "--no-retry-failed",
        action="store_false",
        dest="retry_failed",
        help="Do not retry failed images"
    )

    args = parser.parse_args()

    # Create processor
    processor = ImageProcessor(
        h5_path=args.h5,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        tile_size_meters=args.tile_size,
        num_rotations=args.num_rotations,
    )

    # Process images
    try:
        processor.process_all_images(max_images=args.max_images, retry_failed=args.retry_failed)

        # Print summary
        if args.stats:
            processor.get_summary_stats()
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()






