#!/usr/bin/env python3

"""
Merge results CSV files from multiple GPU workers and deduplicate results.

By default, this script is non-destructive:
- It does not delete, rename, or modify the original worker CSV files.
- It does not overwrite an existing output CSV. If the requested output path already
  exists, it writes to a timestamped filename next to it instead.

Usage:
    python merge_results_safe.py

    This will find all files matching results_gpu*.csv in the ../data directory
    and merge them into a single CSV, keeping the best version of each image_id.

    Example with custom paths:
    python merge_results_safe.py --input-dir /path/to/data --output-csv /path/to/final_results.csv

    To explicitly replace an existing output CSV:
    python merge_results_safe.py --overwrite
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class CSVMerger:
    """Merge multiple worker result CSVs and deduplicate by image_id."""

    def __init__(self, input_dir: Path, output_csv: Path, overwrite: bool = False):
        self.input_dir = input_dir
        self.requested_output_csv = output_csv
        self.overwrite = overwrite
        self.output_csv = self._resolve_output_path(output_csv)

    def _resolve_output_path(self, output_csv: Path) -> Path:
        """
        Return a safe output path.

        Default behavior is to keep existing files. If output_csv already exists and
        overwrite is False, create a timestamped output filename instead.
        """
        if self.overwrite or not output_csv.exists():
            return output_csv

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_output = output_csv.with_name(
            f"{output_csv.stem}_{timestamp}{output_csv.suffix}"
        )

        # Extremely unlikely, but avoid collision if the script is run multiple times
        # within the same second.
        counter = 1
        while safe_output.exists():
            safe_output = output_csv.with_name(
                f"{output_csv.stem}_{timestamp}_{counter}{output_csv.suffix}"
            )
            counter += 1

        logger.info(
            "Output file already exists and --overwrite was not set. "
            "Keeping original file and writing new output to %s",
            safe_output,
        )
        return safe_output

    def find_worker_csv_files(self) -> List[Path]:
        """Find all GPU worker CSV files matching results_gpu*.csv pattern."""
        files = sorted(self.input_dir.glob("results_gpu*.csv"))
        logger.info("Found %d GPU worker CSV files", len(files))
        for f in files:
            logger.info("  - %s", f.name)
        return files

    @staticmethod
    def _is_successful_prediction(row: Dict[str, Any]) -> bool:
        """Return True when a row appears to contain a successful prediction."""
        error_message = (row.get("error_message", "") or "").strip()
        pred_x_meters = (row.get("pred_x_meters", "") or "").strip()
        return not error_message and bool(pred_x_meters)

    def merge_csvs(self, worker_files: List[Path]) -> Dict[int, Dict[str, Any]]:
        """
        Merge multiple CSV files and keep the best version of each image_id.

        Priority:
        1. Successful predictions: no error_message and has pred_x_meters.
        2. Failed predictions are replaced by successful ones.
        3. If both rows have the same success status, keep the first one encountered.
        """
        merged_data: Dict[int, Dict[str, Any]] = {}

        for worker_file in worker_files:
            logger.info("Reading GPU worker CSV: %s", worker_file.name)

            try:
                with worker_file.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    if reader.fieldnames is None:
                        logger.warning("CSV file has no headers: %s", worker_file)
                        continue

                    for row in reader:
                        try:
                            image_id = int(row.get("image_id", ""))
                        except (ValueError, TypeError):
                            logger.warning("Skipping row with invalid image_id: %s", row)
                            continue

                        if image_id not in merged_data:
                            merged_data[image_id] = row
                            continue

                        existing = merged_data[image_id]
                        existing_success = self._is_successful_prediction(existing)
                        new_success = self._is_successful_prediction(row)

                        if new_success and not existing_success:
                            merged_data[image_id] = row

            except Exception as exc:
                logger.error("Error reading GPU worker CSV %s: %s", worker_file, exc)
                continue

        logger.info("Merged %d unique images", len(merged_data))
        return merged_data

    def write_merged_csv(self, merged_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Write merged data to output CSV, sorted by image_id.

        This method only writes the merged output file. It never deletes or modifies
        the source worker CSV files.
        """
        if not merged_data:
            logger.warning("No data to write")
            return

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Use the union of all fieldnames so rows with extra columns are preserved.
        fieldnames: List[str] = []
        seen = set()
        for row in merged_data.values():
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        sorted_rows = [
            merged_data[image_id]
            for image_id in sorted(merged_data.keys())
        ]

        logger.info("Writing merged CSV to %s", self.output_csv)

        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted_rows)

        logger.info("Successfully wrote %d rows to %s", len(sorted_rows), self.output_csv)

        if self.output_csv != self.requested_output_csv:
            logger.info("Original output file was kept unchanged: %s", self.requested_output_csv)

    def merge(self) -> None:
        """Execute the full merge process."""
        worker_files = self.find_worker_csv_files()

        if not worker_files:
            logger.warning("No GPU worker CSV files found in %s", self.input_dir)
            return

        merged_data = self.merge_csvs(worker_files)
        self.write_merged_csv(merged_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CSV results from multiple OrienterNet GPU workers.",
    )

    parser.add_argument(
        "--input-dir",
        default="../data",
        help="Directory containing GPU worker CSV files (results_gpu*.csv). Default: ../data",
    )

    parser.add_argument(
        "--output-csv",
        default="../data/orienternet_results-272.csv",
        help="Output CSV file path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite --output-csv if it already exists. By default, existing files "
            "are kept and a timestamped output CSV is created instead."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        merger = CSVMerger(
            input_dir=Path(args.input_dir),
            output_csv=Path(args.output_csv),
            overwrite=args.overwrite,
        )
        merger.merge()
        logger.info("Merge complete!")

    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
