#!/usr/bin/env python3

"""
Helper script to automatically launch multiple worker processes for batch processing.

This script spawns worker subprocesses with appropriate GPU assignment.
It's useful for automating the launch sequence.

Usage:
    python launch_workers.py --num-workers 2 --output-dir ../data
    python launch_workers.py --num-workers 4 --gpus 0 1 --output-dir ../data
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_count() -> int:
    """Get number of available GPUs (requires nvidia-smi)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return len([line for line in result.stdout.split("\n") if line.strip()])
    except Exception as e:
        logger.warning("Could not detect GPU count: %s. Defaulting to 1.", e)
        return 1


def launch_worker(
    worker_id: int,
    num_workers: int,
    output_dir: Path,
    gpu_id: Optional[int] = None,
    max_images: Optional[int] = None,
    tile_size: int = 64,
    num_rotations: int = 256,
) -> subprocess.Popen:
    """Launch a worker subprocess."""
    output_csv = output_dir / f"results_worker_{worker_id}.csv"

    cmd = [
        sys.executable,
        "batch_process.py",
        f"--worker-id={worker_id}",
        f"--num-workers={num_workers}",
        f"--output-csv={output_csv}",
        f"--tile-size={tile_size}",
        f"--num-rotations={num_rotations}",
    ]

    if max_images is not None:
        cmd.append(f"--max-images={max_images}")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(
            "Launching worker %d/%d on GPU %d (output: %s)",
            worker_id,
            num_workers,
            gpu_id,
            output_csv,
        )
    else:
        logger.info(
            "Launching worker %d/%d (output: %s)",
            worker_id,
            num_workers,
            output_csv,
        )

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return proc


def read_process_output(proc: subprocess.Popen, worker_id: int) -> None:
    """Read and log output from a worker process."""
    try:
        for line in proc.stdout:
            if line.strip():
                print(f"[Worker {worker_id}] {line.rstrip()}")
    except Exception as e:
        logger.error("Error reading output from worker %d: %s", worker_id, e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple worker processes for parallel image processing.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers to launch.",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to assign to workers (defaults to auto-detect available GPUs).",
    )

    parser.add_argument(
        "--output-dir",
        default="../data",
        help="Directory for output CSV files.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images per worker (optional).",
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
        default=256,
        help="Number of rotation hypotheses.",
    )

    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all workers to complete before exiting.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine GPU assignment
    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        gpu_count = get_gpu_count()
        logger.info("Detected %d GPU(s)", gpu_count)
        # Assign workers to GPUs in round-robin fashion
        gpu_ids = [i % gpu_count for i in range(args.num_workers)]

    if len(gpu_ids) < args.num_workers:
        logger.warning(
            "Only %d GPUs available for %d workers. Some workers will share GPUs.",
            len(gpu_ids),
            args.num_workers,
        )
        gpu_ids = gpu_ids * (args.num_workers // len(gpu_ids) + 1)
        gpu_ids = gpu_ids[:args.num_workers]

    logger.info("GPU assignment: %s", {i: gpu_ids[i] for i in range(args.num_workers)})

    # Launch all workers
    processes = []
    for worker_id in range(args.num_workers):
        gpu_id = gpu_ids[worker_id]
        proc = launch_worker(
            worker_id=worker_id,
            num_workers=args.num_workers,
            output_dir=output_dir,
            gpu_id=gpu_id,
            max_images=args.max_images,
            tile_size=args.tile_size,
            num_rotations=args.num_rotations,
        )
        processes.append(proc)

        # Stagger worker start times slightly
        time.sleep(2)

    logger.info("All %d workers launched", args.num_workers)

    if args.wait:
        logger.info("Waiting for workers to complete...")
        try:
            for i, proc in enumerate(processes):
                returncode = proc.wait()
                logger.info("Worker %d completed with return code %d", i, returncode)
        except KeyboardInterrupt:
            logger.info("Interrupted. Terminating workers...")
            for proc in processes:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            sys.exit(1)

        logger.info("All workers completed")
        logger.info("Next step: run 'python merge_results.py --input-dir %s'", output_dir)
    else:
        logger.info(
            "Workers running in background. Monitor logs in %s",
            output_dir,
        )
        logger.info("Run 'python merge_results.py --input-dir %s' after all workers complete", output_dir)


if __name__ == "__main__":
    main()

