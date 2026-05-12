#!/usr/bin/env python3
import argparse
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def to_bytes(obj):
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, np.bytes_):
        return bytes(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.uint8:
            return obj.tobytes()
        if obj.dtype == object and obj.size == 1:
            return to_bytes(obj.item())

    arr = np.asarray(obj)
    if arr.dtype == np.uint8:
        return arr.tobytes()

    raise TypeError(f"Cannot convert {type(obj)} to bytes")


def guess_ext(raw):
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG"):
        return ".png"
    if raw.startswith(b"II*\x00") or raw.startswith(b"MM\x00*"):
        return ".tif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def extract_range(h5_path, start, count, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    with h5py.File(h5_path, "r") as f:
        images = f["/images"]
        end = min(start + count, len(images))

        for i in range(start, end):
            try:
                raw = to_bytes(images[i])
                ext = guess_ext(raw)

                if ext == ".bin":
                    out_path = output_dir / f"image_{i}.bin"
                    out_path.write_bytes(raw)
                    print(f"{i}: unknown format, saved raw bytes. First bytes: {raw[:16]!r}")
                    continue

                img = Image.open(BytesIO(raw))
                out_path = output_dir / f"image_{i}{ext}"
                img.save(out_path)

                print(f"{i}: saved {out_path}, size={img.size}, mode={img.mode}")
                saved += 1

            except Exception as e:
                print(f"{i}: skipped: {e}")

    print(f"Done. Saved {saved} image(s).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("-o", "--output-dir", default="extracted_images")

    args = parser.parse_args()
    extract_range(args.h5_file, args.start, args.count, args.output_dir)


if __name__ == "__main__":
    main()