#!/usr/bin/env python3
import argparse
import h5py


def describe_h5(path):
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\nDATASET: /{name}")
                print(f"  shape: {obj.shape}")
                print(f"  dtype: {obj.dtype}")
                print(f"  ndim:  {obj.ndim}")

                try:
                    print(f"  compression: {obj.compression}")
                except Exception:
                    pass

                # Print attributes
                if obj.attrs:
                    print("  attrs:")
                    for k, v in obj.attrs.items():
                        print(f"    {k}: {v}")

            elif isinstance(obj, h5py.Group):
                print(f"\nGROUP: /{name}")
                if obj.attrs:
                    print("  attrs:")
                    for k, v in obj.attrs.items():
                        print(f"    {k}: {v}")

        f.visititems(visit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file")
    args = parser.parse_args()

    describe_h5(args.h5_file)