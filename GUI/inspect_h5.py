import h5py
import sys

def inspect_item(name, obj):
    indent = "  " * (name.count("/") - 1 if "/" in name else 0)

    if isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}")
        print(f"{indent}  shape: {obj.shape}")
        print(f"{indent}  dtype: {obj.dtype}")

        if obj.attrs:
            print(f"{indent}  attributes:")
            for k, v in obj.attrs.items():
                print(f"{indent}    {k}: {v}")

    elif isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
        if obj.attrs:
            print(f"{indent}  attributes:")
            for k, v in obj.attrs.items():
                print(f"{indent}    {k}: {v}")


def inspect_h5_file(filename):
    with h5py.File(filename, "r") as f:
        print(f"HDF5 file: {filename}")
        print("-" * 60)
        f.visititems(inspect_item)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_h5.py <file.h5>")
        sys.exit(1)

    inspect_h5_file(sys.argv[1])
