import h5py
import os
import glob

# point this to your folder with the .h5 files
path = r"D:\UED_measurements\2025\11 November\25\r000040\RAW"  # change to yours

file_list = sorted(glob.glob(os.path.join(path, "*.h5")))
print(f"Found {len(file_list)} files")
if not file_list:
    raise SystemExit("No files found, check path")

fname = file_list[0]
print("\nInspecting:", fname)

def print_structure(h5obj, indent=0):
    """Recursively print HDF5 structure."""
    for key in h5obj:
        item = h5obj[key]
        prefix = "  " * indent + f"- {key}: "
        if isinstance(item, h5py.Group):
            print(prefix + "(Group)")
            print_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(prefix + f"(Dataset) shape={item.shape}, dtype={item.dtype}")
        else:
            print(prefix + f"(Unknown type {type(item)})")

with h5py.File(fname, "r") as f:
    print("\nTop-level keys:", list(f.keys()))
    print("\nFull structure:")
    print_structure(f)
