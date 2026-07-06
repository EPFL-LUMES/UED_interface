import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import os

# -----------------------------
# USER SETTINGS
# -----------------------------

# Folder containing your .h5 files
path = r'G:\UED_measurements\2025\12 December\10\r000046\RAW'

# ROI in pixels (rows = y, cols = x)
y0, y1 = 150, 200   # rows
x0, x1 = 200, 250   # columns

# -----------------------------
# FIND FILES
# -----------------------------

file_list = sorted(glob.glob(os.path.join(path, '*.h5')))
print(f"Found {len(file_list)} files:")
for fpath in file_list:
    print("  ", os.path.basename(fpath))

if not file_list:
    raise SystemExit("No .h5 files found, check 'path' and extension.")

# -----------------------------
# INSPECT FIRST FILE & FIND DATASET
# -----------------------------

def print_structure(h5obj, indent=0, datasets=None, prefix=""):
    """Recursively print structure and collect dataset paths."""
    if datasets is None:
        datasets = []
    for key in h5obj:
        obj = h5obj[key]
        full_path = f"{prefix}/{key}" if prefix else key
        pad = "  " * indent
        if isinstance(obj, h5py.Group):
            print(f"{pad}- {full_path} (Group)")
            print_structure(obj, indent+1, datasets, full_path)
        elif isinstance(obj, h5py.Dataset):
            print(f"{pad}- {full_path} (Dataset) shape={obj.shape}, dtype={obj.dtype}")
            datasets.append((full_path, obj.shape, obj.dtype))
        else:
            print(f"{pad}- {full_path} (Unknown type {type(obj)})")
    return datasets

first_file = file_list[0]
print("\nInspecting structure of:", first_file)

with h5py.File(first_file, "r") as h5f:
    print("\nTop-level keys:", list(h5f.keys()))
    print("\nFull structure:")
    datasets = print_structure(h5f)

# Try to auto-select the first 512x512 dataset
DATASET_PATH = None
for path_, shape_, dtype_ in datasets:
    if shape_ == (512, 512):
        DATASET_PATH = path_
        break

if DATASET_PATH is None:
    raise RuntimeError("Could not find any dataset of shape (512, 512). Use the printed structure to choose one.")

print(f"\nUsing dataset: '{DATASET_PATH}' as the image (512x512)\n")

# -----------------------------
# LOOP OVER FILES & COMPUTE ROI SUMS
# -----------------------------

sum_roi_list = []
indices = []

for i, fpath in enumerate(file_list):
    print(f"Processing file {i} / {len(file_list)-1}: {os.path.basename(fpath)}")
    if i == 20000:
        break
    with h5py.File(fpath, 'r') as h5f:
        dset = h5f[DATASET_PATH]
        img = dset[()]   # read full 2D array

        if img.shape != (512, 512):
            print("  WARNING: unexpected shape:", img.shape)

        roi = img[y0:y1, x0:x1]
        sum_roi = np.sum(roi)

        sum_roi_list.append(sum_roi)
        indices.append(i)

sum_roi_array = np.array(sum_roi_list)
indices = np.array(indices)

print("\nDone. First 10 ROI sums:", sum_roi_array[:10])

# -----------------------------
# PLOT SUM(ROI) vs IMAGE INDEX
# -----------------------------

plt.figure()
plt.plot(indices, sum_roi_array, marker='o')
plt.xlabel('Image index (H5 file order)')
plt.ylabel('Sum of ROI (a.u.)')
plt.title(f'ROI intensity vs index\nDataset: {DATASET_PATH}')
plt.grid(True)
plt.tight_layout()
plt.show()
