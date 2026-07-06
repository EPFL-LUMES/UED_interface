import numpy as np 
import matplotlib.pyplot as plt
#plt.rc('font', family='Latin Modern Roman', size=12)
#plt.rc('text', usetex=True)

import glob
import h5py
import os

# -----------------------------
# USER SETTINGS
# -----------------------------

# Folder containing your "data/*.h5"
path = r'G:\UED_measurements\2025\12 December\10\r000046\RAW'   

# HDF5 dataset name that contains the image(s)

# Define ROI (in pixels, 0-based, end index is exclusive)
# Example: 50×50 box around (x=200..249, y=150..199)
y0, y1 = 150, 200   # rows
x0, x1 = 200, 250   # columns

# -----------------------------
# LOAD FILES & COMPUTE ROI SUMS
# -----------------------------

# Collect and sort file list
file_list = sorted(glob.glob(os.path.join(path, '*.h5')))
print(f"Found {len(file_list)} files:")
for f in file_list:
    print("  ", os.path.basename(f))
    

print("\nTop-level keys:", list(f.keys()))
print("\nFull structure:")

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


sum_roi_list = []
indices = []

for i, file in enumerate(file_list):
    with h5py.File(file, 'r') as input_df:
        # Inspect the dataset on first file
        if i == 0:
            print("Keys in file:", list(input_df.keys()))
            print("Shape of dataset 'data':", input_df[DATASET_NAME].shape)

       # Read full image (512x512)
        print("  Reading dataset...")
        img = f[DATASET_NAME][()]   # <-- this actually reads the array
        print("  Read done. img.shape =", img.shape)

        # Sanity check
        if img.shape != (512, 512):
            print("  WARNING: unexpected shape:", img.shape)

        # Extract ROI
        roi = img[y0:y1, x0:x1]
        print("  ROI shape:", roi.shape)

        sum_roi = np.sum(roi)
        print("  Sum(ROI) =", sum_roi)

        sum_roi_list.append(sum_roi)
        indices.append(i)

sum_roi_array = np.array(sum_roi_list)
indices = np.array(indices)

print("\nDone. First 10 sums:", sum_roi_array[:10])

# -----------------------------
# PLOT SUM(ROI) vs IMAGE INDEX
# -----------------------------

plt.figure()
plt.plot(indices, sum_roi_array, marker='o')
plt.xlabel('Image index (H5 file order)')
plt.ylabel('Sum of ROI (a.u.)')
plt.title('Evolution of ROI intensity')
plt.grid(True)
plt.tight_layout()
plt.show()
