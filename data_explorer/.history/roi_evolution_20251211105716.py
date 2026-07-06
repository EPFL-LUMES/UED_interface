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
# Change this to whatever your files actually use
DATASET_NAME = 'data'   # from your code: input_df['data']

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

sum_roi_list = []
indices = []

for i, file in enumerate(file_list):
    with h5py.File(file, 'r') as input_df:
        # Inspect the dataset on first file
        if i == 0:
            print("Keys in file:", list(input_df.keys()))
            print("Shape of dataset 'data':", input_df[DATASET_NAME].shape)

        # Read the dataset
        arr = input_df[DATASET_NAME][()]  # load as numpy array

        # Handle shape:
        # - If it's (512,512): single image
        # - If it's (N,512,512): multiple images, we can take e.g. the first one or loop
        if arr.ndim == 2:
            img = arr
        elif arr.ndim == 3:
            # Example: take first image; adapt if needed
            img = arr[0]
        else:
            raise ValueError(f"Unexpected shape {arr.shape} in {file}")

        if img.shape != (512, 512):
            print(f"WARNING: {os.path.basename(file)} has shape {img.shape}, expected (512, 512)")

        # ROI: remember [row, col] = [y, x]
        roi = img[y0:y1, x0:x1]
        sum_roi = np.sum(roi)

        sum_roi_list.append(sum_roi)
        indices.append(i)

sum_roi_array = np.array(sum_roi_list)
indices = np.array(indices)

print("First 10 ROI sums:", sum_roi_array[:10])

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
