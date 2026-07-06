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

# Dataset name containing the 512x512 image
DATASET_PATH = 'data'

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
    raise SystemExit("No .h5 files found, check the path.")

# -----------------------------
# LOOP OVER FILES & COMPUTE ROI SUMS
# -----------------------------

sum_roi_list = []
indices = []

for i, fpath in enumerate(file_list):
    #print(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(fpath)}")
    if i == 15000:
        break
    with h5py.File(fpath, 'r') as h5f:
        # Load the image
        img = h5f[DATASET_PATH][()]   # numpy array (512x512)

        if img.shape != (512, 512):
            print("  WARNING: unexpected image shape:", img.shape)

        # Extract ROI
        roi = img[y0:y1, x0:x1]

        # Compute the sum of all pixel values in the ROI
        sum_roi = np.sum(roi)

        sum_roi_list.append(sum_roi)
        indices.append(i)

sum_roi_array = np.array(sum_roi_list)
indices = np.array(indices)

print("\nDone! First 10 ROI sums:", sum_roi_array[:10])

# -----------------------------
# PLOT SUM(ROI) vs IMAGE INDEX
# -----------------------------

plt.figure()
plt.plot(indices, sum_roi_array, marker='o', markersize=6)
plt.xlabel('Image index (H5 file order)')
plt.ylabel('Sum of ROI (a.u.)')
plt.title('ROI Intensity vs Image Index')
plt.grid(True)
plt.tight_layout()

save_path = r"G:\UED_measurements\2025\12 December\10\r000046\roi_evolution_plot.png"   
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {save_path}")

plt.show()
