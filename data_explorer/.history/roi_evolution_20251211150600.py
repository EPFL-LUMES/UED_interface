import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import os
from matplotlib.patches import Rectangle

# -----------------------------
# USER SETTINGS
# -----------------------------

# Folder containing your .h5 files
path = r'G:\UED_measurements\2025\12 December\10\r000046\RAW'

# Dataset name containing the 512x512 image
DATASET_PATH = 'data'

# ROI in pixels (rows = y, cols = x)
y0, y1 = 190, 260   # rows
x0, x1 = 140, 210   # columns

# -----------------------------
# FIND FILES
# -----------------------------

file_list = sorted(glob.glob(os.path.join(path, '*.h5')))
print(f"Found {len(file_list)} files:")
for fpath in file_list:
    print("  ", os.path.basename(fpath))

if not file_list:
    raise SystemExit("No .h5 files found, check the path.")


# -----------------------------------------------------------
# VISUALIZE FIRST IMAGE WITH ROI BOX OVERLAY
# -----------------------------------------------------------

first_file = file_list[0]
print("\nShowing first image with ROI overlay:", first_file)

with h5py.File(first_file, "r") as h5f:
    img0 = h5f[DATASET_PATH][()]

plt.figure(figsize=(6, 5))
plt.imshow(img0, cmap='turbo', origin='lower')
plt.colorbar(label='Intensity')

# Add ROI rectangle
width = x1 - x0
height = y1 - y0
rect = Rectangle((x0, y0), width, height,
                 linewidth=2, edgecolor='cyan', facecolor='none')
plt.gca().add_patch(rect)

plt.title("First Image with ROI Overlay")
plt.xlabel("X (pixel)")
plt.ylabel("Y (pixel)")
plt.tight_layout()
plt.show()

# -----------------------------
# LOOP OVER FILES & COMPUTE ROI SUMS
# -----------------------------

sum_roi_list = []
indices = []

for i, fpath in enumerate(file_list):
    print(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(fpath)}")
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
plt.xlabel('Image index')
plt.ylabel('Sum of ROI (a.u.)')
#plt.title('ROI Intensity vs Image Index')
#plt.grid(True)
plt.tight_layout()

save_path = r"G:\UED_measurements\2025\12 December\10\r000046\roi_evolution_plot.png"   
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {save_path}")

plt.show()
