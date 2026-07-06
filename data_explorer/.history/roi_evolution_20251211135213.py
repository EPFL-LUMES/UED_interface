import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import os

plt.rc('font', family='Latin Modern Roman', size=12)
plt.rc('text', usetex=True)

path = r"D:\UED_measurements\2025\11 November\25\r000040\RAW"  # your folder
DATASET_NAME = "data"  # we know this is correct

file_list = sorted(glob.glob(os.path.join(path, "*.h5")))
print(f"Found {len(file_list)} h5 files:")
for fpath in file_list:
    print("  ", os.path.basename(fpath))

if not file_list:
    raise RuntimeError("No .h5 files found, check 'path' and extension.")

y0, y1 = 150, 200   # ROI (rows)
x0, x1 = 200, 250   # ROI (cols)

sum_roi_list = []
indices = []

for i, fpath in enumerate(file_list):
    print(f"\n--- File {i} / {len(file_list)-1}: {os.path.basename(fpath)} ---")
    with h5py.File(fpath, "r") as h5f:
        if i == 0:
            print("Keys in file:", list(h5f.keys()))

        obj = h5f[DATASET_NAME]
        print("  type(h5f[DATASET_NAME]) =", type(obj))
        # for a real dataset, this should be <class 'h5py._hl.dataset.Dataset'>

        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"h5f['{DATASET_NAME}'] is not a Dataset, got {type(obj)}")

        # Read full image (512x512)
        print("  Reading dataset...")
        img = obj[()]   # this should now be a 2D numpy array
        print("  Read done. img.shape =", img.shape)

        if img.shape != (512, 512):
            print("  WARNING: unexpected shape:", img.shape)

        roi = img[y0:y1, x0:x1]
        print("  ROI shape:", roi.shape)

        sum_roi = np.sum(roi)
        print("  Sum(ROI) =", sum_roi)

        sum_roi_list.append(sum_roi)
        indices.append(i)

sum_roi_array = np.array(sum_roi_list)
indices = np.array(indices)

print("\nDone. First 10 sums:", sum_roi_array[:10])

plt.figure()
plt.plot(indices, sum_roi_array, marker="o")
plt.xlabel("Image index")
plt.ylabel("Sum ROI (a.u.)")
plt.title("ROI intensity vs image index")
plt.grid(True)
plt.tight_layout()
plt.show()

