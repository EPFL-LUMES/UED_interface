import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

EXPOSURE_S = 0.005          # 5 ms. Reduce if saturated, increase if too dark.
FRAME_PERIOD_S = 0.05       # 0.05 s = 20 fps target.
RING_BUFFER_FRAMES = 50

# Beam centroid settings
THRESHOLD_REL = 0.15        # Keep pixels above 15% of peak after background subtraction.
BACKGROUND_PERCENTILE = 5   # Background estimate.
BLUR_SIGMA = 1.0            # Slight blur to suppress hot pixels/noise.

# Optional processing ROI: None = use full image.
# Format: (x0, y0, width, height)
PROCESSING_ROI = None
# Example:
# PROCESSING_ROI = (300, 200, 500, 400)

# Pixel-size conversion.
# Use MAGNIFICATION = 1 if the beam is directly on the sensor.
# If you image the beam with optics, set the imaging magnification.
PIXEL_SIZE_UM = 5.2
MAGNIFICATION = 1.0

LOG_FILE = Path("beam_tracking_log.csv")


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    """
    Opens the C1285R12M / old Thorlabs camera through PyLabLib.
    Tries both Thorlabs uc480 and IDS uEye backends.
    """
    from pylablib.devices import uc480

    errors = []

    for backend in ["uc480", "ueye"]:
        try:
            cams = uc480.list_cameras(backend=backend)
            print(f"\nBackend '{backend}' detected cameras:")
            for c in cams:
                print("  ", c)

            if not cams:
                continue

            cam_info = cams[0]
            cam = uc480.UC480Camera(dev_id=cam_info.dev_id, backend=backend)

            print(f"\nUsing backend: {backend}")
            print("Camera info:", cam.get_device_info())

            # Make frame array indexing compatible with OpenCV: frame[y, x]
            try:
                cam.set_image_indexing("rct")
            except Exception as e:
                print("Warning: could not set image indexing:", e)

            # Internal/free-running acquisition
            try:
                cam.set_trigger_mode("int")
            except Exception as e:
                print("Warning: could not set trigger mode:", e)

            # Ignore occasional skipped-frame events instead of crashing
            try:
                cam.set_frameskip_behavior("ignore")
            except Exception as e:
                print("Warning: could not set frameskip behavior:", e)

            # Exposure and frame rate
            try:
                cam.set_exposure(EXPOSURE_S)
            except Exception as e:
                print("Warning: could not set exposure:", e)

            try:
                cam.set_frame_period(FRAME_PERIOD_S)
            except Exception as e:
                print("Warning: could not set frame period:", e)

            # Full frame
            try:
                cam.set_roi()
            except Exception as e:
                print("Warning: could not set full-frame ROI:", e)

            # Start live acquisition
            cam.setup_acquisition(nframes=RING_BUFFER_FRAMES)
            cam.start_acquisition()

            return cam

        except Exception as e:
            errors.append((backend, repr(e)))

    print("\nCould not open camera.")
    print("Make sure ThorCam is closed and that ThorCam can see the camera.")
    print("Errors:")
    for backend, err in errors:
        print(f"  {backend}: {err}")

    raise RuntimeError("No usable uc480/uEye camera found.")


def get_newest_frame(cam):
    """
    Reads newest available frame.
    Returns None if no frame is ready yet.
    """
    frame = cam.read_newest_image()
    if frame is None:
        return None
    return np.asarray(frame)


def close_camera(cam):
    try:
        cam.stop_acquisition()
    except Exception:
        pass

    try:
        cam.close()
    except Exception:
        pass


# ============================================================
# IMAGE PROCESSING
# ============================================================

def to_gray_float(frame):
    """
    Convert frame to float32 grayscale.
    Works for mono or RGB-like frames.
    """
    frame = np.asarray(frame)

    if frame.ndim == 3:
        frame = frame.mean(axis=2)

    return frame.astype(np.float32)


def find_beam_centroid(frame, roi=None):
    """
    Returns beam centroid and size estimates in full-frame pixel coordinates.
    """
    img = to_gray_float(frame)

    if roi is None:
        x0, y0 = 0, 0
        crop = img
    else:
        x0, y0, w, h = roi
        crop = img[y0:y0 + h, x0:x0 + w]

    if crop.size == 0:
        return None

    bg = np.percentile(crop, BACKGROUND_PERCENTILE)

    I = crop - bg
    I[I < 0] = 0

    if BLUR_SIGMA > 0:
        I = cv2.GaussianBlur(I, ksize=(0, 0), sigmaX=BLUR_SIGMA)

    peak = float(I.max())
    if peak <= 0:
        return None

    mask = I > THRESHOLD_REL * peak
    W = I * mask

    total = float(W.sum())
    if total <= 0:
        return None

    yy, xx = np.indices(W.shape)

    x_local = float((xx * W).sum() / total)
    y_local = float((yy * W).sum() / total)

    sx = float(np.sqrt((((xx - x_local) ** 2) * W).sum() / total))
    sy = float(np.sqrt((((yy - y_local) ** 2) * W).sum() / total))

    return {
        "x_px": x_local + x0,
        "y_px": y_local + y0,
        "sx_px": sx,
        "sy_px": sy,
        "fwhm_x_px": 2.355 * sx,
        "fwhm_y_px": 2.355 * sy,
        "total": total,
        "peak": peak,
        "background": float(bg),
    }


# ============================================================
# DISPLAY
# ============================================================

def normalize_for_display(frame):
    img = to_gray_float(frame)

    lo, hi = np.percentile(img, [1, 99.7])
    if hi <= lo:
        hi = lo + 1

    img8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)


def draw_overlay(frame, pos, ref, trail, roi=None):
    vis = normalize_for_display(frame)
    h, w = vis.shape[:2]

    # Image center cross
    cv2.drawMarker(
        vis,
        (w // 2, h // 2),
        (180, 180, 180),
        markerType=cv2.MARKER_CROSS,
        markerSize=35,
        thickness=1,
    )

    # Processing ROI
    if roi is not None:
        x0, y0, rw, rh = roi
        cv2.rectangle(vis, (x0, y0), (x0 + rw, y0 + rh), (255, 255, 0), 1)

    # Reference beam position
    if ref is not None:
        rx, ry = int(round(ref[0])), int(round(ref[1]))
        cv2.drawMarker(
            vis,
            (rx, ry),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=45,
            thickness=1,
        )

    # Trail
    if len(trail) > 1:
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in trail],
            dtype=np.int32,
        )
        cv2.polylines(vis, [pts], isClosed=False, color=(0, 180, 255), thickness=1)

    # Current beam position
    if pos is not None:
        x = int(round(pos["x_px"]))
        y = int(round(pos["y_px"]))

        cv2.drawMarker(
            vis,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=45,
            thickness=2,
        )
        cv2.circle(vis, (x, y), 10, (0, 0, 255), 1)

        if ref is None:
            text = f"x={pos['x_px']:.1f}px  y={pos['y_px']:.1f}px"
        else:
            dx = pos["x_px"] - ref[0]
            dy = pos["y_px"] - ref[1]
            scale = PIXEL_SIZE_UM / MAGNIFICATION
            text = (
                f"x={pos['x_px']:.1f}px  y={pos['y_px']:.1f}px   "
                f"dx={dx:+.1f}px  dy={dy:+.1f}px   "
                f"dx={dx * scale:+.1f}um  dy={dy * scale:+.1f}um"
            )

        cv2.putText(
            vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        "q=quit/save   r=set reference   c=clear trail",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# MAIN
# ============================================================

def main():
    cam = open_camera()

    scale_um_per_px = PIXEL_SIZE_UM / MAGNIFICATION

    ref = None
    trail = deque(maxlen=300)

    t0 = time.time()
    frame_counter = 0

    print("\nRunning beam tracking.")
    print("Controls:")
    print("  q : quit and save")
    print("  r : set current beam position as reference")
    print("  c : clear trail")
    print(f"\nLogging to: {LOG_FILE.resolve()}")

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "elapsed_s",
            "frame",
            "x_px",
            "y_px",
            "dx_px",
            "dy_px",
            "x_um",
            "y_um",
            "dx_um",
            "dy_um",
            "sx_px",
            "sy_px",
            "fwhm_x_px",
            "fwhm_y_px",
            "peak",
            "total",
            "background",
        ])
        f.flush()

        try:
            while True:
                frame = get_newest_frame(cam)

                if frame is None:
                    time.sleep(0.002)
                    continue

                frame_counter += 1
                elapsed = time.time() - t0

                pos = find_beam_centroid(frame, roi=PROCESSING_ROI)

                if pos is not None:
                    if ref is None:
                        ref = (pos["x_px"], pos["y_px"])

                    dx_px = pos["x_px"] - ref[0]
                    dy_px = pos["y_px"] - ref[1]

                    trail.append((pos["x_px"], pos["y_px"]))

                    writer.writerow([
                        datetime.now().isoformat(timespec="milliseconds"),
                        elapsed,
                        frame_counter,
                        pos["x_px"],
                        pos["y_px"],
                        dx_px,
                        dy_px,
                        pos["x_px"] * scale_um_per_px,
                        pos["y_px"] * scale_um_per_px,
                        dx_px * scale_um_per_px,
                        dy_px * scale_um_per_px,
                        pos["sx_px"],
                        pos["sy_px"],
                        pos["fwhm_x_px"],
                        pos["fwhm_y_px"],
                        pos["peak"],
                        pos["total"],
                        pos["background"],
                    ])

                    # Flush regularly so data is not lost if script is interrupted
                    if frame_counter % 20 == 0:
                        f.flush()

                vis = draw_overlay(
                    frame,
                    pos=pos,
                    ref=ref,
                    trail=trail,
                    roi=PROCESSING_ROI,
                )

                cv2.imshow("C1285R12M beam tracking", vis)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("r") and pos is not None:
                    ref = (pos["x_px"], pos["y_px"])
                    trail.clear()
                    print(f"Reference set to x={ref[0]:.2f}px, y={ref[1]:.2f}px")

                if key == ord("c"):
                    trail.clear()
                    print("Trail cleared.")

        finally:
            f.flush()
            close_camera(cam)
            cv2.destroyAllWindows()

    print(f"\nDone. Saved log to: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()