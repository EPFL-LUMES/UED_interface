import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from instrumental.drivers.cameras import uc480


# ============================================================
# USER SETTINGS
# ============================================================

EXPOSURE_TIME = "20ms"
FRAMERATE = "2Hz"       # deliberately slow for debugging; increase later
HSUB = 4
VSUB = 4

THRESHOLD_REL = 0.15
BACKGROUND_PERCENTILE = 5
BLUR_SIGMA = 1.0

PIXEL_SIZE_UM = 5.2
MAGNIFICATION = 1.0

LOG_FILE = Path("beam_tracking_log.csv")


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    print("Searching UC480 cameras with Instrumental...")
    instruments = uc480.list_instruments()

    print("Detected instruments:")
    for i, inst in enumerate(instruments):
        print(f"  [{i}] {inst}")

    if len(instruments) == 0:
        raise RuntimeError(
            "No UC480 camera detected by Instrumental. "
            "Check that ThorCam is closed and that uc480_64.dll is in PATH."
        )

    cam = uc480.UC480_Camera(instruments[0], reopen_policy="reuse")

    print("Opened camera.")
    print("Model:", getattr(cam, "model", "unknown"))
    print("Serial:", getattr(cam, "serial", "unknown"))

    try:
        cam.set_trigger(mode="off")
        print("Trigger mode set to off/internal.")
    except Exception as e:
        print("Warning: could not set trigger mode:", repr(e))

    print("Starting live video...")
    cam.start_live_video(
        framerate=FRAMERATE,
        exposure_time=EXPOSURE_TIME,
        hsub=HSUB,
        vsub=VSUB,
    )

    return cam


def close_camera(cam):
    try:
        cam.stop_live_video()
    except Exception:
        pass

    try:
        cam.close()
    except Exception:
        pass


def get_frame(cam):
    ready = cam.wait_for_frame(timeout="3s")

    if not ready:
        print("No frame ready within timeout.")
        return None

    frame = cam.latest_frame(copy=True)
    return np.asarray(frame)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def to_gray_float(frame):
    frame = np.asarray(frame)

    if frame.ndim == 3:
        frame = frame.mean(axis=2)

    return frame.astype(np.float32)


def find_beam_centroid(frame):
    img = to_gray_float(frame)

    bg = np.percentile(img, BACKGROUND_PERCENTILE)

    I = img - bg
    I[I < 0] = 0

    if BLUR_SIGMA > 0:
        I = cv2.GaussianBlur(I, ksize=(0, 0), sigmaX=BLUR_SIGMA)

    peak = float(I.max())
    if peak <= 0:
        return None

    W = I * (I > THRESHOLD_REL * peak)

    total = float(W.sum())
    if total <= 0:
        return None

    yy, xx = np.indices(W.shape)

    x = float((xx * W).sum() / total)
    y = float((yy * W).sum() / total)

    sx = float(np.sqrt((((xx - x) ** 2) * W).sum() / total))
    sy = float(np.sqrt((((yy - y) ** 2) * W).sum() / total))

    return {
        "x_px": x,
        "y_px": y,
        "sx_px": sx,
        "sy_px": sy,
        "fwhm_x_px": 2.355 * sx,
        "fwhm_y_px": 2.355 * sy,
        "peak": peak,
        "total": total,
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


def draw_overlay(frame, pos, ref, trail):
    vis = normalize_for_display(frame)
    h, w = vis.shape[:2]

    # Image center
    cv2.drawMarker(
        vis,
        (w // 2, h // 2),
        (180, 180, 180),
        markerType=cv2.MARKER_CROSS,
        markerSize=35,
        thickness=1,
    )

    # Reference position
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

    # Current centroid
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
            dx_px = 0.0
            dy_px = 0.0
        else:
            dx_px = pos["x_px"] - ref[0]
            dy_px = pos["y_px"] - ref[1]

        scale = PIXEL_SIZE_UM / MAGNIFICATION

        text = (
            f"x={pos['x_px']:.1f}px  y={pos['y_px']:.1f}px   "
            f"dx={dx_px:+.1f}px  dy={dy_px:+.1f}px   "
            f"dx={dx_px * scale:+.1f}um  dy={dy_px * scale:+.1f}um"
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

    ref = None
    trail = deque(maxlen=300)

    scale_um_per_px = PIXEL_SIZE_UM / MAGNIFICATION

    t0 = time.time()
    frame_counter = 0

    cv2.namedWindow("C1285R12M Instrumental beam tracking", cv2.WINDOW_NORMAL)

    print("\nRunning.")
    print("Controls:")
    print("  q : quit and save")
    print("  r : set current beam as reference")
    print("  c : clear trail")
    print("Log file:", LOG_FILE.resolve())

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
                frame = get_frame(cam)

                if frame is None:
                    continue

                frame_counter += 1
                elapsed = time.time() - t0

                if frame_counter == 1:
                    print("First frame received.")
                    print("Frame shape:", frame.shape)
                    print("Frame dtype:", frame.dtype)
                    print("Frame min/max:", frame.min(), frame.max())

                pos = find_beam_centroid(frame)

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

                    if frame_counter % 10 == 0:
                        f.flush()

                vis = draw_overlay(frame, pos, ref, trail)
                cv2.imshow("C1285R12M Instrumental beam tracking", vis)

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

    print("Done. Saved:", LOG_FILE.resolve())


if __name__ == "__main__":
    main()