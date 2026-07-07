import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from instrumental import instrument
from instrumental.drivers.cameras import uc480


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_EXPOSURE = "10ms"
DEFAULT_FRAMERATE = "5Hz"

# Use subsampling to make old USB camera acquisition more reliable.
# hsub=vsub=4 means the displayed/tracked image is reduced by 4 in x and y.
DEFAULT_HSUB = 4
DEFAULT_VSUB = 4

PIXEL_SIZE_UM = 5.2
DEFAULT_MAGNIFICATION = 1.0

THRESHOLD_REL = 0.15
BACKGROUND_PERCENTILE = 5
BLUR_SIGMA = 1.0


# ============================================================
# ROI SELECTION STATE
# ============================================================

mouse_state = {
    "dragging": False,
    "p0": None,
    "roi": None,
    "temp_roi": None,
}


def normalize_roi(x0, y0, x1, y1):
    x = min(x0, x1)
    y = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    return x, y, w, h


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["dragging"] = True
        mouse_state["p0"] = (x, y)
        mouse_state["temp_roi"] = None

    elif event == cv2.EVENT_MOUSEMOVE and mouse_state["dragging"]:
        x0, y0 = mouse_state["p0"]
        mouse_state["temp_roi"] = normalize_roi(x0, y0, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["dragging"] = False
        x0, y0 = mouse_state["p0"]
        roi = normalize_roi(x0, y0, x, y)

        if roi[2] > 10 and roi[3] > 10:
            mouse_state["roi"] = roi
            print(f"ROI set to x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        else:
            mouse_state["roi"] = None
            print("ROI cleared.")

        mouse_state["temp_roi"] = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse_state["roi"] = None
        mouse_state["temp_roi"] = None
        print("ROI cleared.")


# ============================================================
# CAMERA
# ============================================================

def open_camera(args):
    print("Searching for UC480 / old Thorlabs camera...")

    cams = uc480.list_instruments()

    print("Detected cameras:")
    for i, cam_info in enumerate(cams):
        print(f"  [{i}] {cam_info}")

    if len(cams) == 0:
        raise RuntimeError(
            "No UC480 camera detected. Close ThorCam, unplug/replug the camera, "
            "check that ThorCam sees it, then close ThorCam again."
        )

    cam = instrument(cams[0])
    print("Camera opened.")

    try:
        cam.set_trigger(mode="off")
        print("Trigger set to internal/off.")
    except Exception as e:
        print("Warning: could not set trigger mode:", repr(e))

    print("Starting live video...")
    print(f"Exposure: {args.exposure}")
    print(f"Framerate: {args.framerate}")
    print(f"Subsampling: hsub={args.hsub}, vsub={args.vsub}")

    try:
        cam.start_live_video(
            framerate=args.framerate,
            exposure_time=args.exposure,
            hsub=args.hsub,
            vsub=args.vsub,
        )
    except TypeError:
        print("start_live_video did not accept hsub/vsub. Trying without subsampling arguments.")
        cam.start_live_video(
            framerate=args.framerate,
            exposure_time=args.exposure,
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
    try:
        ready = cam.wait_for_frame(timeout="3s")

        if not ready:
            return None

        frame = cam.latest_frame(copy=True)

        if frame is None:
            return None

        return np.asarray(frame)

    except Exception as e:
        print("Frame acquisition failed:", repr(e))
        return None


# ============================================================
# IMAGE PROCESSING
# ============================================================

def to_gray_float(frame):
    frame = np.asarray(frame)

    if frame.ndim == 3:
        frame = frame.mean(axis=2)

    return frame.astype(np.float32)


def clamp_roi_to_image(roi, image_shape):
    if roi is None:
        return None

    h, w = image_shape[:2]
    x, y, rw, rh = roi

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))

    return x, y, rw, rh


def find_beam_centroid(frame, roi=None):
    img = to_gray_float(frame)

    roi = clamp_roi_to_image(roi, img.shape)

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

    W = I * (I > THRESHOLD_REL * peak)

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
        "peak": peak,
        "total": total,
        "background": float(bg),
        "roi": roi,
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


def draw_overlay(frame, pos, ref, trail, args, fps):
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

    # ROI or currently dragged ROI
    roi = mouse_state["roi"]
    temp_roi = mouse_state["temp_roi"]

    if roi is not None:
        x, y, rw, rh = clamp_roi_to_image(roi, vis.shape)
        cv2.rectangle(vis, (x, y), (x + rw, y + rh), (255, 255, 0), 1)

    if temp_roi is not None:
        x, y, rw, rh = clamp_roi_to_image(temp_roi, vis.shape)
        cv2.rectangle(vis, (x, y), (x + rw, y + rh), (255, 0, 255), 1)

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

        cv2.polylines(
            vis,
            [pts],
            isClosed=False,
            color=(0, 180, 255),
            thickness=1,
        )

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
            dx_px = 0.0
            dy_px = 0.0
        else:
            dx_px = pos["x_px"] - ref[0]
            dy_px = pos["y_px"] - ref[1]

        dx_um = dx_px * args.hsub * PIXEL_SIZE_UM / args.magnification
        dy_um = dy_px * args.vsub * PIXEL_SIZE_UM / args.magnification

        text = (
            f"x={pos['x_px']:.1f}px  y={pos['y_px']:.1f}px   "
            f"dx={dx_px:+.1f}px  dy={dy_px:+.1f}px   "
            f"dx={dx_um:+.1f}um  dy={dy_um:+.1f}um"
        )

        cv2.putText(
            vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if pos["peak"] >= 254:
            cv2.putText(
                vis,
                "WARNING: saturated beam peak",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    else:
        cv2.putText(
            vis,
            "No beam detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    logging_text = "logging ON" if args.log is not None else "logging OFF"

    cv2.putText(
        vis,
        f"{logging_text}   FPS={fps:.1f}",
        (10, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        "q=quit   r=set reference   c=clear trail   right-click=clear ROI   left-drag=set ROI",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# LOGGING
# ============================================================

def open_logger(log_path):
    if log_path is None:
        return None, None

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f = open(path, "w", newline="")
    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "elapsed_s",
        "frame",
        "x_img_px",
        "y_img_px",
        "dx_img_px",
        "dy_img_px",
        "x_sensor_px",
        "y_sensor_px",
        "dx_sensor_px",
        "dy_sensor_px",
        "dx_um",
        "dy_um",
        "sx_img_px",
        "sy_img_px",
        "fwhm_x_img_px",
        "fwhm_y_img_px",
        "peak",
        "total",
        "background",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
    ])

    f.flush()

    print(f"Logging to: {path.resolve()}")

    return f, writer


def log_position(writer, f, pos, ref, elapsed, frame_counter, args):
    if writer is None or pos is None:
        return

    if ref is None:
        dx_px = 0.0
        dy_px = 0.0
    else:
        dx_px = pos["x_px"] - ref[0]
        dy_px = pos["y_px"] - ref[1]

    x_sensor_px = pos["x_px"] * args.hsub
    y_sensor_px = pos["y_px"] * args.vsub

    dx_sensor_px = dx_px * args.hsub
    dy_sensor_px = dy_px * args.vsub

    dx_um = dx_sensor_px * PIXEL_SIZE_UM / args.magnification
    dy_um = dy_sensor_px * PIXEL_SIZE_UM / args.magnification

    roi = pos["roi"]
    if roi is None:
        roi_x, roi_y, roi_w, roi_h = "", "", "", ""
    else:
        roi_x, roi_y, roi_w, roi_h = roi

    writer.writerow([
        datetime.now().isoformat(timespec="milliseconds"),
        elapsed,
        frame_counter,
        pos["x_px"],
        pos["y_px"],
        dx_px,
        dy_px,
        x_sensor_px,
        y_sensor_px,
        dx_sensor_px,
        dy_sensor_px,
        dx_um,
        dy_um,
        pos["sx_px"],
        pos["sy_px"],
        pos["fwhm_x_px"],
        pos["fwhm_y_px"],
        pos["peak"],
        pos["total"],
        pos["background"],
        roi_x,
        roi_y,
        roi_w,
        roi_h,
    ])

    if frame_counter % 10 == 0:
        f.flush()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Live beam tracker for Thorlabs/IDS C1285R12M camera.")

    parser.add_argument("--log", default=None, help="Optional CSV path. If omitted, no file is saved.")
    parser.add_argument("--exposure", default=DEFAULT_EXPOSURE, help="Exposure time, e.g. 5ms, 20ms, 100ms.")
    parser.add_argument("--framerate", default=DEFAULT_FRAMERATE, help="Frame rate, e.g. 2Hz, 5Hz, 10Hz.")
    parser.add_argument("--hsub", type=int, default=DEFAULT_HSUB, help="Horizontal subsampling.")
    parser.add_argument("--vsub", type=int, default=DEFAULT_VSUB, help="Vertical subsampling.")
    parser.add_argument("--magnification", type=float, default=DEFAULT_MAGNIFICATION, help="Imaging magnification.")

    args = parser.parse_args()

    cam = open_camera(args)

    log_file, writer = open_logger(args.log)

    window_name = "C1285R12M beam tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    ref = None
    trail = deque(maxlen=300)

    frame_counter = 0
    t0 = time.time()
    frame_times = deque(maxlen=30)

    print("\nBeam tracker running.")
    print("Controls:")
    print("  left-drag mouse : define tracking ROI")
    print("  right-click     : clear ROI")
    print("  r               : set current beam as reference")
    print("  c               : clear trail")
    print("  q or ESC        : quit")
    print("\nNo CSV is written unless you use --log path\\file.csv")

    try:
        while True:
            frame = get_frame(cam)

            if frame is None:
                continue

            frame_counter += 1
            elapsed = time.time() - t0

            frame_times.append(time.time())
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            else:
                fps = 0.0

            if frame_counter == 1:
                print("First frame received.")
                print("Frame shape:", frame.shape)
                print("Frame dtype:", frame.dtype)
                print("Frame min/max:", frame.min(), frame.max())

            roi = mouse_state["roi"]
            pos = find_beam_centroid(frame, roi=roi)

            if pos is not None:
                if ref is None:
                    ref = (pos["x_px"], pos["y_px"])

                trail.append((pos["x_px"], pos["y_px"]))

                log_position(writer, log_file, pos, ref, elapsed, frame_counter, args)

            vis = draw_overlay(frame, pos, ref, trail, args, fps)
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("r") and pos is not None:
                ref = (pos["x_px"], pos["y_px"])
                trail.clear()
                print(f"Reference set to x={ref[0]:.2f}px, y={ref[1]:.2f}px")

            if key == ord("c"):
                trail.clear()
                print("Trail cleared.")

    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()
            print(f"Saved log to: {Path(args.log).resolve()}")

        close_camera(cam)
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()