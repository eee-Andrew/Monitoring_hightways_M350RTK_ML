#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster real-time ML road detection with CLIPSeg.

Speed-ups:
  • CUDA autocast + half precision (if GPU available)
  • Single text prompt by default ("road")
  • Frame skipping (reuse last mask between inferences)
  • ROI cropped to ignore HUD/borders
"""

import argparse, sys, time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# -----------------------------
# Tunables (can tweak live)
# -----------------------------
PROMPTS = ["road"]          # keep to 1 prompt for speed
THRESH = 0.35               # mask threshold (hotkeys -/=)
MASK_SMOOTH_WINDOW = 5      # temporal smoothing
IGNORE_BORDER_PCT = 0.10    # inset ROI to avoid HUD (hotkeys [ / ])
FRAME_SKIP = 2              # run model every N frames (hotkeys , / .)
SHOW_FPS = True
MAX_WRITE_FPS_FALLBACK = 25.0

# -----------------------------
# Geometry helpers
# -----------------------------
def fit_axis_from_mask(mask: np.ndarray) -> Optional[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    w, h = max(w, 1.0), max(h, 1.0)
    long_len = max(w, h)
    short_len = min(w, h)
    theta = np.deg2rad(angle)
    if h > w:
        theta += np.pi / 2.0
    vx, vy = np.cos(theta), np.sin(theta)
    p1 = (cx - 0.5 * long_len * vx, cy - 0.5 * long_len * vy)
    p2 = (cx + 0.5 * long_len * vx, cy + 0.5 * long_len * vy)
    return {
        "center": (cx, cy),
        "axis": (p1, p2),
        "length": long_len,
        "width": short_len,
        "angle_deg": float((np.rad2deg(theta) + 360.0) % 360.0),
        "rect": rect,
        "contour": c,
    }

def overlay(frame, full_mask, axis_info, conf, roi_box=None, show_mask=False):
    out = frame.copy()
    if show_mask:
        cm = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        cm[:, :, 1] = np.maximum(cm[:, :, 1], full_mask)
        out = cv2.addWeighted(out, 0.6, cm, 0.4, 0)

    if roi_box is not None:
        x0, y0, x1, y1 = roi_box
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 1)

    if axis_info is not None:
        box = cv2.boxPoints(axis_info["rect"]).astype(int)
        cv2.polylines(out, [box], True, (0, 255, 255), 2)
        (x1, y1), (x2, y2) = axis_info["axis"]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cx, cy = axis_info["center"]
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    y = 28
    def put(txt):
        nonlocal y
        cv2.putText(out, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 255, 240), 1, cv2.LINE_AA)
        y += 22

    put(f"Road detection: {'FOUND' if conf>0.35 else 'searching…'}   conf={conf:.2f}   thr={THRESH:.2f}   skip={FRAME_SKIP}")
    if axis_info is not None:
        put(f"angle={axis_info['angle_deg']:.1f}°  width={axis_info['width']:.1f}px  length={axis_info['length']:.1f}px")
    return out

# -----------------------------
# CLIPSeg engine (with autocast/half on CUDA)
# -----------------------------
class ClipSegRoad:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.model.eval()
        # Use half precision on GPU
        self.use_autocast = self.device.type == "cuda"
        if self.use_autocast:
            self.model.half()

    @torch.no_grad()
    def predict_prob(self, bgr_roi: np.ndarray, prompts) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            text=prompts,
            images=[pil] * len(prompts),
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        if self.use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(**inputs).logits
        else:
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits)                 # (P, Hm, Wm)
        combined = torch.max(probs, dim=0).values     # max over prompts
        prob = combined.detach().float().cpu().numpy()
        prob = cv2.resize(prob, (bgr_roi.shape[1], bgr_roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        return prob

# -----------------------------
# Main loop
# -----------------------------
def run(video_source, save_out=None, max_width=960):
    global THRESH, IGNORE_BORDER_PCT, FRAME_SKIP

    seg = ClipSegRoad()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    writer = None
    show_mask = True
    paused = False
    mask_hist = deque(maxlen=MASK_SMOOTH_WINDOW)

    fps_t0 = time.time()
    fps_frames = 0
    smoothed_fps = 0.0

    last_smooth_roi = None
    frame_idx = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break

            # Resize for speed
            H, W = frame.shape[:2]
            if W > max_width:
                scale = max_width / float(W)
                frame = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
                H, W = frame.shape[:2]

            # Inset ROI to ignore HUD/borders
            bw = int(W * IGNORE_BORDER_PCT)
            bh = int(H * IGNORE_BORDER_PCT)
            x0, y0, x1, y1 = bw, bh, W - bw, H - bh
            roi = frame[y0:y1, x0:x1]

            # Run model every (FRAME_SKIP+1) frames; reuse last mask otherwise
            run_net = (frame_idx % (FRAME_SKIP + 1) == 0) or (last_smooth_roi is None)
            if run_net:
                prob = seg.predict_prob(roi, prompts=PROMPTS)
                mask_bin = (prob >= THRESH).astype(np.uint8) * 255
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
                mask_hist.append(mask_bin)
            # temporal smoothing (even on reused frames)
            if mask_hist:
                smooth_roi = (np.mean(np.stack(mask_hist, axis=0), axis=0) > 127).astype(np.uint8) * 255
                last_smooth_roi = smooth_roi.copy()
            else:
                smooth_roi = np.zeros(roi.shape[:2], dtype=np.uint8)

            # Fit axis/rect (ROI coords) then map to full-frame
            info = fit_axis_from_mask(last_smooth_roi)
            info_full = None
            conf = float(np.mean((prob if run_net else last_smooth_roi/255.0)[last_smooth_roi > 0])) if np.any(last_smooth_roi > 0) else 0.0

            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[y0:y1, x0:x1] = last_smooth_roi

            if info is not None:
                (cx, cy) = info["center"]
                (p1, p2) = info["axis"]
                rect = info["rect"]
                info_full = {
                    "center": (cx + x0, cy + y0),
                    "axis": ((p1[0] + x0, p1[1] + y0), (p2[0] + x0, p2[1] + y0)),
                    "length": info["length"],
                    "width": info["width"],
                    "angle_deg": info["angle_deg"],
                    "rect": ((rect[0][0] + x0, rect[0][1] + y0), rect[1], rect[2])
                }

            vis = overlay(frame, full_mask, info_full, conf, roi_box=(x0, y0, x1, y1), show_mask=show_mask)

            # FPS
            fps_frames += 1
            now = time.time()
            if now - fps_t0 >= 0.5:
                smoothed_fps = 2.0 * fps_frames / (now - fps_t0)
                fps_frames = 0
                fps_t0 = now
            if SHOW_FPS:
                cv2.putText(vis, f"{smoothed_fps:.1f} FPS", (vis.shape[1]-100, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, f"{smoothed_fps:.1f} FPS", (vis.shape[1]-100, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # Init writer
            if save_out and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps_guess = cap.get(cv2.CAP_PROP_FPS) or MAX_WRITE_FPS_FALLBACK
                writer = cv2.VideoWriter(save_out, fourcc, float(fps_guess), (vis.shape[1], vis.shape[0]))
            if writer:
                writer.write(vis)

            cv2.imshow("Road Detection (CLIPSeg fast) — h for help", vis)
            frame_idx += 1

        # Keys
        key = cv2.waitKey(1 if not paused else 40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('h'):
            print("[Keys] q quit | p pause | m mask | -/=: thr -/+ | [/]: border -/+ | ,/.: frame-skip -/+")
        elif key in (ord('-'), ord('_')):
            THRESH = max(0.05, THRESH - 0.02); print("THRESH =", round(THRESH, 3))
        elif key in (ord('='), ord('+')):
            THRESH = min(0.95, THRESH + 0.02); print("THRESH =", round(THRESH, 3))
        elif key == ord('['):
            IGNORE_BORDER_PCT = max(0.0, IGNORE_BORDER_PCT - 0.01); print("IGNORE_BORDER_PCT =", round(IGNORE_BORDER_PCT, 3))
        elif key == ord(']'):
            IGNORE_BORDER_PCT = min(0.25, IGNORE_BORDER_PCT + 0.01); print("IGNORE_BORDER_PCT =", round(IGNORE_BORDER_PCT, 3))
        elif key == ord(','):
            FRAME_SKIP = max(0, FRAME_SKIP - 1); print("FRAME_SKIP =", FRAME_SKIP)
        elif key == ord('.'):
            FRAME_SKIP = min(9, FRAME_SKIP + 1); print("FRAME_SKIP =", FRAME_SKIP)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Fast road detection with CLIPSeg (zero-shot)")
    ap.add_argument("--video", required=True, help="Video path or camera index (e.g., 0)")
    ap.add_argument("--save", default="", help="Optional path to save annotated MP4")
    ap.add_argument("--max-width", type=int, default=960, help="Resize width for speed")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    src = args.video
    try:
        if src.isdigit(): src = int(src)
    except Exception:
        pass
    out = args.save if args.save.strip() else None
    run(src, save_out=out, max_width=args.max_width)
