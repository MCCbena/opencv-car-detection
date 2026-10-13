import cv2 as cv
import numpy as np

W, H = 640, 480
LINE_Y = H // 2

def run(frames):
    bg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25)
    tracked = {}
    counted = set()
    next_id = 0
    for frame in frames:
        mask = bg.apply(frame)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN,
                               cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            if w * h < 4000:
                continue
            cx, cy = x + w // 2, y + h // 2
            best = None
            for k, (px, py) in tracked.items():
                if (px - cx) ** 2 + (py - cy) ** 2 < 90 ** 2:
                    best = k
                    break
            if best is None:
                tracked[next_id] = (cx, cy)
                best = next_id
                next_id += 1
            else:
                tracked[best] = (cx, cy)
            if cy > LINE_Y and best not in counted:
                counted.add(best)
    return len(counted)

frames = []
for t in range(300):
    f = np.zeros((H, W, 3), np.uint8)
    f[:20, :] = (200, 200, 200)     # 空
    f[200:, :] = (60, 90, 140)      # 地面
    if t > 50:
        x = min(60 + t * 4, W - 100)
        cv.rectangle(f, (x, 300), (x + 100, 380), (0, 120, 250), -1)   # 車1
    if t > 150:
        x = min(60 + (t - 150) * 4, W - 100)
        cv.rectangle(f, (x, 310), (x + 90, 385), (200, 60, 60), -1)    # 車2
    frames.append(f)

print("expected 2, count:", run(frames))
