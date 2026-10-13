import cv2 as cv
import numpy as np

F = 1000.0; W = 1.8
beetle = cv.imread(r"./beetle.jpg")
beetle = cv.resize(beetle, (960, int(beetle.shape[0] * 960 / beetle.shape[1])))
car_crop = beetle[540:610, 655:811]

frames = []
truth = []
for t in range(20):
    D = 8.0 + t * 0.6
    w_px = int(F * W / D)
    h_px = int(w_px * car_crop.shape[0] / car_crop.shape[1])
    canvas = np.full((720, 1280, 3), 120, np.uint8)
    crop = cv.resize(car_crop, (w_px, h_px))
    x0 = 300 + t * 30
    canvas[300:300 + h_px, x0:x0 + w_px] = crop
    frames.append(canvas)
    truth.append((D, x0, 300, w_px, h_px))

print("TrackerCSRT 使用可能:", hasattr(cv, "TrackerCSRT_create"))
tracker = cv.TrackerCSRT_create()
bx, by, bw, bh = truth[0][1:]
tracker.init(frames[0], (bx, by, bw, bh))

for t in range(1, 20):
    ok, box = tracker.update(frames[t])
    D_true = truth[t][0]
    if not ok:
        print(t, "追跡失敗")
        continue
    D_est = F * W / box[2]
    print("t=%2d D_true=%.1f -> tracker w_px=%d, D_est=%.2f m, 誤差=%+.1f%%"
          % (t, D_true, box[2], D_est, 100 * (D_est - D_true) / D_true))
