import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromONNX(r"./yolov5n.onnx")

def detect_cars(frame, conf_thr=0.30):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1/255, (640, 640))
    net.setInput(blob)
    out = net.forward()[0]
    boxes, scores = [], []
    for row in out:
        obj = row[4]
        if obj < 0.05:
            continue
        cls = int(np.argmax(row[5:]))
        score = float(obj * row[5 + cls])
        if score < conf_thr:
            continue
        if cls != 2:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        x = int(cx / 640 * w); y = int(cy / 640 * h)
        bw_px = int(bw / 640 * w); bh_px = int(bh / 640 * h)
        boxes.append([x - bw_px // 2, y - bh_px // 2, bw_px, bh_px])
        scores.append(score)
    keep = np.array(cv.dnn.NMSBoxes(boxes, scores, conf_thr, 0.45)).reshape(-1)
    return [(boxes[i], scores[i]) for i in keep]

# 実画像で検出された Beetle を切り抜き(実測 box x=655 y=540 w=156 h=70, 960スケール)
beetle = cv.imread(r"./beetle.jpg")
beetle = cv.resize(beetle, (960, int(beetle.shape[0] * 960 / beetle.shape[1])))
car_crop = beetle[540:610, 655:811]

W_CAR = 1.8          # ビートルの実車幅(メートル)
F = 1000.0           # 合成カメラの焦点距離(px) — 実カメラでは §校正 で測定
print("f=%.0f, W=%.2f m" % (F, W_CAR))

for D in [8.0, 10.0, 15.0, 20.0]:
    w_px = int(F * W_CAR / D)
    canvas = np.full((720, 1280, 3), 120, np.uint8)
    crop = cv.resize(car_crop, (w_px, int(w_px * car_crop.shape[0] / car_crop.shape[1])))
    x0 = 600 - w_px // 2
    canvas[300:300 + crop.shape[0], x0:x0 + w_px] = crop
    dets = detect_cars(canvas)
    if not dets:
        print("D=%.1f m -> 検出されず(検出感度不足)" % D)
        continue
    (x, y, w, h), s = dets[0]
    D_est = F * W_CAR / w
    print("D=%.1f m -> w_px=%d, D_est=%.2f m, 誤差=%+.1f%%, conf=%.2f"
          % (D, w, D_est, 100 * (D_est - D) / D, s))

# 地面プラン法(固定カメラ): Z = h_cam * f / (y_bottom - y_horizon)
h_cam = 1.2
y0 = 300          # ホライゾン(消失点)の行
for y_bottom in [360, 420, 600]:
    Z = h_cam * F / (y_bottom - y0)
    print("y_bottom=%d -> Z=%.2f m (h_cam=%.1f, y0=%d)" % (y_bottom, Z, h_cam, y0))
