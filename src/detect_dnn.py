import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromONNX(r"./yolov5n.onnx")

def detect_cars(frame, conf_thr=0.30):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1/255, (640, 640))
    net.setInput(blob)
    out = net.forward()          # (1, 25200, 85): cx, cy, w, h, obj, 80 classes
    out = out[0]
    boxes, scores = [], []
    for row in out:
        obj = row[4]
        if obj < 0.05:
            continue
        cls = int(np.argmax(row[5:]))
        score = float(obj * row[5 + cls])
        if score < conf_thr:
            continue
        if cls != 2:             # COCO: car=2
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        x = int(cx / 640 * w); y = int(cy / 640 * h)
        bw_px = int(bw / 640 * w); bh_px = int(bh / 640 * h)
        boxes.append([x - bw_px // 2, y - bh_px // 2, bw_px, bh_px])
        scores.append(score)
    keep = np.array(cv.dnn.NMSBoxes(boxes, scores, conf_thr, 0.45)).reshape(-1)
    return [(boxes[i], scores[i]) for i in keep]

for name in ["car.jpg", "beetle.jpg"]:
    img = cv.imread(r"./\" + name)
    img = cv.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
    for (x, y, w, h), s in detect_cars(img):
        print(name, "car det x=%d y=%d w=%d h=%d conf=%.2f" % (x, y, w, h, s))
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imwrite(r"./dnn_" + name.replace(".jpg", "") + ".png", img)
