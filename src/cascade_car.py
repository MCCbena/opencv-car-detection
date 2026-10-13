import cv2 as cv

cascade = cv.CascadeClassifier(r"./cars.xml")

for name in ["car.jpg", "beetle.jpg"]:
    img = cv.imread(r"./\" + name)
    img = cv.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for minN in [1, 3, 5]:
        dets = cascade.detectMultiScale(gray, 1.1, minN, minSize=(60, 60))
        print(name, f"minNeighbors={minN} -> {len(dets)} dets")
        for (x, y, w, h) in dets:
            print("  det", x, y, w, h)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imwrite(r"./cascade_" + name.replace(".jpg", "") + ".png", img)
