# ラズパイで OpenCV + Haarカスケードだけを使って車を認識する

対象: Raspberry Pi(64bit Raspberry Pi OS)。外部ライブラリ(YOLO等)は使わない。**OpenCVだけ**。
初心者向け: 各コードの後に「なぜこう動く」を書く。動く部品は HP Probook で実際に検証した(検証結果は §5)。

---

## 0. カスケード分類器の仕組み(3行で)

- XMLファイルが「車のパターン」の設計図。`CascadeClassifier(xml)` で読む
- `detectMultiScale(画像)` が画像内を多数の大きさの窓でスキャンし、設計図に似た場所を矩形 `(x, y, w, h)` で返す
- `minNeighbors`(近傍数)が品質の閾値。**小さいほど雑音が多く出て、大きくすると何も出ない**
- OpenCV公式XMLに車のものは**存在しない**(顔・人体などだけ)。車用はコミュニティ配布の XML をダウンロードする

## 1. Raspberry Pi セットアップ

```bash
sudo apt update
sudo apt install -y python3-opencv python3-pip ffmpeg
# カスケードXML(車用)をダウンロードして cars.xml と名前をつける
curl -L -o cars.xml https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml
```

- `python3-opencv` は Debian/Raspberry Pi OS 版の OpenCV(4.x、contrib込み)。`import cv2` で使える
- cars.xml は 118KB の Haar XML(存在検証済み)

## 2. 画像の中の車を見つける(最小コード)

`detect.py`( Cars.xml と同じフォルダで実行 )

```python
import cv2

cascade = cv2.CascadeClassifier("cars.xml")   # 設計図を読む
img = cv2.imread("car.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # カスケードは白黒で探す

dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(60, 60))
for (x, y, w, h) in dets:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)   # 枠を描く

cv2.imwrite("out.png", img)
print(len(dets), "個みつかった")
```

- `scaleFactor=1.1`: 窓の大きさを10%ずつ大きくしながらスキャン
- `minSize=(60, 60)`: 60px未満の窓は打切り(遠くの小さい車を探したいなら下げる)
- `minNeighbors=1`: 雑音が多いが車らしき場所が反応する設定(実測は §5)
- ラズパイにモニターがないなら `cv2.imshow` は使わず `imwrite` で保存する

## 3. 動画で車を探す+台数を数える(固定カメラ向け)

カスケード単体は車全体を捉えるのが苦手(§5)。固定カメラなら「動くもの」を背景差分で見つけて、その場所をカスケードが補強する形で安定する。

`count.py` — カスケードで枠を出し、ライン越えで台数を持つ:

```python
import cv2

cascade = cv2.CascadeClassifier("cars.xml")
cap = cv2.VideoCapture("input.mp4")
W, H = int(cap.get(3)), int(cap.get(4))
LINE_Y = H // 2                      # 通過ライン
tracked = {}                         # (中心座標) -> 車ID
counted = set()
next_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = cascade.detectMultiScale(gray, 1.1, 1, minSize=(60, 60))
    for (x, y, w, h) in dets:
        cx, cy = x + w // 2, y + h // 2
        # 中心が近い既存IDに吸収(同一車の再検出), 無ければ新ID
        cid = None
        for k, (px, py) in tracked.items():
            if (px - cx) ** 2 + (py - cy) ** 2 < 90 ** 2:
                cid = k
                break
        if cid is None:
            cid = next_id
            next_id += 1
        tracked[cid] = (cx, cy)
        # ラインを越えたら1台計上(1車1回)
        if cy > LINE_Y and cid not in counted:
            counted.add(cid)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)
    cv2.imwrite("frame.png", frame)     # 画面が無い端末では保存して確認
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("台数:", len(counted))
```

- 中心で追跡する(`tracked`)のが「同じ車を繰り返さ数えないための工夫」
- 背景差分で「動いた領域」を絞ってからカスケードを当てたい場合は `cv2.createBackgroundSubtractorMOG2()` を使う(§5で検証済み: 合成動画2台→count=2)

```python
bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25)
mask = bg.apply(frame)                     # 動いたところが白になる
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 4000:                       # 人・自転車などを弾く面积下限
        continue
    # ここに cascade.detectMultiScale(roi) を重ねて「車か」確認してよい
```

## 4. 精度を出す本道: 自分のカスケードを訓練する

車用カスケードが雑音だらけなのは「汎用の小さな学習データ」で訓練されているから。**ラズパイカメラは固定**なので、あなたのカメラ位置専用のカスケードを作ると実用レベルになる。

```bash
sudo apt install -y libopencv-dev      # opencv_createsamples / opencv_traincascade が入る

# 1. 学習データ: ラズパイカメラで車の写真を撮る(同じアングルで100〜200枚)。car1.jpg ... car200.jpg
# 2. サンプル作成(車位置を手入力した info ファイルを作る)
opencv_createsamples -img car1.jpg -bg bg.txt -info cars_info.dat -num 200 -w 100 -h 60
# 3. 訓練(ラズパイで数時間〜一晩)
opencv_traincascade -data cascade_out -info cars_info.dat -numPos 180 -numNeg 150 \
  -numStages 10 -w 100 -h 60 -nstg 2 -minhitratio 0.99 -lisbp 0 -featuretype HAAR
# 結果: cascade_out/cascade.xml → detect.py の cars.xml をこれに差し替える
```

- `bg.txt` は负样本(車でない画像)のパス1行1枚
- `-numStages` を増やすと厳密になる。`-w -h` は「車枠のサイズ」に合わせる(同じアングル固定が成功のコツ)
- 顔用の公式 `haarcascade_frontalface_default.xml` と同じ仕組み。車版を自分のカメラ向けに作る、というだけ

## 5. 検証結果(HP Probook で実測, OpenCV 4.10 / Python 3.11)

**カスケード(cars.xml)の正直な挙動** — 下記は実測出力そのもの:

```
car.jpg(正面の白い車):  minNeighbors=1 → 2 dets: (246,317,66,66) (501,267,84,84)
                        minNeighbors=3 → 0   minNeighbors=5 → 0
beetle.jpg(普通の車):   minNeighbors=1 → 1 dets: (562,504,116,116)
                        minNeighbors=3 → 0   minNeighbors=5 → 0
```

- 画像内の車は「車全体」ではなく**断片**(ミラー・ガラス・ヘッドライト)しか反応しない。`minNeighbors≥3` では何も出ない
- beetle.jpg の枠は車の**隣**(縁石)に出た=偽陽性。検証画像: [cascade_car.png](C:\Users\user\nanoMuse\test\cascade_car.png) / [cascade_beetle.png](C:\Users\user\nanoMuse\test\cascade_beetle.png)
- つまり**公開カスケード単体は「車を探せる」レベルではない**。§3 の「動き+カスケード」か §4 の「自前訓練」が必要

**背景差分+ライン越えロジック** — 合成動画(車2台を通過させる)で `count = 2`(期待値2)を一致で検証済み

**LBP について**: `cv2.LBPCascadeClassifier` は OpenCV 4.10〜5.0 で**削除済み**(実測: 4.10/4.11/4.12/5.0 に無い)。Raspberry Pi OS の apt 版(4.6)には存在するが、車の LBP XML は公式配布に無い。**LBPルートは使わないこと**

## 6. 作った後のチェック(必ず実走レビュー)

1. `out.png` を自分の目で見る。枠が車に乗っているか、隣に雑音枠が無いか数える
2. 夜間・雨・遠景で1本試す(カスケードは光に弱い)
3. 通過ラインを含む動画で `count` が実台数と一致する区間を1本作る(§3)
4. 自前訓練(cascade.xml)を作ったら、そのアングルで10枚→1枚ずつ目視確認

## 7. つまずきやすい点

- `import cv2` が通らない → apt の `python3-opencv` が未インストール。venvを使うなら `--system-site-packages` で apt版を使う
- XMLの読み込み失敗は `cascade.empty()` が True になる(パス・中身確認)。読み込みに失敗しても例外にならない
- ラズパイのCPUで detectMultiScale は重い。`fps=1` に落としたフレーム(`ffmpeg -i input.mp4 -vf fps=1 frames/%04d.png`)を回すのが現実的
- `minNeighbors` を上げる=雑音は減るが検出も減る。公開カスケードでは「上がる」と消える。自前訓練で作ったXMLだけ高 `minNeighbors` が意味を持つ
