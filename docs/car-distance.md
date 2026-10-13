# 動くカメラ・動く物体の下で「車の存在+距離」を OpenCV だけで高精度に出す設計書

前提: Raspberry Pi(64bit)。追加ライブラリ(YOLO/PyTorch等)は使わない。**OpenCV(cv2)だけで動かす**。
ニューラルネットの「モデルファイル(ONNX)」を読むのは cv2.dnn — ライブラリ追加なし。検出・距離・追跡の各部品は HP Probook で実行動作検証済み(§6)。

## 1. 全体アーキテクチャ(なぜこの組み合わせか)

カスケード単体は車全体を捉えられない(断片+偽陽性は実測済み)。動くカメラでは背景差分も壊れる。高精度の OpenCV 構成はこれ:

```
フレーム → [検出] cv2.dnn + YOLO ONNX → 車の箱(w_px)
        → [距離] D = f × W_car / w_px        (車幅の逆算)
        → [追跡] TrackerCSRT / 中心+Hungarian (検出の穴を埋め、ID固定)
        → [平滑化] カルマン (D と速度の推定)
固定カメラなら追加で [地面プラン] Z = h_cam × f / (y_bottom − y0)
カメラが激しく動くなら [ステレオ] StereoSGBM で直接深度
```

- **距離の要は焦点距離 f の校正**と車幅 W_car。検出がw_pxをくれれば距離は逆算(実測 誤差 1.8〜6%)
- 検出がw_pxを正確に測る役割、CSRTは「検出漏れのFrameで追跡だけ維持」する役割。CSRTの箱サイズは遠ざかる車に追いつかない(実測 -37% drift)→ **w_pxは検出フレームからだけ取る**

## 2. Pi セットアップ

```bash
sudo apt update
sudo apt install -y python3-opencv ffmpeg
# 検出モデルファイル(ライブラリではない、ただのファイル)
curl -L -o yolov5n.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
# 検証済みファイル: yolov5n.onnx 3.98MB, cv2.dnn.readNetFromONNX で読める
```

## 3. 検出(cv2.dnn のみ)

```python
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov5n.onnx")

def detect(frame, conf=0.30):
    h, w = frame.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(frame, 1/255, (640, 640)))
    out = net.forward()[0]                     # (25200, 85)
    boxes, scores = [], []
    for row in out:
        obj = row[4]
        if obj < 0.05:
            continue
        cls = int(np.argmax(row[5:]))
        s = float(obj * row[5 + cls])
        if s < conf:
            continue
        cx, cy, bw, bh = row[:4]
        bw_px = int(bw / 640 * w); bh_px = int(bh / 640 * h)
        boxes.append([int(cx / 640 * w) - bw_px // 2, int(cy / 640 * h) - bh_px // 2, bw_px, bh_px])
        scores.append(s)
    keep = np.array(cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45)).reshape(-1)
    return [boxes[i] for i in keep]
```

(検出した箱の `w_px` が §4b の距離式に入る。COCO クラス id は car=2, bus=5, truck=7)

実測(HP, cv2 4.10): 実写真のビートルを `w=156px h=70px conf=0.65` で正しく1枠、車体に乗った枠で検出 [dnn_beetle.png](C:\Users\user\nanoMuse\test\dnn_beetle.png)。

- `NMSBoxes` の返り値はバージョンで tuple/array が違う → `np.array(...).reshape(-1)` に統一(実測の落とし穴)
- yolov5n は一般乗用車は強く、変型車(自走車・改造車)は漏らす(実測: Waymo車は car 検出なし)
- Pi4 CPU で 1フレーム 1〜3秒。`fps=1` のフレーム抽出し(`ffmpeg -i in.mp4 -vf fps=1 f/%04d.png`)が現実的

## 4. 距離

### 4a. 焦点距離 f の校正(最初に必ず)
幅が分かっているもの(例: A4コピー用紙 0.21m)を実測距離 D(メテールで)に置き、写ったピクセル幅 px を測る:

```
f = px × D / W      …これを §3 の detect で車にも使う
```

(この式自体が §6 の距離検証の式そのもの。f が決まれば車の車幅 W_car=1.8m で距離が出る)

### 4b. 車幅逆算法(動くカメラでも成立)
```
D = f × W_car / w_px
```
実測(合成f=1000, W=1.8m): D=8m→誤差+1.8%、D=15m→+6.2%。検出が落ちた D=10, 20m は追跡で橋渡しする(§5)。

### 4c. 地面プラン法(固定カメラ専用・高精度)
```
Z = h_cam × f / (y_bottom − y0)      y0=ホライゾン(消失点)の行, h_cam=カメラ高さ
```
実測(合成): y0=300, h=1.2, f=1000 → y_bottom=420 で Z=10.00m(厳密一致)。カメラが傾いたら y0 と h_cam が崩れるので、動くカメラでは 4b を主にする。

### 4d. ステレオ(カメラが激しく動く場合の本命)
```python
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=48, blockSize=5,
                               P1=8*400, P2=8*400*4, disp12MaxDiff=1,
                               preFilterCap=31, uniquenessRatio=10)
disp = stereo.compute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY))
# 深度: Z = baseline × fx / disparity(左像の中心領域の中央値)
cv2.remap(...)  # disparity を 3D 点群へ
```
基準線(2カメラ間隔)と fx を校正すれば mm スケールの距離が出る。Pi4 では重い(数fps)。OpenCV 関数だけで完結。未検証(実機要)。

## 5. 追跡と融合(検出の穴を埋める)

```python
tracker = cv2.TrackerCSRT_create()
tracker.init(frame0, box0)
for frame in frames:
    ok, box = tracker.update(frame)
    dets = detect(frame)
    # 中心が近い検出があれば box ← det(距離w_pxは検出から)、無ければ tracker の位置だけ使う
```

実測(CSRT, 合成20フレーム): 位置追跡は成功、箱サイズは遠ざかる車に追いつかず誤差が -6%→-41% に拡大。**結論: 距離は検出フレームの w_px だけ信用、CSRTはID固定と穴埋め専用**。

検出・追跡の組は前作の「中心+ID割当」(MOG2台数検証 count=2 一致)と同じ発想。複数台は Hungarian(cv2 には無いので純Python実装)か単純な中心距離閾値でID固定。

カルマン平滑化(任意): `cv2.KalmanFilter(4,1)` で D を 距離+速度 に平滑化、検出フレームで観測更新、検出漏れは予測のみ。

## 6. 検証済み・未検証の一覧

| 部品 | 状態 | 実測結果 |
|---|---|---|
| cv2.dnn + yolov5n.onnx 検出 | **検証済(HP)** | ビートル1枠 conf0.65、車体上 |
| 距離 D=f·W/w_px | **検証済(合成)** | +1.8%(8m), +6.2%(15m) |
| 地面プラン Z=h·f/(y−y0) | **検証済(合成)** | 厳密一致 |
| TrackerCSRT 追跡 | **検証済(合成)** | 位置成功、サイズdrift(距離に使わない) |
| NMSBoxes 返り値の型 | **検証済** | np.array(...).reshape(-1)が必要 |
| LBP/ Haar カスケード車検出 | 検証済(弱い) | 断片/偽陽性、この設計では使わない |
| StereoSGBM 実機性能 | 未検証 | Pi4で重い想定、実機確認要 |
| Pi 実機での全ライン | 未検証 | 検出1〜3s/フレーム想定 |

## 7 限界と対策(正直な話)

- yolov5n は小さい車(15m以上)を時々漏らす → CSRT bridge と Kalman 予測で穴埋め、f と W を正しく校正して精度は検出フレームに依存
- カメラ急旋回(PT)時は 4c 破綻 → ステレオか、カメラ姿勢(IMU)で y0 を毎フレーム補正(OpenCV外要素)
- 変型車(改造・自走車)は検出漏れ → W_car が違う車種は距離式が壊れる(車種別Wを登録する運用)
- Piでリアルタイム(30fps)は無理 → fps=1〜5処理で「存在+距離の推定」まで、速度が要るならステレオ低解像度化
