# OpenCV のみで車の存在と距離を検出する

動くカメラ・動く物体の下で、**追加ライブラリなし(cv2 のみ)**で車を検出し距離を出す検証済みコード集。

- 検出: `cv2.dnn` + YOLOv5n ONNX(モデルファイル1個だけ。PyTorch/ultralytics 不要)
- 距離: 車幅逆算 `D = f·W/w_px`(実測 誤差+1.8%〜+6.2%)、固定カメラなら地面プラン `Z = h·f/(y−y0)`(厳密一致)
- 追跡: `TrackerCSRT`(位置追跡成功、箱サイズはdriftするので距離は検出フレームの w_px のみ採用)
- カスケード(Haar/LBP)車検出は実測で断片・偽陽性のみと確認済み。**車用カスケード単体は実用不可**という結論も同梱

動作検証は HP Probook(Windows, Python 3.11, OpenCV 4.10 contrib)。Raspberry Pi(64bit)では未検証(apt 版 cv2 でも同じ API)。

## 必要もの

```bash
# Python 3.10+ / OpenCV (contrib 版推奨)
pip install opencv-contrib-python numpy
# Raspberry Pi(64bit)なら:
sudo apt install -y python3-opencv ffmpeg

# モデルファイル(このリポジトリに含めない。実行前にダウンロード)
curl -L -o yolov5n.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
# 車用 Haar XML(比較用・非推奨)
curl -L -o cars.xml https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml
```

テスト画像(任意の車写真2枚)を `car.jpg`, `beetle.jpg` としてスクリプトと同じフォルダに置く。

## スクリプト

| ファイル | 何をする | 検証結果 |
|---|---|---|
| `src/detect_dnn.py` | cv2.dnn+ONNX で車検出(クラスid=2) | 実写真のビートルを車体上1枠 conf=0.65 で検出 |
| `src/distance.py` | 車幅逆算で距離+地面プラン式 | D=8m 誤差+1.8%, D=15m +6.2%, 地面プラン厳密一致 |
| `src/tracker_csrt.py` | CSRT追跡の距離driftを実測 | 位置成功、drift -6%→-41%(距離に使わない根拠) |
| `src/cascade_car.py` | 公開Haarカスケードの正直な挙動 | minNeighbors≥3で検出0、=1で断片/偽陽性 |
| `src/mog2_count.py` | 背景差分+ライン越えで台数 | 合成2台→count=2 一致 |

`NMSBoxes` の返り値の型はバージョンで違う → `np.array(...).reshape(-1)` に統一している(落とし穴実測済み)。

## ドキュメント

- `docs/car-distance.md` — 動くカメラで「存在+距離」を出す全体設計(検出→距離→追跡→カルマン、ステレオStereoSGBM オプション)
- `docs/car-cascade.md` — カスケードのみでの車検出(ラズパイ向け、限界と自前訓練手順)

## 検証済み/未検証の一覧

| 部品 | 状態 |
|---|---|
| cv2.dnn+ONNX検出、距離式、地面プラン、CSRT追跡、MOG2台数、NMSBoxes型 | 実行動作検証済(Windows) |
| LBP(4.10以降削除済み) | 実測確認済 |
| StereoSGBM性能、Pi実機全体、リアルタイム性能 | 未検証(実機必要) |

## License

MIT
