# 風洞流線分析（Wind Tunnel Streamline Analysis）

這個專案從風洞實驗的照片與影片中，自動抽取「綠色雷射片光照亮的煙流線」座標，
產生標注圖、逐幀標注影片與量化數據，作為流體力學報告的證據：

- 從照片/影片標出每一條流線的像素座標（CSV / JSON）；
- 用無因次量比較大、小模型（幾何相似、雷諾數、無因次尾流寬 `W/D`）；
- 估計尾流寬度 `W`、表觀流速與流線偏折。

報告的論述框架：

> 把大模型當作近似真實物體，小模型當作縮尺模型，
> 用幾何相似、雷諾數相似、以及 `W/D` 判斷小模型能否代表大模型的流場。

## 快速開始

```bash
# 1. 啟用準備好的虛擬環境（必要套件已裝好：opencv-python、numpy）
source ~/myenv/bin/activate

# 2. 跑第一張圖（全圖抽線，輸出到 complete_img/image1/）
python extract_streamlines.py image1.jpg \
  --out-dir complete_img/image1 \
  --detector ridge --ridge-threshold 6 --ridge-height 31 \
  --min-value 20 --min-saturation 15 --horizontal-ratio 0 \
  --min-length 100 --min-horizontal-span 50 --close-width 9 \
  --stitch-gap 160 --stitch-y-tolerance 7 --stitch-overlap 35 \
  --sample-every 2 --smooth-window 7

# 3. 打開 complete_img/image1/overlay.png 檢查結果
```

調參數時一律看彩色的 `overlay.png`——它畫的是偵測器「實際找到」的線，
比黑底報告圖誠實。

## 主要檔案

| 檔案 | 用途 |
|---|---|
| `extract_streamlines.py` | 核心：單張影像抽流線，也提供偵測函式給影片管線共用 |
| `analyze_video_streamlines.py` | 影片管線：逐幀流線、時間平均流線、尾流寬、光流速度箭頭、標注影片 |
| `scripts/run_all_videos.sh` | 一鍵批次跑 `video_inputs/` 下所有影片 |
| `analyze_similarity.py` | 從 `measurements.csv` 算 `d/D`、`r_out/D`、`r_in/D`、`Re`、`W/D` |
| `extract_streamline_report.py` | 實驗性：黑底報告風格圖（彎曲區與右側水平區分開處理，仍用舊偵測路徑） |
| `steps.md` | 實驗與報告流程筆記 |
| `measurements.csv` | 大小模型尺寸與風速的輸入表 |

## 背後原理（偵測管線怎麼運作）

整條管線用 OpenCV + NumPy 實作，核心想法是：
**煙流線在影像裡是「細而亮的水平亮脊（ridge）」，逐欄找亮度峰值就能精確定位中心線。**

依序五個步驟（都在 `extract_streamlines.py`）：

1. **Ridge response（亮脊強度圖）**
   取綠色通道，用「垂直方向的形態學 opening」（`cv2.morphologyEx`，高度
   `--ridge-height`）估出局部背景——比這個高度細的水平亮線會被抹掉，剩下的就
   是背景。原圖減背景，亮線就被凸顯出來，即使整片背景本來就是綠的也有效。

2. **對比正規化（`--contrast-norm`，預設開）**
   畫面左右背景亮度差很多：右側背景本身就很亮綠，同一條煙線在那裡的「原始對
   比」遠比暗背景區弱（亮度空間被壓縮）。所以把 response 除以局部的亮度餘裕
   （headroom = 255 − 背景亮度），等效於亮區自動降門檻、暗區維持原門檻，一個
   `--ridge-threshold` 就能同時適用全圖。用 `--no-contrast-norm` 可關閉。

3. **逐欄次像素峰值偵測（`--tracer peak`，預設）**
   對每一個影像欄（column），在 response 上找超過 `--ridge-threshold` 的局部
   亮度極大值，再用拋物線內插把 y 座標精確到次像素。這就是每條線在這一欄的中
   心點。

4. **峰值串接成軌跡（slope-predicting tracker）**
   從左到右逐欄處理：每條進行中的軌跡用最近的斜率預測它在下一欄的 y 位置，把
   最接近預測的峰值接上去。這樣即使線又密又彎（例如繞過模型的偏折區），也不
   會跳到隔壁那條線。容許 `--peak-max-gap` 欄的中斷（線被模型擋住等）。
   （舊版做法是「二值 mask → 骨架化 → 追蹤」，保留為 `--tracer skeleton`，
   但精度較差：相鄰線黏住時骨架會歪掉。）

5. **過濾與輸出**
   依長度（`--min-length`）、水平跨度（`--min-horizontal-span`）、水平比
   （`--horizontal-ratio`，dx/dy）過濾雜訊——真煙線又長又連續又偏水平，柱子
   亮斑與玻璃反光則短而彎。剩下的線段用 `--stitch-gap` 等參數跨缺口補綴，再
   平滑、降採樣後輸出。

每次執行的輸出：

| 檔案 | 內容 |
|---|---|
| `overlay.png` | 彩色標注圖（調參數看這張） |
| `streamlines.csv` | 每條線逐點座標（`line_id, x_px, y_px, ...`） |
| `streamlines.json` | 同上，依線分組 |
| `mask.png` | 超過門檻的亮脊像素 |
| `skeleton.png` | 一像素寬中心線 |
| `streamline_map.png` | 黑線報告風格圖（僅供說明用） |

## 單張影像

### 全圖（`complete_img/`）

`complete_img/image1` ~ `complete_img/image5` 是 `image1.jpg` 與
`image2.png`–`image5.png` 的全圖（不加 `--roi`）抽線結果。全圖會看到煙管柱、
百葉窗反光、牆面紋理，所以長度門檻要拉高。

`image1.jpg`（寬 1706 px）用快速開始那組指令；
`image2.png`–`image5.png`（寬約 2900 px，門檻按解析度放大約 1.7 倍）：

```bash
python extract_streamlines.py image2.png \
  --out-dir complete_img/image2 \
  --detector ridge --ridge-threshold 8 --ridge-height 41 \
  --min-value 20 --min-saturation 15 --horizontal-ratio 0 \
  --min-length 150 --min-horizontal-span 80 --close-width 9 \
  --stitch-gap 200 --stitch-y-tolerance 9 --stitch-overlap 40 \
  --sample-every 2 --smooth-window 9
```

### 只看右側重點區（ROI）

只分析模型周圍與右側密集流線區時，用 ROI 裁切可以少看很多雜訊、門檻也能放寬：

```bash
python extract_streamlines.py image1.jpg \
  --out-dir image1_color_tuned_ridge5_len12 \
  --roi 1150,120,556,760 \
  --detector ridge --ridge-threshold 5 --ridge-height 31 \
  --min-value 20 --min-saturation 15 --horizontal-ratio 0 \
  --min-length 12 --min-horizontal-span 8 --close-width 9 \
  --stitch-gap 160 --stitch-y-tolerance 7 --stitch-overlap 35 \
  --sample-every 2 --smooth-window 7
```

ROI 格式是 `--roi x,y,寬,高`（像素）。`1150,120,556,760` 表示從 x=1150 起涵蓋
右側到影像邊界，包含密集水平流線區與模型尾流區。尾流左半被切掉就把 x 調小；
拍到太多無關背景就把 x 調大或縮小高度。

## 參數調整指南

| 參數 | 效果 |
|---|---|
| `--ridge-threshold` 調低 | 線更密（抓到更暗的線），但亮物附近碎片變多 |
| `--min-length` 調高 | 清掉短碎雜訊，但可能刪掉真實的短彎線段 |
| `--horizontal-ratio 2` | 只留 dx/dy ≥ 2 的偏水平線；清反光很有效，太斜的扇形線會被切 |
| `--stitch-y-tolerance` 調高 | 更積極接合斷線，但可能把相鄰兩條線誤接 |
| `--peak-y-tolerance` 調高 | tracker 追線更寬鬆；線很密時不要調太大以免跳線 |
| `--contrast-norm` / `--no-contrast-norm` | 亮背景區抓不到線時確認它是開的；雜訊太多可關掉比較 |

原則：先動 `--ridge-threshold` 和 `--min-length`，一次只動一個，跑完看
`overlay.png` 比較。

## 影片分析（逐幀標注）

影片放在 `video_inputs/*.MOV`。影片管線透過 `detect_streamline_polylines()`
與單張影像共用同一套偵測程式，所以上面的原理與參數全部適用。

### 先跑 smoke test（只處理 5 幀，確認參數）

```bash
source ~/myenv/bin/activate
python analyze_video_streamlines.py video_inputs/IMG_5616.MOV \
  --out-dir video_output_smoke \
  --frame-step 1 --max-frames 5 \
  --roi 0,120,1920,760 \
  --detector ridge --ridge-threshold 7 \
  --min-length 180 --min-horizontal-span 120 --horizontal-ratio 2 \
  --wake-x 1700 --wake-y-range 250,760 \
  --speed-analysis --speed-frame-gap 1 --speed-step 1 \
  --speed-flow-direction left --speed-max-angle-deg 75 \
  --overlay-video video_output_smoke/frame_overlay.mp4 \
  --speed-video video_output_smoke/speed_overlay.mp4
```

ROI `0,120,1920,760`：x 方向全寬標注，y 方向沿用裁掉頂部字幕區與底部地面的
範圍。這組過濾參數每幀約留 115 條長煙線，柱子亮斑和玻璃反光幾乎清光。

### 整批跑所有影片

```bash
bash scripts/run_all_videos.sh           # 已完成的輸出資料夾會自動跳過
FORCE=1 bash scripts/run_all_videos.sh   # 強制全部重跑
```

注意：既有的 `video_output_*` 是舊版骨架偵測器＋只看右側 ROI 跑的；
要用新偵測器重新產生請加 `FORCE=1`。

### 影片輸出

| 輸出 | 意義 |
|---|---|
| `frame_overlay.mp4` | 逐幀流線標注影片（主要成果） |
| `speed_overlay.mp4` | 速度箭頭影片 |
| `frame_streamlines.csv` | 每個處理幀的流線座標 |
| `mean_streamlines.csv` | 時間平均流線 |
| `wake_width_summary.csv` | 尾流寬 `W` 估計 |
| `deflection_summary.csv` | 流線偏折輔助指標 |
| `speed_summary.csv` / `speed_samples.csv` | 光流表觀速度統計 / 逐格樣本 |
| `mean_overlay.png` / `wake_width_overlay.png` / `speed_overlay.png` | 單幀示意圖 |

### 速度方向與單位

- 這批影像的物理流向是「右到左」，所以用 `--speed-flow-direction left`；
  若自由流區箭頭朝右，代表幀對方向或正負號錯了。
- 速度預設單位是 `px/s`；有像素—長度換算時加上：

```bash
  --length-per-px 0.00025 --speed-unit m/s
```

## 相似性分析（報告數據）

填好 `measurements.csv`：

```csv
model,D,d,r_out,r_in,V,rho,mu,W
large,,,,, ,1.2,1.8e-5,
small,,,,, ,1.2,1.8e-5,
```

執行：

```bash
python analyze_similarity.py measurements.csv \
  --out similarity_summary.csv \
  --plot video_output/dimensionless_comparison.png
```

會算出 `d/D`、`r_out/D`、`r_in/D`、`Re = rho*V*D/mu`、`W/D`，
兩列都有值時還會給大小模型的百分比差異。

### 報告解讀順序

1. 用 `d/D`、`r_out/D`、`r_in/D` 檢查幾何相似。
2. 用雷諾數檢查動力相似。
3. 用 `W/D` 與流線形狀檢查流場相似。
4. 三者都接近 → 小模型可合理代表大模型。
5. `Re` 差太多 → 小模型結果只能當定性參考，除非調整風速。

## 注意事項

- 調偵測參數看彩色 `overlay.png`；黑底 `streamline_map.png` 只拿來做報告的
  定性示意，不要宣稱它是純原始量測（水平線有經過重建/延伸）。
- 密集水平煙線用 `--detector ridge` 通常是最穩的起點。
- 在遠端機器（如 ws2）跑之前，記得同步 `extract_streamlines.py` 和
  `analyze_video_streamlines.py` 兩支檔案——影片管線 import 前者。
