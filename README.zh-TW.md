# Fire Vision Command Center

[English](readme.md) | [繁體中文](README.zh-TW.md)

這是一個以 YOLO 物件偵測、RGB 畫面溫度估算、規則式安全判斷、影片遙測、即時攝影機監控，以及四路攝影機 VCN 疏散流程為核心的火災與煙霧篩檢原型。

目前本專案定位為本機研究與整合原型，適合展示、受控實驗與系統串接；它不是經認證的火災警報器，也不能取代經校正的熱成像攝影機、現場人員判斷或正式消防安全程序。

## 專案內容

- 單張圖片推論：輸出標註圖片、偵測結果、信心度、判斷結果與可解釋文字。
- 影片推論：可選擇決策疊圖，輸出處理後 MP4、預覽圖片，以及風險／溫度遙測資料。
- 即時攝影機或本機影片監控：使用推論 worker、Server-Sent Events（SSE）與最新影格查詢。
- VCN 流程：處理專案內附的四路攝影機範例與疏散地圖合成。
- FastAPI 後端提供的瀏覽器指揮中心介面。
- 已定義溫度資料契約，方便未來接上可輸出實際溫度的輻射式熱成像感測器。

## 示範影像

版本庫內附 `out.gif`，展示目前 Dashboard 與推論流程：

![Fire Vision Command Center 示範影像](out.gif)

## 目前狀態與重要限制

目前偵測器使用 RGB 圖片與專案內附的 YOLO 權重。`FireTemperatureEstimator` 是從畫面推導出的視覺估算值，不是經校正的物理溫度。即時 worker 回報的主機／CPU 溫度只作為診斷資訊，絕不會被當成安全判斷使用的場景溫度。

HTTP API 接受圖片與影片請求中的選填欄位 `sensor_temperature_celsius`。如果有提供，後端會把它視為場景溫度的權威來源，並在回應中標示為 `thermal_sensor`／已校正。這是為了接入真實輻射式熱感測器所預留的整合介面，不是硬體驅動程式。正式部署時應傳入與每一幀同步的熱感測資料，而不是由使用者手動輸入一個全域溫度。

目前的決策引擎是篩檢輔助工具。警報門檻、持續時間、校正、感測器位置、放射率、環境條件、誤報處理與緊急升級流程，仍需要由領域專家驗證。

## 目錄與重要路徑

以下指令以這個 checkout 為例：

```text
C:\Code\FIRE
```

重要路徑如下：

| 路徑 | 用途 |
| --- | --- |
| `AppYOLO/app.py` | FastAPI 應用程式、上傳驗證、路徑邊界、即時 worker、遙測 API 與靜態檔案服務。 |
| `AppYOLO/start.ps1` | 專案啟動腳本，會建立或使用專用的 Python 3.13 環境並啟動 Uvicorn。 |
| `AppYOLO/src/inference/infer.py` | 推論 facade，將圖片／影片工作分派給對應的推論類別。 |
| `AppYOLO/src/inference/image.py` | 單張圖片 YOLO 推論、標註、溫度估算與決策 payload 建立。 |
| `AppYOLO/src/inference/video.py` | 逐幀追蹤、可選決策疊圖、影片輸出與遙測紀錄。 |
| `AppYOLO/src/inference/utils.py` | YOLO 轉換工具與目前的 RGB 溫度估算器。 |
| `AppYOLO/src/decision_engine.py` | 規則式安全決策引擎與可解釋決策軌跡。 |
| `AppYOLO/src/pipeline_service.py` | 原始主流程與 VCN 流程的 service wrapper。 |
| `AppYOLO/src/api_contract.py` | 不依賴外部套件的上傳、SSE、溫度來源與驗證輔助函式。 |
| `AppYOLO/frontend/index.html` | 英文操作指揮中心的 HTML 結構。 |
| `AppYOLO/frontend/app.js` | Dashboard API 呼叫、即時更新、圖表、預覽與狀態處理。 |
| `AppYOLO/frontend/styles.css` | 深色、響應式指揮中心視覺系統。 |
| `AppYOLO/models/release11.pt` | 後端優先載入的預設模型。 |
| `AppYOLO/models/release26.pt` | 備援／替代模型。 |
| `AppYOLO/test/` | 受控測試用的少量圖片、影片與 VCN 攝影機套件。 |
| `AppYOLO/tests/` | API 契約與 Python 執行版本基準的確定性測試。 |
| `data/` | 本機 YOLO 資料集，包含 train、validation、test 圖片與標籤。 |
| `data.example.yaml` | 可攜式的本機 Ultralytics 資料集設定；被忽略的 `data.yaml` 可使用相同內容。 |
| `AppYOLO/outputs/images/` | 標註圖片輸出。 |
| `AppYOLO/outputs/videos/` | 處理後影片輸出。 |
| `AppYOLO/outputs/live_logs/` | 即時 worker 產生的 JSON Lines 遙測紀錄。 |

`AppYOLO/outputs/uploads/` 下的執行期上傳檔案是暫存資料。產生的圖片、影片與即時 log 可能很大，使用清理功能前請先檢查檔案清單。

## 啟動方式（Python 3.13）

### 環境需求

- 目前文件以 Windows PowerShell 為主要操作環境。
- Python 3.13.x 是本專案的執行版本基準。版本庫根目錄的 `.python-version` 記錄 `3.13`，啟動腳本也會強制後端模組使用此版本。
- 安裝與 PyTorch／Ultralytics 相容的 CPU 或 CUDA 環境；專案不會強制安裝 CUDA。
- `AppYOLO/requirements.txt` 中列出的 Python 套件。
- 開啟 JavaScript 的瀏覽器。Dashboard 會從 jsDelivr 載入 Chart.js；離線部署時應改為自行提供或替換該依賴。

### 第一次啟動：建立虛擬環境並安裝依賴

開啟 PowerShell，切到後端目錄。從 `AppYOLO` 啟動可以讓既有推論模組的模型與輸出相對路徑保持一致。

建議使用 `AppYOLO/start.ps1` 啟動。它會在需要時使用 Windows Python launcher 建立 `.venv313`，第一次執行會安裝依賴，並且一定使用 Python 3.13 啟動後端：

```powershell
Set-Location C:\Code\FIRE\AppYOLO
.\start.ps1 -Install
```

要停止 server，回到該 PowerShell 視窗按 `Ctrl+C`。開發時可加上 `-Reload` 啟用自動重新載入。

```powershell
Set-Location C:\Code\FIRE\AppYOLO

py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1

python --version
# 預期：Python 3.13.x

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

如果 PowerShell 阻擋虛擬環境啟動腳本，可以只對目前視窗暫時放寬執行政策後再啟動：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv313\Scripts\Activate.ps1
```

### 已安裝依賴時的快速啟動

```powershell
Set-Location C:\Code\FIRE\AppYOLO
.\.venv313\Scripts\Activate.ps1
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

啟動後開啟：

- Dashboard：<http://127.0.0.1:8000/>
- 互動式 API 文件：<http://127.0.0.1:8000/docs>
- ReDoc API 文件：<http://127.0.0.1:8000/redoc>
- 影片播放器：`http://127.0.0.1:8000/player?path=outputs/videos/<file>.mp4`

開發時若要自動重新載入，可以使用以下指令；但 reload 可能重複初始化 YOLO 模型，資源有限的機器不建議當作預設啟動方式：

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 如何確認後端正常

在另一個 PowerShell 視窗執行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 6
```

看到 `status: "ok"` 且 `model_ready: true`，代表模型已成功載入。若 HTTP server 正常但模型不可用，API 會回傳 `status: "degraded"`，此時可以查看啟動視窗的錯誤訊息與 `/api/health` 內容。

## Dashboard 使用流程

1. 先查看頁首 API 狀態。`READY` 代表模型已載入；`DEGRADED` 代表 HTTP server 還在運作，但推論不可用。
2. 已知圖片使用 **Image inference**。可以填入同步的熱感測器攝氏溫度；留白時使用目前 RGB 估算值。
3. 本機影片使用 **Video inference**。CPU 處理可能需要一段時間；目前選填的感測器溫度會暫時套用到處理序列。
4. 使用 **VCN pipeline** 執行 `AppYOLO/WebCamPackage/` 內附的多攝影機範例。
5. 使用 **Start live monitor** 啟動即時監控。輸入 `0` 使用預設 webcam，或輸入工作區相對影片路徑，例如 `test/dataset/forest1.avi`。可在效能設定調整信心度、跳幀數與最大影像寬度。
6. 查看 **Telemetry and decisions**。溫度旁會顯示來源，避免把 RGB 估算值誤認為經校正的熱感測器讀值。
7. 在 **Generated artifacts** 預覽產生的圖片與影片。**Clean old** 會刪除產出目錄中的舊檔，執行前會再次確認。

## API 使用方式

JSON request body 中的路徑都相對於 `AppYOLO`，而且必須解析到該工作區內。上傳 API 使用 multipart form data，並使用隨機暫存名稱，不信任客戶端檔名。

| 方法與路徑 | 用途 |
| --- | --- |
| `GET /api/health` | 模型就緒狀態、即時 worker 狀態、上傳上限與溫度契約。模型不可用時仍回傳 HTTP `200`，但 `status` 會是 `degraded`。 |
| `GET /api/model/info` | 已載入模型路徑與類別對應。 |
| `POST /api/inference/local` | JSON 圖片請求：`image_path`、`save_annotated`、選填 `sensor_temperature_celsius`。 |
| `POST /api/inference/image` | multipart 圖片上傳：`file`、`save_annotated`、選填 `sensor_temperature_celsius`。 |
| `POST /api/inference/video/local` | JSON 影片請求：`video_path`、`output_video_path`、`with_decision`、選填 `sensor_temperature_celsius`。 |
| `POST /api/inference/video` | multipart 影片上傳：`file`、`with_decision`、選填 `sensor_temperature_celsius`。 |
| `POST /api/pipeline/main/run` | 透過 service layer 執行原始圖片／影片流程。 |
| `POST /api/pipeline/vcn/run` | 執行四路攝影機 VCN 與地圖合成流程。 |
| `GET /api/generated/files?limit=48` | 列出產出檔案的 metadata 與預覽 URL。 |
| `POST /api/generated/files/cleanup?keep_latest=40` | 清理 `AppYOLO/outputs/` 下的舊檔，呼叫前請先確認範圍。 |
| `POST /api/live/start` | JSON body：`source`、`conf`、`frame_skip`、`max_frame_width`。檔案來源必須位於 `AppYOLO` 內。 |
| `POST /api/live/stop` | 停止即時 worker。 |
| `GET /api/live/state` | 目前狀態、最新指標、熱感測診斷資料與有上限的歷史陣列。 |
| `GET /api/live/frame` | 最新的標註 JPEG 影格。 |
| `GET /api/live/events` | 狀態變更與新影格的 SSE stream。 |
| `GET /api/video/thumbnail?path=outputs/videos/<file>.mp4` | 產生影片的第一幀預覽。 |
| `GET /api/video/mjpeg?path=outputs/videos/<file>.mp4&fps=15&loop=true` | 適合瀏覽器預覽影片的 MJPEG stream。 |

### 圖片上傳範例

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/inference/image" `
  -F "file=@C:\Code\FIRE\AppYOLO\test\test_fire.jpg" `
  -F "save_annotated=true" `
  -F "sensor_temperature_celsius=83.4"
```

回應會同時包含 RGB 估算值與實際交給決策引擎的場景溫度：

```json
{
  "vision_temperature_celsius": 410.0,
  "scene_temperature_celsius": 83.4,
  "scene_temperature_source": "thermal_sensor",
  "scene_temperature_calibrated": true,
  "sensor_temperature_celsius": 83.4
}
```

上面的數字只是回應格式示例，不代表範例圖片的實際溫度。

### 本機圖片範例

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/inference/local" `
  -H "Content-Type: application/json" `
  -d '{"image_path":"test/test_fire.jpg","save_annotated":true}'
```

### 即時監控範例

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/live/start" `
  -H "Content-Type: application/json" `
  -d '{"source":"test/dataset/forest1.avi","conf":0.25,"frame_skip":2,"max_frame_width":1280}'
```

接著查詢 `GET /api/live/state`、查看 `GET /api/live/frame`，或訂閱 `GET /api/live/events`。

## 資料集

本機資料集遵循 Ultralytics YOLO 目錄結構：

```text
data/
├── train/
│   ├── images/   14,122 files
│   └── labels/   14,122 files
├── val/
│   ├── images/    3,099 files
│   └── labels/    3,099 files
└── test/
    ├── images/    4,306 files
    └── labels/    4,306 files
```

資料集盤點：共 21,527 張圖片與 21,527 個標籤檔。標籤是正規化的 YOLO rows，格式為 `class_id x_center y_center width height`。目前類別對應如下：

```text
0 = smoke（煙霧）
1 = fire（火焰）
```

部分標籤檔是空的，代表背景／無物件樣本。被忽略的 `data.yaml` 可能仍包含舊環境的 Kaggle 絕對路徑（`/kaggle/working/D Fire Dataset`）。在本機操作時，請使用版本庫內的 `data.example.yaml`，或使用相對於此 checkout 的等效設定：

```yaml
path: .
train: data/train/images
val: data/val/images
test: data/test/images
names:
  0: smoke
  1: fire
```

資料集原先參考 [Smoke/Fire Detection YOLO on Kaggle](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo?resource=download)。原始範例程式參考 [johnmartinsson/fire-event-detection-dataset](https://github.com/johnmartinsson/fire-event-detection-dataset)。Web service 不會下載或修改資料集。

## VCN 攝影機套件

VCN 範例使用 `AppYOLO/WebCamPackage/` 及以下攝影機地圖：

```json
{
  "left1.png": "100,100",
  "mid.jpg": "500,100",
  "right1.png": "850,100",
  "right2.jpg": "700,200"
}
```

可以從 Dashboard 執行，或呼叫：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/pipeline/vcn/run" `
  -H "Content-Type: application/json" `
  -d '{}'
```

## 系統架構

```text
瀏覽器 Dashboard
      │  REST + SSE + JPEG/MJPEG
      ▼
FastAPI 應用程式（AppYOLO/app.py）
      ├── 路徑與上傳邊界
      ├── 圖片／影片／local／pipeline endpoints
      ├── Live worker + 有上限的遙測歷史
      └── 靜態前端與產出檔案服務
      │
      ▼
YOLOInfer facade
      ├── ImageInfer ── YOLO + RGB 溫度估算
      └── VideoInfer ── 追蹤 + 選用決策疊圖
      │
      ▼
SafetyDecisionEngine
      └── 風險分數 + 警報旗標 + 建議動作 + 可解釋資訊

選用／未來輸入
輻射式熱成像感測器 ── 同步場景溫度／熱影像
                         │
                         └── temperature contract ──► 決策融合
```

即時 worker 目前刻意限制為單一程序內的一個 worker，且只保留固定大小的歷史視窗。它適合本機原型，尚未支援多使用者排程、耐久化工作佇列或水平擴充部署。

## 溫度契約與熱成像感測器規劃

### 目前契約

每個可以進行決策的圖片／影片結果，都應區分以下欄位：

- `scene_temperature_celsius`：實際提供給決策引擎的值。
- `scene_temperature_source`：`thermal_sensor`、`rgb_estimate` 或 `ambient_fallback`。
- `scene_temperature_calibrated`：該值是否來自經校正的輻射式來源。
- `vision_temperature_celsius`：目前從 RGB 推導的估算值，保留作比較。
- `host_temperature_celsius`／`system_temperature_celsius`：只作主機診斷；絕不能當成場景溫度。

### 建議的正式整合設計

下一個最重要的硬體步驟，是使用能輸出實際溫度資料的輻射式熱成像感測器，而不只是彩色化預覽影像。保留 RGB 攝影機負責空間與情境辨識，使用熱影像負責溫度量測：

1. 新增 `ThermalSensorAdapter`，輸出原始／輻射式影像、校正 metadata、感測器 ID、時間戳、單位與有效溫度範圍。
2. 用共用的 `frame_id` 與 timestamp 同步 RGB 與熱影像。在把數值用於警報規則前，先完成感測器校正，並定義放射率與環境假設。
3. 對齊 RGB 與熱影像，把每個 YOLO 火焰／煙霧 bounding box 投影到熱影像，計算區域最大值、P95、平均值、超過門檻的面積與升溫速率等統計。
4. 在決策引擎融合視覺信心度、熱影像統計、時間持續性與感測器健康狀態。加入 hysteresis／debounce，避免單一雜訊影格觸發或解除警報。
5. 每次決策都保存來源追蹤資料：感測器 ID、校正版號、熱影像 frame ID、RGB frame ID、區域統計與決策軌跡。
6. 以受控的高溫物體／火焰／煙霧場景、負樣本、日夜、反光表面、遮擋、距離變化與感測器故障情境驗證。不能只靠模型信心度宣稱安全認證。

未來每幀資料可以採用以下 envelope：

```json
{
  "frame_id": 1042,
  "timestamp": "2026-09-08T00:00:00Z",
  "rgb_image": "frames/rgb/1042.jpg",
  "thermal": {
    "frame_path": "frames/thermal/1042.radiometric",
    "temperature_unit": "celsius",
    "max_temperature_celsius": 83.4,
    "p95_temperature_celsius": 79.1,
    "calibrated": true,
    "sensor_id": "thermal-01"
  },
  "detections": [
    {"class_name": "fire", "confidence": 0.92, "bbox": [0.41, 0.34, 0.22, 0.31]}
  ]
}
```

應用程式最終應接受上述 envelope 或 sensor-stream adapter，而不是單一全域溫度欄位。目前的選填表單欄位是刻意保留的最小整合步驟，主要用於整合測試。

## 設定

後端支援以下選填環境變數：

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `FIRE_MAX_IMAGE_UPLOAD_BYTES` | `15728640`（15 MiB） | 圖片串流上傳大小上限。 |
| `FIRE_MAX_VIDEO_UPLOAD_BYTES` | `262144000`（250 MiB） | 影片串流上傳大小上限。 |
| `FIRE_ALLOWED_ORIGINS` | `http://127.0.0.1:8000,http://localhost:8000` | UI 分離部署時使用的逗號分隔 CORS origin。 |

範例：

```powershell
$env:FIRE_MAX_VIDEO_UPLOAD_BYTES = "524288000"
$env:FIRE_ALLOWED_ORIGINS = "http://127.0.0.1:8000"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## 驗證指令

以下指令從 `C:\Code\FIRE\AppYOLO` 使用 Python 3.13 執行：

```powershell
py -3.13 -m unittest discover -s tests -v
py -3.13 -m py_compile app.py src/api_contract.py src/inference/infer.py src/inference/image.py src/inference/video.py
node --check frontend/app.js
```

前兩項是確定性的原始碼檢查。真正的推論執行仍取決於本機 Python 環境、模型載入、攝影機／影片 codec 與硬體。成功 import 或收到 HTTP `200`，不代表正式部署或外部熱感測器已經連線。

## 參考資料

- [Smoke/Fire Detection YOLO dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo?resource=download)
- [Fire event detection sample code](https://github.com/johnmartinsson/fire-event-detection-dataset)
- [VisiFire demo](http://signal.ee.bilkent.edu.tr/VisiFire/Demo/)
- [Ultralytics documentation](https://docs.ultralytics.com/)
