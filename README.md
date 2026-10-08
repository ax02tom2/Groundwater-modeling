# 💧 地下水位模擬與補遺工具 (Groundwater Level Imputation Tool)

## 📖 專案概述 (Overview)
本專案為一個基於 Streamlit 開發的 Web 應用程式，專門用於水文工程與地質分析領域。
主要功能為：透過輸入「歷史降雨量 (Rainfall)」與「地下水位 (Water Level)」，利用機器學習演算法 (主要為 Gradient Boosting / Random Forest) 來預測並補遺 (Impute) 儀器故障或遺失時間段的水位資料。

本系統的核心亮點在於結合了**「水文物理特性 (EWMA 指數衰減)」**與**「機器學習」**，並提供了強大的**「後期無縫微調演算法 (Post-processing)」**，解決了純資料驅動模型常見的基準面偏移與波形過平的問題。

---

## 🛠️ 技術疊代與核心邏輯 (Core Architecture & Logic)

為了避免決策樹模型 (Tree-based models) 常見的「階梯狀誤差 (Step artifacts)」、「過度擬合歷史極端值 (Ghost peaks)」以及「長期基準面漂移」，本程式採用了以下核心架構：

### 1. 特徵工程 (Feature Engineering)
* **物理降雨衰減 (EWMA, Exponential Weighted Moving Average):**
  * 棄用單純的移動總和 (Rolling Sum)，改用 `ewm(span=X).mean()`。
  * **目的:** 完美模擬地下水文的物理延滯效應 (剛下雨時水位快速上升，隨後呈指數型緩慢消退)。預設使用 `[14, 30, 60, 90, 180, 365]` 天作為記憶衰減週期。
* **季節週期特徵 (Cyclical Time Encoding):**
  * 棄用絕對時間 (如 `DayOfYear` 或具體日期)，改採正餘弦編碼：`sin(2π*DOY/365.25)` 與 `cos(2π*DOY/365.25)`。
  * **目的:** 讓模型學習平滑的春夏秋冬「季節基準面」，避免模型死背特定日期（如某年 8 月 8 日的颱風暴漲），從而消除無雨時的「幽靈突坡」。

### 2. 機器學習模型 (Modeling)
* 預設採用 **Gradient Boosting Regressor** (`n_estimators=250, max_depth=5, learning_rate=0.05`)。
* 透過現有時間段的 `(Rain_EWMA_*, sin_DOY, cos_DOY)` 預測真實的 `WaterLevel`。

### 3. 後期微調與物理校正 (Post-Processing)
* **振幅放大器 (Amplitude Multiplier):** 針對樹狀模型預測傾於保守 (波峰波谷被平均化) 的特性，計算預測均值後，將離差乘上倍率 (`multiplier`) 強制撐開波動幅度。
* **平滑濾波 (Rolling Mean Smoothing):** 透過居中移動平均 (`center=True`) 消除決策樹在特徵節點切換時產生的微小鋸齒狀雜訊。
* **無縫錨點校正 (Seamless Anchoring):** 
  * 計算遺失區段「預測線」與真實觀測藍線在「起點」與「終點」的落差 (`offset_start`, `offset_end`)。
  * 透過 `np.linspace` 產生漸變的傾斜修正量 (`drift_correction`) 加回預測線。
  * **目的:** 強制紅線頭尾 100% 貼合真實觀測值，完美解決長期預測的整體基準面漂移 (Baseline Drift) 問題。

---

## 📂 程式碼結構解析 (Code Structure)

* **`load_raw_data()`**: 負責讀取 CSV/Excel，支援 `utf-8` 與 `big5` 雙重編碼防呆。
* **`clean_and_parse_dates()`**: 強效時間字串清洗器。能處理帶有中文字元 (上午/下午/時分秒) 以及不規則格式的時間字串，統一轉為 `datetime`。
* **`clean_and_parse_numbers()`**: 利用 Regex (`r'([-+]?\d*\.?\d+)'`) 強制萃取數值，過濾掉感測器錯誤產生的字串雜訊。
* **UI 狀態與即時響應**:
  * 捨棄了 Submit Button。所有的 Streamlit Sidebar 元件 (日期、參數拉桿) 只要一經變動，系統即會自動在背景重新執行特徵工程與模型訓練。
* **視覺化 (Plotly)**:
  * 使用 `make_subplots` 建立上下共用 X 軸的雙圖表 (水位折線圖 + 雨量長條圖)。
  * **UI/UX 優化**: 設定 `hovermode="x unified"` 並強制 `hoverformat="%Y-%m-%d"`，移除 Plotly 預設產生的英文日期標頭，提供純數字、極度乾淨的游標預覽體驗。
  * Y 軸維持數學正向 (未設定 `autorange="reversed"`)，越負代表水位越深。

---

## ⚠️ 跨 AI 協作與維護注意事項 (Notes for AI Assistants)

當其他 AI 閱讀此專案並嘗試進行功能擴充時，請嚴格遵守以下限制：

1. **嚴禁將「絕對時間/日期」加入特徵 (Do NOT use raw Date or DayOfYear as features)**
   * 任何將 `DayOfYear`、`Month` 直白丟進樹狀模型當特徵的行為，都會造成水位在跨月/跨日期間產生不自然的「階梯狀突波」。務必維持目前的 Sin/Cos 週期編碼。
2. **切勿改動 EWMA 的衰減邏輯**
   * 不要使用 `rolling().sum()` 替代 `ewm().mean()`。地下水的物理特性是平滑消退，只有 EWMA 能在「無降雨」期間產生單調遞減的特徵，確保模擬線平順下滑。
3. **無縫錨點 (Seamless Anchoring) 的數學穩定性**
   * 當 `past_actuals` 或 `future_actuals` 缺乏資料時（例如預測區間一直連到資料集最後一天），`offset_end` 會自動等於 `offset_start` 以保持平行位移。修改此段邏輯時需注意邊界條件 (Edge Cases) 會導致 `IndexError`。
4. **關於 Plotly X軸時間字串**
   * 絕對不要把 DataFrame 的 Index 轉成 String (`df.index.strftime()`) 來餵給 Plotly，這會破壞 Plotly X 軸的時間連續性，導致圖表線條交錯毀損。去除英文 hover 的正確解法是設定 `update_xaxes(hoverformat="%Y-%m-%d")`。

---

## 🚀 使用流程 (Usage Flow)
1. 側邊欄上傳「雨量」與「水位」資料檔。
2. 選擇正確的日期與數值欄位。
3. 選擇補遺區間 (自動篩選出該區間內 `WaterLevel` 為 `NaN` 的列進行預測)。
4. 調整「振幅放大器」與「消除鋸齒天數」，圖表會**即時 (Real-time)** 渲染更新。
5. 若為極長區間 (如大於半年)，若發現紅線過度傾斜，可手動關閉「無縫錨點校正」。
6. 點擊下方按鈕匯出補遺完成的 CSV 檔案。