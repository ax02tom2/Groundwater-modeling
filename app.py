import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import datetime

st.set_page_config(page_title="💧 地下水位模擬補遺工具", layout="wide")
st.title("💧 地下水位模擬與補遺工具")
st.write("上傳您的「雨量資料」與「地下水位資料」，選擇要補遺的區間，系統將利用機器學習模型自動模擬出遺失區段的水位數據。")

# --- 🛠️ 讀取檔案快取 ---
@st.cache_data
def load_raw_data(file_bytes, file_name):
    if file_name.endswith('.csv'):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='big5')
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    return df

# --- 🛠️ 時間與數字清理函數 ---
def clean_and_parse_dates(date_series):
    def normalize_date_string(s):
        s = str(s).strip().replace('"', '').replace("'", "")
        s = s.replace('時', ':').replace('分', ':').replace('秒', '')
        s = s.replace('上午', 'AM ').replace('下午', 'PM ')
        if s in ('nan', 'NaT', 'None', '', 'NaN'):
            return None
        if s.count(':') == 1:
            s += ':00'
        return s
    normalized_series = date_series.apply(normalize_date_string)
    return pd.to_datetime(normalized_series, errors='coerce')

def clean_and_parse_numbers(num_series):
    extracted = num_series.astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
    return pd.to_numeric(extracted, errors='coerce')

# --- 側邊欄：檔案上傳 ---
st.sidebar.header("📁 1. 資料上傳")
rain_file = st.sidebar.file_uploader("上傳雨量資料 (Excel/CSV)", type=["xlsx", "xls", "csv"])
hobo_file = st.sidebar.file_uploader("上傳水位資料 (CSV)", type=["csv"])

if rain_file and hobo_file:
    if rain_file.name == hobo_file.name:
        st.error(f"❌ 檔案上傳錯誤：您在兩個上傳區都選擇了同一個檔案 (`{rain_file.name}`)！")
        st.stop()

    try:
        rain_df_raw = load_raw_data(rain_file.getvalue(), rain_file.name)
        hobo_df_raw = load_raw_data(hobo_file.getvalue(), hobo_file.name)

        st.sidebar.markdown("---")
        st.sidebar.header("🎯 2. 欄位對應設定")
        
        rain_cols = [str(c) for c in rain_df_raw.columns.tolist()]
        def_rain_date = next((i for i, c in enumerate(rain_cols) if c == 'Time.1'), 0)
        def_rain_val = next((i for i, c in enumerate(rain_cols) if 'R1' in c), min(1, len(rain_cols)-1))
        
        rain_date_col = st.sidebar.selectbox("🌧️ 雨量 - 日期欄位", rain_cols, index=def_rain_date)
        rain_val_col = st.sidebar.selectbox("🌧️ 雨量 - 數值欄位", rain_cols, index=def_rain_val)

        hobo_cols = [str(c) for c in hobo_df_raw.columns.tolist()]
        def_hobo_date = 0
        def_hobo_val = min(1, len(hobo_cols)-1)
        
        hobo_date_col = st.sidebar.selectbox("💧 水位 - 日期欄位", hobo_cols, index=def_hobo_date)
        hobo_val_col = st.sidebar.selectbox("💧 水位 - 數值欄位", hobo_cols, index=def_hobo_val)

        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ 3. 模型參數設定")
        model_choice = st.sidebar.selectbox("選擇預測模型", ["隨機森林 (Random Forest) - 推薦", "線性迴歸 (Linear Regression)"])
        
        # --- 新增的說明區塊 ---
        rolling_windows = st.sidebar.multiselect(
            "選擇降雨累積天數 (特徵工程)", 
            options=[1, 3, 5, 7, 14, 20, 30, 60], 
            default=[1, 3, 7, 14, 30]
        )
        
        st.sidebar.info(
            "💡 **為什麼要設定累積天數？**\n\n"
            "地下水位的升降有「延遲效應」，不僅受當天雨量影響，還受過去幾週的降雨影響。勾選天數可讓 AI 學習這層遲滯關係。\n\n"
            "🎯 **建議設定值：**\n"
            "- **淺層/反應快：** `1, 3, 5, 7`\n"
            "- **深層/反應慢：** `14, 30, 60`\n"
            "- **萬用推薦 (預設)：** `1, 3, 7, 14, 30` (兼顧短期與中長期，適用大多數地質)"
        )

        st.sidebar.markdown("---")
        st.sidebar.header("🗓️ 4. 補遺時間區間")
        st.sidebar.write("請透過日曆選取您想進行 AI 補遺的起訖日期：")
        
        default_start = datetime.date(2018, 1, 1)
        default_end = datetime.date.today()
        impute_date_range = st.sidebar.date_input(
            "選擇區間 (開始與結束)",
            value=(default_start, default_end)
        )

        if st.button("🚀 確認無誤，開始執行模擬預測"):
            if len(impute_date_range) != 2:
                st.warning("⚠️ 請在左側日曆中完整點選「開始日期」與「結束日期」。")
                st.stop()
                
            start_date, end_date = impute_date_range

            with st.spinner("正在清洗資料與訓練模型中..."):
                
                # --- 處理雨量與水位 ---
                rain_df = rain_df_raw[[rain_date_col, rain_val_col]].copy()
                rain_df.columns = ['Date', 'Rainfall']
                rain_df['Date'] = clean_and_parse_dates(rain_df['Date']) 
                rain_df['Rainfall'] = clean_and_parse_numbers(rain_df['Rainfall']) 
                rain_df = rain_df.dropna(subset=['Date']).set_index('Date')
                rain_daily = rain_df.resample('D').sum()

                hobo_df = hobo_df_raw[[hobo_date_col, hobo_val_col]].copy()
                hobo_df.columns = ['Date', 'WaterLevel']
                hobo_df['Date'] = clean_and_parse_dates(hobo_df['Date'])
                hobo_df['WaterLevel'] = clean_and_parse_numbers(hobo_df['WaterLevel']) 
                hobo_df = hobo_df.dropna(subset=['Date']).set_index('Date')
                hobo_daily = hobo_df.resample('D').mean()
                
                # --- 資料合併與特徵工程 ---
                df = pd.merge(rain_daily, hobo_daily, left_index=True, right_index=True, how='outer')
                
                for window in rolling_windows:
                    df[f'Rain_{window}D_Sum'] = df['Rainfall'].rolling(window=window, min_periods=1).sum()
                
                df_model = df.dropna(subset=[f'Rain_{w}D_Sum' for w in rolling_windows])
                
                train_data = df_model.dropna(subset=['WaterLevel'])
                
                all_predict_data = df_model[df_model['WaterLevel'].isna()]
                predict_data = all_predict_data.loc[str(start_date) : str(end_date)]
                
                if len(train_data) == 0:
                    st.error("❌ 找不到雨量與水位重疊的時間段，無法訓練模型！")
                    st.stop()
                if len(predict_data) == 0:
                    st.warning(f"⚠️ 在您選擇的區間 ({start_date} ~ {end_date}) 內，沒有需要補遺的資料！")
                    st.stop()

                # --- 訓練與預測 ---
                st.success(f"**✅ 解析成功！** 訓練歷史 `{len(train_data)}` 天，將模擬 `{len(predict_data)}` 天的水位 (區間: {start_date} ~ {end_date})。")
                
                features = [f'Rain_{w}D_Sum' for w in rolling_windows]
                X_train = train_data[features]
                y_train = train_data['WaterLevel']
                X_predict = predict_data[features]
                
                if "隨機森林" in model_choice:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = LinearRegression()
                    
                model.fit(X_train, y_train)
                predicted_levels = model.predict(X_predict)
                
                predict_data_copy = predict_data.copy()
                predict_data_copy['WaterLevel_Simulated'] = predicted_levels
                
                final_df = df.copy()
                final_df['Simulated'] = False
                final_df.loc[predict_data_copy.index, 'WaterLevel'] = predict_data_copy['WaterLevel_Simulated']
                final_df.loc[predict_data_copy.index, 'Simulated'] = True
                
                # --- 繪製上下分離子圖表 (Subplots) ---
                st.markdown("### 📈 地下水位與雨量動態圖")
                
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.08,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("地下水位變化", "日降雨量")
                )
                
                actual_mask = final_df['Simulated'] == False
                fig.add_trace(go.Scatter(x=final_df[actual_mask].index, y=final_df[actual_mask]['WaterLevel'], 
                                         mode='lines', name='實際觀測水位', line=dict(color='blue')),
                              row=1, col=1)
                
                sim_mask = final_df['Simulated'] == True
                fig.add_trace(go.Scatter(x=final_df[sim_mask].index, y=final_df[sim_mask]['WaterLevel'], 
                                         mode='lines', name='AI 模擬補遺水位', line=dict(color='orange', dash='dot')),
                              row=1, col=1)
                
                fig.add_trace(go.Bar(x=final_df.index, y=final_df['Rainfall'], 
                                     name='日雨量', marker_color='rgba(0, 191, 255, 0.7)'),
                              row=2, col=1)

                fig.update_yaxes(title_text="地下水位 (m)", autorange="reversed", row=1, col=1)
                fig.update_yaxes(title_text="日雨量 (mm)", row=2, col=1)
                fig.update_xaxes(title_text="日期", row=2, col=1)

                fig.update_layout(
                    height=750,
                    hovermode="x unified",
                    legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.8)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 檔案下載 ---
                st.markdown("### 📥 下載模擬結果")
                output_df = final_df[['Rainfall', 'WaterLevel', 'Simulated']].rename(
                    columns={'WaterLevel': 'WaterLevel(m)', 'Simulated': 'Is_Simulated_Data'}
                )
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer)
                csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                
                st.download_button(
                    label="📥 下載完整資料集 (含模擬資料).csv",
                    data=csv_bytes,
                    file_name="waterlevel_simulated.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"檔案解析發生不可預期的錯誤：{e}")
