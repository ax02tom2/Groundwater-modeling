import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import io

st.set_page_config(page_title="💧 地下水位模擬補遺工具", layout="wide")
st.title("💧 地下水位模擬與補遺工具")
st.write("上傳您的「雨量資料」與「地下水位資料」，系統將利用機器學習模型自動模擬出遺失區段的水位數據。")

@st.cache_data
def load_raw_data(file):
    if file.name.endswith('.csv'):
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='big5')
    else:
        df = pd.read_excel(file)
    return df

# --- 🛠️ 終極版時間格式統一器 (解決所有版本衝突與秒數斷層) ---
def normalize_date_string(s):
    s = str(s).strip()
    # 1. 替換掉礙事的中文字
    s = s.replace('時', ':').replace('分', ':').replace('秒', '')
    s = s.replace('上午', 'AM ').replace('下午', 'PM ')
    
    if s in ('nan', 'NaT', 'None', ''):
        return s
        
    # 2. 關鍵防呆：如果字串裡只有一個冒號 (代表只有時、分，缺秒數)
    if s.count(':') == 1:
        s += ':00'  # 強制補上秒數，確保全檔案格式統一
        
    return s

def clean_and_parse_dates(date_series):
    # 先統一字串格式，再交給 pandas 轉換，完全避開 format='mixed' 的版本問題
    normalized_series = date_series.apply(normalize_date_string)
    return pd.to_datetime(normalized_series, errors='coerce')

# --- 側邊欄：檔案上傳 ---
st.sidebar.header("📁 1. 資料上傳")
rain_file = st.sidebar.file_uploader("上傳雨量資料 (Excel/CSV)", type=["xlsx", "xls", "csv"])
hobo_file = st.sidebar.file_uploader("上傳 HOBO 水位資料 (CSV)", type=["csv"])

if rain_file and hobo_file:
    try:
        rain_df_raw = load_raw_data(rain_file)
        hobo_df_raw = load_raw_data(hobo_file)

        st.sidebar.markdown("---")
        st.sidebar.header("🎯 2. 欄位對應設定")
        
        # --- 雨量欄位自動預選 ---
        rain_cols = [str(c) for c in rain_df_raw.columns.tolist()]
        # 強制優先尋找 Time.1 (這欄才有 2026 年的資料)
        def_rain_date = next((i for i, c in enumerate(rain_cols) if c == 'Time.1'), 0)
        def_rain_val = next((i for i, c in enumerate(rain_cols) if 'R1' in c), min(1, len(rain_cols)-1))
        
        rain_date_col = st.sidebar.selectbox("🌧️ 雨量 - 日期欄位 (請選 Time.1)", rain_cols, index=def_rain_date)
        rain_val_col = st.sidebar.selectbox("🌧️ 雨量 - 數值欄位 (請選 R1)", rain_cols, index=def_rain_val)

        # --- 水位欄位自動預選 ---
        hobo_cols = [str(c) for c in hobo_df_raw.columns.tolist()]
        def_hobo_date = 0
        def_hobo_val = min(1, len(hobo_cols)-1)
        
        hobo_date_col = st.sidebar.selectbox("💧 水位 - 日期欄位", hobo_cols, index=def_hobo_date)
        hobo_val_col = st.sidebar.selectbox("💧 水位 - 數值欄位", hobo_cols, index=def_hobo_val)

        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ 3. 模型參數設定")
        model_choice = st.sidebar.selectbox("選擇預測模型", ["隨機森林 (Random Forest) - 推薦", "線性迴歸 (Linear Regression)"])
        rolling_windows = st.sidebar.multiselect(
            "選擇降雨累積天數 (特徵工程)", 
            options=[1, 3, 5, 7, 14, 20, 30, 60], 
            default=[1, 3, 7, 14, 30]
        )

        if st.button("🚀 確認欄位無誤，開始執行模擬預測"):
            with st.spinner("正在清洗資料與訓練模型中..."):
                
                # --- 處理雨量資料 ---
                rain_df = rain_df_raw[[rain_date_col, rain_val_col]].copy()
                rain_df.columns = ['Date', 'Rainfall']
                rain_df['Date'] = clean_and_parse_dates(rain_df['Date']) 
                rain_df = rain_df.dropna(subset=['Date'])
                rain_df.set_index('Date', inplace=True)
                rain_df['Rainfall'] = pd.to_numeric(rain_df['Rainfall'], errors='coerce') 
                rain_daily = rain_df.resample('D').sum()

                # --- 處理水位資料 ---
                hobo_df = hobo_df_raw[[hobo_date_col, hobo_val_col]].copy()
                hobo_df.columns = ['Date', 'WaterLevel']
                hobo_df['Date'] = clean_and_parse_dates(hobo_df['Date'])
                hobo_df = hobo_df.dropna(subset=['Date'])
                hobo_df.set_index('Date', inplace=True)
                hobo_df['WaterLevel'] = pd.to_numeric(hobo_df['WaterLevel'], errors='coerce')
                hobo_daily = hobo_df.resample('D').mean()
                
                # --- 資料合併與特徵工程 ---
                df = pd.merge(rain_daily, hobo_daily, left_index=True, right_index=True, how='outer')
                
                for window in rolling_windows:
                    df[f'Rain_{window}D_Sum'] = df['Rainfall'].rolling(window=window, min_periods=1).sum()
                
                df_model = df.dropna(subset=[f'Rain_{w}D_Sum' for w in rolling_windows])
                
                train_data = df_model.dropna(subset=['WaterLevel'])
                predict_data = df_model[df_model['WaterLevel'].isna()]
                
                st.markdown(f"**📊 資料統計：** 訓練用歷史天數 `{len(train_data)}` 天 │ 待模擬補遺天數 `{len(predict_data)}` 天")
                
                if len(train_data) == 0:
                    st.error("❌ 找不到雨量與水位重疊的時間段！")
                    st.stop()
                if len(predict_data) == 0:
                    st.warning("⚠️ 所有雨量對應的日子都有水位資料，無需補遺！")
                    st.stop()

                # --- 訓練與預測 ---
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
                
                # --- 繪圖 ---
                st.markdown("### 📈 地下水位歷史與模擬預測圖")
                fig = go.Figure()
                
                actual_mask = final_df['Simulated'] == False
                fig.add_trace(go.Scatter(x=final_df[actual_mask].index, y=final_df[actual_mask]['WaterLevel'], 
                                         mode='lines', name='實際觀測水位', line=dict(color='blue')))
                
                sim_mask = final_df['Simulated'] == True
                fig.add_trace(go.Scatter(x=final_df[sim_mask].index, y=final_df[sim_mask]['WaterLevel'], 
                                         mode='lines', name='AI 模擬補遺水位', line=dict(color='orange', dash='dot')))
                
                fig.add_trace(go.Bar(x=final_df.index, y=final_df['Rainfall'], 
                                     name='日雨量', yaxis='y2', marker_color='rgba(0, 191, 255, 0.5)'))

                fig.update_layout(
                    xaxis=dict(title='日期', rangeslider=dict(visible=True), type="date"),
                    yaxis=dict(title='地下水位 (m)', autorange="reversed"),
                    yaxis2=dict(title='日雨量 (mm)', overlaying='y', side='right', autorange="reversed"),
                    legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.8)'),
                    height=600,
                    hovermode="x unified"
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
                    file_name="hobo_waterlevel_simulated.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"檔案解析失敗，錯誤訊息：{e}")
