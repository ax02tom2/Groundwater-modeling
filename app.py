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
st.write("上傳您的「雨量資料」與「地下水位資料」，調整左側參數即可**即時自動模擬**遺失區段的水位數據。")

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

st.sidebar.header("📁 1. 資料上傳")
rain_file = st.sidebar.file_uploader("上傳雨量資料 (Excel/CSV)", type=["xlsx", "xls", "csv"])
hobo_file = st.sidebar.file_uploader("上傳水位資料 (CSV)", type=["csv"])

if rain_file and hobo_file:
    if rain_file.name == hobo_file.name:
        st.error(f"❌ 檔案上傳錯誤：您在兩個上傳區都選擇了同一個檔案！")
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
        st.sidebar.header("⚙️ 3. 模型與特徵設定")
        
        model_choice = st.sidebar.selectbox("預測模型", ["隨機森林 (Random Forest) - 推薦", "線性迴歸 (Linear Regression)"])
        
        rolling_windows = st.sidebar.multiselect(
            "降雨累積天數 (特徵)", 
            options=[1, 3, 7, 14, 30, 60, 90, 180, 365], 
            default=[1, 3, 7, 14, 30, 60, 90, 180, 365]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ 4. 預測結果後期微調")
        
        amplitude_multiplier = st.sidebar.slider(
            "🚀 振幅放大器 (強制撐開波峰波谷)", 
            min_value=0.5, max_value=3.0, value=1.0, step=0.1
        )
        
        smoothing_days = st.sidebar.slider("消除鋸齒平滑天數", min_value=1, max_value=14, value=5)

        st.sidebar.markdown("---")
        st.sidebar.header("🗓️ 5. 補遺時間區間")
        default_start = datetime.date(2018, 1, 1)
        default_end = datetime.date(2026, 9, 22)
        impute_date_range = st.sidebar.date_input("選擇區間 (開始與結束)", value=(default_start, default_end))
        validation_mode = st.sidebar.checkbox("🧪 啟動驗證模式 (隱藏實際資料測試準確度)", value=False)

        if len(impute_date_range) != 2:
            st.warning("⚠️ 請在左側日曆中完整點選「開始日期」與結束日期。")
            st.stop()
            
        start_date, end_date = impute_date_range

        # --- 資料清洗與對齊 ---
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
        
        df = pd.merge(rain_daily, hobo_daily, left_index=True, right_index=True, how='outer')
        original_df = df.copy()
        
        if validation_mode:
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            df.loc[mask, 'WaterLevel'] = np.nan
        
        features = []
        for window in rolling_windows:
            feat_name = f'Rain_{window}D_Sum'
            df[feat_name] = df['Rainfall'].rolling(window=window, min_periods=1).sum()
            features.append(feat_name)
        
        df['DayOfYear'] = df.index.dayofyear
        features.append('DayOfYear')
        
        df_model = df.dropna(subset=features)
        train_data = df_model.dropna(subset=['WaterLevel'])
        predict_data = df_model.loc[str(start_date) : str(end_date)]
        predict_data = predict_data[predict_data['WaterLevel'].isna()]
        
        if len(train_data) == 0 or len(predict_data) == 0:
            st.warning("⚠️ 查無需要預測的區間，或無足夠訓練資料。請調整左側的日曆區間。")
            st.stop()

        # --- 訓練與預測 ---
        X_train = train_data[features]
        y_train = train_data['WaterLevel']
        X_predict = predict_data[features]
        
        if "隨機森林" in model_choice:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)
        predicted_levels = model.predict(X_predict)
        
        if amplitude_multiplier != 1.0 and len(predicted_levels) > 0:
            pred_mean = predicted_levels.mean()
            predicted_levels = pred_mean + (predicted_levels - pred_mean) * amplitude_multiplier
        
        predict_data_copy = predict_data.copy()
        predict_data_copy['WaterLevel_Simulated'] = predicted_levels
        
        if smoothing_days > 1:
            predict_data_copy['WaterLevel_Simulated'] = predict_data_copy['WaterLevel_Simulated'].rolling(
                window=smoothing_days, min_periods=1, center=True
            ).mean()
        
        if validation_mode:
            val_compare = pd.DataFrame({
                'Actual': original_df.loc[predict_data_copy.index, 'WaterLevel'],
                'Predicted': predict_data_copy['WaterLevel_Simulated']
            }).dropna()
            if len(val_compare) > 0:
                mae = np.abs(val_compare['Actual'] - val_compare['Predicted']).mean()
                st.success(f"**🧪 盲測驗證中** │ 目前設定平均誤差：**{mae:.3f} 公尺**")
        else:
            st.success(f"**⚡ 即時模擬中** │ 已自動補遺 `{len(predict_data)}` 天的水位。")
        
        final_df = original_df.copy()
        final_df['Simulated'] = False
        final_df['WaterLevel_Simulated'] = np.nan
        final_df.loc[predict_data_copy.index, 'WaterLevel_Simulated'] = predict_data_copy['WaterLevel_Simulated']
        
        # --- 繪圖與呈現 ---
        st.markdown("### 📈 地下水位與雨量動態圖")
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.7, 0.3], subplot_titles=("地下水位變化", "日降雨量")
        )
        
        actual_mask = final_df['WaterLevel'].notna()
        # 🌟 自定義 hovertemplate：第一行直接顯示純數字日期，後面接數值
        fig.add_trace(go.Scatter(
            x=final_df[actual_mask].index, 
            y=final_df.loc[actual_mask, 'WaterLevel'], 
            mode='lines', 
            name='實際觀測水位', 
            line=dict(color='rgba(31, 119, 180, 0.4)', width=2.5),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>實際水位: %{y:.3f} m<extra></extra>'
        ), row=1, col=1)
        
        sim_mask = final_df['WaterLevel_Simulated'].notna()
        fig.add_trace(go.Scatter(
            x=final_df[sim_mask].index, 
            y=final_df[sim_mask]['WaterLevel_Simulated'], 
            mode='lines', 
            name='AI 模擬補遺水位', 
            line=dict(color='#FF4B4B', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>模擬水位: %{y:.3f} m<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=final_df.index, 
            y=final_df['Rainfall'], 
            name='日雨量', 
            marker_color='rgba(0, 191, 255, 0.7)',
            hovertemplate='日期: %{x|%Y-%m-%d}<br>日雨量: %{y:.1f} mm<extra></extra>'
        ), row=2, col=1)

        fig.update_yaxes(title_text="地下水位 (m)", row=1, col=1)
        fig.update_yaxes(title_text="日雨量 (mm)", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)
        
        # 🌟 透過 hoverlabel 設定把預設的英文標頭隱藏，並讓畫面極致簡潔
        fig.update_layout(
            height=750, 
            hovermode="x unified", 
            legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            hoverlabel=dict(bgcolor="white", font_size=13, font_family="monospace")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📥 下載模擬結果")
        if validation_mode:
            output_df = final_df[['Rainfall', 'WaterLevel', 'WaterLevel_Simulated']].rename(
                columns={'WaterLevel': 'Actual_WaterLevel(m)', 'WaterLevel_Simulated': 'Simulated_WaterLevel(m)'})
        else:
            merged_waterlevel = final_df['WaterLevel'].fillna(final_df['WaterLevel_Simulated'])
            is_simulated = final_df['WaterLevel'].isna() & final_df['WaterLevel_Simulated'].notna()
            output_df = pd.DataFrame({'Rainfall': final_df['Rainfall'], 'WaterLevel(m)': merged_waterlevel, 'Is_Simulated_Data': is_simulated})
        
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer)
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
        st.download_button(label="📥 下載資料集", data=csv_bytes, file_name="waterlevel_simulated.csv", mime="text/csv")
    except Exception as e:
        st.error(f"檔案解析發生錯誤：{e}")
