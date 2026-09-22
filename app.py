import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import io

st.set_page_config(page_title="💧 地下水位模擬補遺工具", layout="wide")
st.title("💧 地下水位模擬與補遺工具")
st.write("上傳您的「雨量資料」與「地下水位資料」，系統將利用機器學習模型（學習降雨與水位的遲滯關係）自動模擬出遺失區段的水位數據。")

# --- 側邊欄：檔案上傳與參數設定 ---
st.sidebar.header("📁 1. 資料上傳")
rain_file = st.sidebar.file_uploader("上傳雨量資料 (Excel/CSV)", type=["xlsx", "xls", "csv"])
hobo_file = st.sidebar.file_uploader("上傳 HOBO 水位資料 (CSV)", type=["csv"])

st.sidebar.header("⚙️ 2. 模型參數設定")
model_choice = st.sidebar.selectbox("選擇預測模型", ["隨機森林 (Random Forest) - 推薦", "線性迴歸 (Linear Regression)"])
rolling_windows = st.sidebar.multiselect(
    "選擇降雨累積天數 (特徵工程)", 
    options=[1, 3, 5, 7, 14, 20, 30, 60], 
    default=[1, 3, 7, 14, 30],
    help="地下水位反應有延遲性，加入前幾天的累積雨量可提高模擬準確度。"
)

# --- 資料處理函數 ---
@st.cache_data
def load_and_preprocess_data(rain_file, hobo_file):
    # 1. 讀取雨量資料 (處理格式與欄位)
    if rain_file.name.endswith('.csv'):
        rain_df = pd.read_csv(rain_file)
    else:
        rain_df = pd.read_excel(rain_file)
    
    # 尋找可能的日期與雨量欄位 (依據提供的下田埔 Excel 結構)
    # 這裡我們做一個防呆，讓使用者可以自選欄位，但先嘗試自動抓取
    rain_cols = rain_df.columns.tolist()
    rain_date_col = next((c for c in rain_cols if 'Time' in str(c) or '日期' in str(c)), rain_cols[0])
    rain_val_col = next((c for c in rain_cols if 'R1' in str(c) or '雨量' in str(c)), rain_cols[1])
    
    rain_df = rain_df[[rain_date_col, rain_val_col]].dropna()
    rain_df.columns = ['Date', 'Rainfall']
    rain_df['Date'] = pd.to_datetime(rain_df['Date'], errors='coerce')
    rain_df = rain_df.dropna(subset=['Date'])
    rain_df.set_index('Date', inplace=True)
    rain_daily = rain_df.resample('D').sum() # 雨量用加總

    # 2. 讀取 HOBO 水位資料
    try:
        hobo_df = pd.read_csv(hobo_file, encoding='utf-8')
    except:
        hobo_df = pd.read_csv(hobo_file, encoding='big5') # 處理中文編碼
        
    hobo_cols = hobo_df.columns.tolist()
    hobo_date_col = hobo_cols[0]
    hobo_val_col = hobo_cols[1]
    
    hobo_df = hobo_df[[hobo_date_col, hobo_val_col]].dropna()
    hobo_df.columns = ['Date', 'WaterLevel']
    hobo_df['Date'] = pd.to_datetime(hobo_df['Date'], errors='coerce')
    hobo_df = hobo_df.dropna(subset=['Date'])
    hobo_df.set_index('Date', inplace=True)
    hobo_daily = hobo_df.resample('D').mean() # 水位用平均
    
    # 3. 合併資料
    merged_df = pd.merge(rain_daily, hobo_daily, left_index=True, right_index=True, how='outer')
    return merged_df

# --- 主程式 ---
if rain_file and hobo_file:
    with st.spinner("正在讀取與清洗資料..."):
        try:
            df = load_and_preprocess_data(rain_file, hobo_file)
            st.success("✅ 資料載入並對齊完成！(已轉換為日平均/日加總格式)")
            
            # 特徵工程：計算過去 N 天的累積降雨
            for window in rolling_windows:
                df[f'Rain_{window}D_Sum'] = df['Rainfall'].rolling(window=window, min_periods=1).sum()
            
            # 將資料分為「有水位資料(訓練集)」與「無水位資料(待預測集)」
            # 確保待預測集的特徵 (雨量) 是有數值的
            df_model = df.dropna(subset=[f'Rain_{w}D_Sum' for w in rolling_windows])
            
            train_data = df_model.dropna(subset=['WaterLevel'])
            predict_data = df_model[df_model['WaterLevel'].isna()]
            
            st.markdown(f"**📊 資料統計：** 訓練用歷史天數 `{len(train_data)}` 天 │ 待模擬補遺天數 `{len(predict_data)}` 天")
            
            if len(train_data) == 0:
                st.error("找不到雨量與水位重疊的時間段，無法訓練模型！")
                st.stop()
            if len(predict_data) == 0:
                st.warning("所有雨量對應的日子都有水位資料，無需進行補遺！")
                st.stop()

            # --- 模型訓練與預測 ---
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
            
            # 將預測結果填回 DataFrame
            predict_data_copy = predict_data.copy()
            predict_data_copy['WaterLevel_Simulated'] = predicted_levels
            
            # 合併原始資料與模擬資料供圖表顯示
            final_df = df.copy()
            final_df['Simulated'] = False
            final_df.loc[predict_data_copy.index, 'WaterLevel'] = predict_data_copy['WaterLevel_Simulated']
            final_df.loc[predict_data_copy.index, 'Simulated'] = True
            
            # --- 繪製互動式圖表 (Plotly) ---
            st.markdown("### 📈 地下水位歷史與模擬預測圖")
            
            fig = go.Figure()
            # 實際水位 (藍色)
            actual_mask = final_df['Simulated'] == False
            fig.add_trace(go.Scatter(x=final_df[actual_mask].index, y=final_df[actual_mask]['WaterLevel'], 
                                     mode='lines', name='實際觀測水位', line=dict(color='blue')))
            
            # 模擬水位 (橘/紅色)
            sim_mask = final_df['Simulated'] == True
            fig.add_trace(go.Scatter(x=final_df[sim_mask].index, y=final_df[sim_mask]['WaterLevel'], 
                                     mode='lines', name='AI 模擬補遺水位', line=dict(color='orange', dash='dot')))
            
            # 雨量長條圖 (倒掛在上方，符合水文慣例)
            fig.add_trace(go.Bar(x=final_df.index, y=final_df['Rainfall'], 
                                 name='日雨量', yaxis='y2', marker_color='rgba(0, 191, 255, 0.5)'))

            # 設定雙 Y 軸
            fig.update_layout(
                xaxis=dict(title='日期', rangeslider=dict(visible=True), type="date"),
                yaxis=dict(title='地下水位 (m)', autorange="reversed"), # 地下水位通常越深數值越小，可以視需求移除 autorange
                yaxis2=dict(title='日雨量 (mm)', overlaying='y', side='right', autorange="reversed"),
                legend=dict(x=0.01, y=0.01),
                height=600,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 下載區塊 ---
            st.markdown("### 📥 下載模擬結果")
            
            output_df = final_df[['Rainfall', 'WaterLevel', 'Simulated']].rename(
                columns={'WaterLevel': 'WaterLevel(m)', 'Simulated': 'Is_Simulated_Data'}
            )
            
            csv_buffer = io.StringIO()
            output_df.to_csv(csv_buffer)
            csv_bytes = csv_buffer.getvalue().encode('utf-8-sig') # 加上 BOM 讓 Excel 讀取中文不亂碼
            
            st.download_button(
                label="📥 下載完整資料集 (含模擬資料).csv",
                data=csv_bytes,
                file_name="hobo_waterlevel_simulated.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"資料處理發生錯誤，請確認檔案格式是否與預期相符。錯誤訊息：{e}")
else:
    st.info("💡 請從左側面板上傳「雨量」與「地下水位」資料檔案以開始分析。")