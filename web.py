
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta, date
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forecast Harga IDR/USD", layout="wide") 
st.title("Aplikasi Prediksi Harga IDR/USD") 


def create_time_series_features(df, target_col='Close'):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    lag_days = [1, 7, 14]
    for lag in lag_days:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    window_sizes = [7, 14]
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df


@st.cache_data
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None 

@st.cache_data
def load_metrics(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def forecast_future_date(model, future_date_obj, historical_data, adjust_for_long_term=False, annual_growth_rate=0.0):
    try:
        
        last_data = historical_data.tail(30).copy()
        
        
        future_df = pd.DataFrame(index=[pd.to_datetime(future_date_obj)])
        future_df['Close'] = np.nan 
        
        
        combined_df = pd.concat([last_data, future_df])
        
        
        combined_features = create_time_series_features(combined_df, target_col='Close')
        
        
        future_row = combined_features.tail(1)
        
        
        FEATURES = [
            'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
            'lag_1', 'lag_7', 'lag_14',
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14'
        ]
        
        
        X_future = future_row[FEATURES]
        
        
        prediction = model.predict(X_future)
        
        
        if adjust_for_long_term:
            last_historical_date = historical_data.index.max()
            years_diff = (pd.to_datetime(future_date_obj) - last_historical_date).days / 365.25
            if years_diff > 0:
                
                prediction = prediction[0] * ((1 + annual_growth_rate) ** years_diff)
                return prediction
        
        return prediction[0] 
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        return None


MODEL_PATH = 'model_forecast_idr.joblib'
METRICS_PATH = 'evaluasi_model.json'
DATA_PATH = 'dataset.csv'


model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)


if model is None or metrics is None:
    st.error(f"Error: File model ('{MODEL_PATH}') atau file evaluasi ('{METRICS_PATH}') tidak ditemukan.")
    st.warning("Pastikan Anda sudah menjalankan skrip training yang baru untuk menghasilkan kedua file tersebut.")
else:
    st.success("Model prediksi dan data evaluasi berhasil dimuat!")

    
    with st.expander("📊 Lihat Performa & Evaluasi Model Terbaru", expanded=True):
        
        mae = metrics.get('mae', 0); mse = metrics.get('mse', 0); rmse = metrics.get('rmse', 0)
        mape = metrics.get('mape', 0); r2 = metrics.get('r2', 0); accuracy = metrics.get('accuracy', 0)
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Tingkat Akurasi (100% - MAPE)", value=f"{accuracy:.2f} %")
            st.metric(label="Persentase Kesalahan (MAPE)", value=f"{mape:.2f} %")
        with col2:
            st.metric(label="Mean Absolute Error (MAE)", value=f"Rp {mae:,.2f}")
            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:,.2f}")
        with col3:
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"Rp {rmse:,.2f}")
            st.metric(label="R² Score", value=f"{r2:.4f}")

    
    st.header("Pilih Tanggal untuk Prediksi")
    try:
        
        historical_df = pd.read_csv(DATA_PATH)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df.set_index('Date')[['Close']].dropna()

        
        today = date.today()
        selected_date = st.date_input(
            "Pilih tanggal untuk prediksi:",
            value=today,
            min_value=today
        )
        
        if st.button("🔮 Prediksi Harga"):
            
            selected_date_dt = pd.to_datetime(selected_date)
            last_historical_date = historical_df.index.max()

            
            if selected_date_dt <= last_historical_date:
                try:
                    
                    past_value = historical_df.loc[selected_date_dt, 'Close']
                    st.subheader(f"Data Historis untuk {selected_date.strftime('%d-%m-%Y')}")
                    st.metric(label="Harga Penutupan (dari dataset)", value=f"Rp {past_value:,.2f}")
                except KeyError:
                    st.warning(f"Tidak ada data untuk tanggal {selected_date.strftime('%d-%m-%Y')}. Ini mungkin akhir pekan atau hari libur.")
            else:
                
                with st.spinner('Menghitung prediksi...'):
                    
                    start_date = historical_df.index.min()
                    end_date = last_historical_date
                    start_price = historical_df.loc[start_date, 'Close']
                    end_price = historical_df.loc[end_date, 'Close']
                    num_years = (end_date - start_date).days / 365.25
                    
                    
                    
                    if num_years > 0 and start_price > 0 and end_price > 0:
                        annual_growth_rate = (end_price / start_price)**(1 / num_years) - 1
                    else:
                        annual_growth_rate = 0.0

                    
                    hasil_prediksi = forecast_future_date(
                        model, 
                        selected_date, 
                        historical_df,
                        adjust_for_long_term=True,
                        annual_growth_rate=annual_growth_rate
                    )

                
                if hasil_prediksi is not None:
                    st.subheader("Hasil Prediksi:")
                    st.metric(label=f"Prediksi Harga pada {selected_date.strftime('%d-%m-%Y')}", value=f"Rp {hasil_prediksi:,.2f}")
                    with st.expander("Detail Prediksi"):
                        st.info(f"Prediksi disesuaikan dengan tingkat pertumbuhan tahunan rata-rata sebesar **{annual_growth_rate:.2%}**(CAGR) yang dihitung dari data historis(01-01-2020 sampai 18-06-2025)")

    except FileNotFoundError:
        st.error(f"Dataset '{DATA_PATH}' tidak ditemukan. File ini diperlukan untuk membuat prediksi.")


st.markdown("---")
st.markdown("KELOMPOK 8 | ZIDAN | JOSHUA | SEN ARYA | ALEVIAN | RAMA | Model: XGBoost")