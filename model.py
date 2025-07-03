import pandas as pd  
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np  
import joblib  
import json

def create_time_series_features(df, target_col='Close'):
    """
    Fungsi utama untuk membuat fitur (variabel input) dari data time series.
    Fitur ini akan digunakan oleh model untuk belajar pola dari data.
    """
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
        
    df = df.dropna()
    
    return df


def calculate_mape(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluate_model(model, X_test, y_test):
    """Mengevaluasi performa model pada data uji dan mengembalikan hasilnya dalam bentuk dictionary."""
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  
    mape = calculate_mape(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "mae": mae, "mse": mse, "rmse": rmse,
        "mape": mape, "r2": r2, "accuracy": 100 - mape  
    }
    return metrics


def train_and_save_all(data_path, model_filename, metrics_filename):
    """Fungsi utama untuk menjalankan seluruh proses: memuat data, melatih, mengevaluasi, dan menyimpan."""
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])  
    df = df.set_index('Date')[['Close']].dropna()  
    
    df_features = create_time_series_features(df, target_col='Close')
    
    FEATURES = [
        'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14'
    ]
    TARGET = 'Close'  
    
    X = df_features[FEATURES]  
    y = df_features[TARGET]   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    eval_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50
    )
    print("Melatih model untuk evaluasi dengan fitur baru...")
    eval_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    evaluation_results = evaluate_model(eval_model, X_test, y_test)
    print(f"Menyimpan hasil evaluasi ke file: {metrics_filename}")
    with open(metrics_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4)  
    print("Hasil evaluasi berhasil disimpan.")
    
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=eval_model.best_iteration,  
        learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    print("Melatih model final dengan seluruh data...")
    final_model.fit(X, y, verbose=False)
    
    print(f"Menyimpan model ke file: {model_filename}")
    joblib.dump(final_model, model_filename)  
    print("Model berhasil disimpan.")

if __name__ == '__main__':
    file_path = 'dataset.csv'
    model_file = 'model_forecast_idr.joblib'
    metrics_file = 'evaluasi_model.json'
    
    train_and_save_all(file_path, model_file, metrics_file)
    print("\nProses training dan penyimpanan selesai.")