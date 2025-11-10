"""
Quick script to train and export regression model for app.py
Based on Final.ipynb logic
"""
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Training Regression Model for app.py")
print("="*60)

# 1. Fetch Data
print("\n📊 Fetching AAPL 5-minute data...")
df = yf.download("AAPL", period="60d", interval="5m", progress=False, auto_adjust=True)
df = df.reset_index()
print(f"✅ Fetched {len(df)} rows")

# 2. Engineer Features
print("\n🔧 Engineering features...")
df = ta.add_all_ta_features(
    df, open="Open", high="High", low="Low", 
    close="Close", volume="Volume", fillna=True
)
df['Volume'] = np.log1p(df['Volume'])

# 3. Create Target (10 minutes ahead = 2 bars)
df['Close_future'] = df['Close'].shift(-2)
df = df.dropna(subset=['Close_future'])

# 4. Select Features (same as app.py expects)
feature_cols = [
    'Volume', 'trend_macd_diff', 'trend_adx', 'momentum_rsi',
    'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_roc',
    'volatility_atr', 'volatility_bbw', 'volume_mfi'
]

X = df[feature_cols].copy()
y = df['Close_future'].copy()

# Remove any remaining NaN/inf
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]

print(f"✅ Features: {X.shape}")

# 5. Train/Test Split (80/20, no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# 6. Scale Data
print("\n⚖️ Scaling features...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# 7. Train Model (RandomForest as in original)
print("\n🤖 Training RandomForest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_scaled)
print("✅ Model trained!")

# 8. Evaluate
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test.values

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)

print("\n📊 Model Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: ${rmse:.2f}")
print(f"   MAE: ${mae:.2f}")

# 9. Export Model Files
print("\n💾 Exporting model files...")
joblib.dump(model, 'stock_predictor_model.joblib')
joblib.dump(scaler_X, 'stock_scaler_X.joblib')
joblib.dump(scaler_y, 'stock_scaler_y.joblib')

print("✅ Model saved: stock_predictor_model.joblib")
print("✅ Scaler X saved: stock_scaler_X.joblib")
print("✅ Scaler y saved: stock_scaler_y.joblib")

print("\n" + "="*60)
print("🎉 Model training complete!")
print("="*60)
print("\nYou can now run: streamlit run app.py")
