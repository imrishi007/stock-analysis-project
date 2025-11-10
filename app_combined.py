import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import RobustScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AlphaVision - Dual Model Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tagline {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .up-signal {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(81, 207, 102, 0.3);
    }
    .down-signal {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.3);
    }
    .regression-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(79, 172, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_classifier_model():
    """Load XGBoost classifier model"""
    try:
        model = joblib.load('models/aapl_1h_xgboost_final.joblib')
        scaler = joblib.load('models/aapl_1h_scaler_final.joblib')
        
        with open('models/aapl_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open('models/aapl_feature_names.txt', 'r') as f:
            features = [line.strip() for line in f]
        
        return model, scaler, metadata, features, True
    except Exception as e:
        st.error(f"Classifier model error: {e}")
        return None, None, None, None, False

@st.cache_resource
def load_regression_model():
    """Load regression model (if available)"""
    try:
        model = joblib.load('stock_predictor_model.joblib')
        scaler_X = joblib.load('stock_scaler_X.joblib')
        scaler_y = joblib.load('stock_scaler_y.joblib')
        return model, scaler_X, scaler_y, True
    except:
        return None, None, None, False

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period, interval):
    """Fetch stock data from Yahoo Finance"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        
        df = df.reset_index()
        cols = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = cols
        
        if "Datetime" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "Datetime"})
            else:
                df.insert(0, "Datetime", pd.to_datetime(df.index))
                df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def engineer_features_1h(df):
    """Engineer features for 1-hour classifier"""
    try:
        base = df.copy()
        base = ta.add_all_ta_features(
            base, open="Open", high="High", low="Low", 
            close="Close", volume="Volume", fillna=True
        )
        base["Volume"] = np.log1p(base["Volume"])
        return base
    except Exception as e:
        st.error(f"Feature engineering error: {e}")
        return None

def engineer_features_5m(df):
    """Engineer features for 5-min regression (basic features only)"""
    try:
        df = df.copy()
        df['Volume'] = np.log1p(df['Volume'])
        
        # Add basic technical indicators using ta library
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low",
            close="Close", volume="Volume", fillna=True
        )
        
        # Select only the features used in original model
        feature_cols = [
            'Volume', 'trend_macd_diff', 'trend_adx', 'momentum_rsi',
            'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_roc',
            'volatility_atr', 'volatility_bbw', 'volume_mfi'
        ]
        
        # Check if all features exist
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            st.warning(f"Missing regression features: {missing}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Regression feature error: {e}")
        return None

# Header
st.markdown('<h1 class="main-header">📊 AlphaVision Dual Model</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Compare Classifier (Direction) vs Regression (Price) Predictions</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model selection
    st.subheader("Select Models")
    use_classifier = st.checkbox("🎯 Classifier (Direction)", value=True, 
                                 help="XGBoost - Predicts UP/DOWN direction")
    use_regression = st.checkbox("💵 Regression (Price)", value=True,
                                 help="Predicts exact price value")
    
    st.markdown("---")
    
    # Stock selection
    st.subheader("Stock Symbol")
    symbol = st.text_input("Ticker", "AAPL").upper()
    
    st.info("🎯 **Classifier** trained on AAPL 1h data\n\n💵 **Regression** trained on 5min data")
    
    st.markdown("---")
    
    # Prediction button
    predict_button = st.button("🚀 Make Predictions", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Model info
    st.subheader("📊 Models Loaded")
    
    clf_model, clf_scaler, clf_metadata, clf_features, clf_loaded = load_classifier_model()
    reg_model, reg_scaler_X, reg_scaler_y, reg_loaded = load_regression_model()
    
    if clf_loaded:
        st.success("✅ Classifier (XGBoost)")
        st.caption(f"~60% CV accuracy")
    else:
        st.error("❌ Classifier missing")
    
    if reg_loaded:
        st.success("✅ Regression")
        st.caption(f"97% R² (train)")
    else:
        st.warning("⚠️ Regression missing")
    
    st.markdown("---")
    st.caption("⚠️ Not financial advice. Educational use only.")

# Main content
if predict_button:
    if not symbol:
        st.error("Please enter a stock symbol")
        st.stop()
    
    # Check if at least one model is selected
    if not use_classifier and not use_regression:
        st.warning("Please select at least one model")
        st.stop()
    
    with st.spinner(f"🔄 Analyzing {symbol}..."):
        # Company info
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'N/A')
            current_price = info.get('regularMarketPrice', 0)
        except:
            company_name = symbol
            sector = "N/A"
            current_price = 0
        
        # Display header
        st.subheader(f"📈 {company_name} ({symbol})")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sector", sector)
        with col2:
            if current_price > 0:
                st.metric("Current Price", f"${current_price:,.2f}")
        with col3:
            st.metric("Analyzing", "Live Data")
        
        st.markdown("---")
        
        # Layout: Side by side predictions
        if use_classifier and use_regression:
            col_clf, col_reg = st.columns(2)
        elif use_classifier:
            col_clf = st.container()
            col_reg = None
        else:
            col_clf = None
            col_reg = st.container()
        
        # CLASSIFIER PREDICTION
        if use_classifier and clf_loaded:
            with (col_clf if col_reg else st.container()):
                st.subheader("🎯 Classifier Model")
                st.caption("XGBoost Direction Prediction (1-hour bars, 2h ahead)")
                
                # Fetch 1h data
                df_1h = fetch_stock_data(symbol, "730d", "1h")
                
                if df_1h is not None and not df_1h.empty:
                    # Engineer features
                    df_features = engineer_features_1h(df_1h)
                    
                    if df_features is not None:
                        # Prepare prediction
                        X_latest = df_features[clf_features].iloc[-1:].copy()
                        X_latest = X_latest.fillna(0).replace([np.inf, -np.inf], 0)
                        X_scaled = clf_scaler.transform(X_latest)
                        
                        # Predict
                        prediction = clf_model.predict(X_scaled)[0]
                        probs = clf_model.predict_proba(X_scaled)[0]
                        confidence = probs[prediction] * 100
                        
                        current = df_1h['Close'].iloc[-1]
                        
                        # Display prediction
                        if prediction == 1:
                            st.markdown(f"""
                            <div class='up-signal'>
                                <h2>📈 UP</h2>
                                <h1>{confidence:.1f}%</h1>
                                <p>Expected to RISE >0.2% in 2 hours</p>
                                <p style='margin-top:1rem; font-size:0.9rem;'>Current: ${current:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='down-signal'>
                                <h2>📉 DOWN</h2>
                                <h1>{confidence:.1f}%</h1>
                                <p>Expected to FALL >0.2% in 2 hours</p>
                                <p style='margin-top:1rem; font-size:0.9rem;'>Current: ${current:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Probability breakdown
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📈 UP Probability", f"{probs[1]*100:.1f}%")
                        with col2:
                            st.metric("📉 DOWN Probability", f"{probs[0]*100:.1f}%")
                        
                        # Model performance
                        with st.expander("📊 Model Performance (CV)"):
                            st.write("**Cross-Validation Results:**")
                            st.write("• Accuracy: ~60.1%")
                            st.write("• Balanced Acc: ~51.9%")
                            st.write("• UP Recall: ~20.6%")
                            st.write("• DOWN Recall: ~83.2%")
                            st.caption("Note: Model is conservative on UP predictions")
                    else:
                        st.error("Failed to engineer features")
                else:
                    st.error(f"No 1h data available for {symbol}")
        
        # REGRESSION PREDICTION
        if use_regression and reg_loaded:
            with (col_reg if col_clf else st.container()):
                st.subheader("💵 Regression Model")
                st.caption("Price Prediction (5-min bars, 10min ahead)")
                
                # Fetch 5min data
                df_5m = fetch_stock_data(symbol, "60d", "5m")
                
                if df_5m is not None and not df_5m.empty:
                    # Engineer features
                    df_features = engineer_features_5m(df_5m)
                    
                    if df_features is not None:
                        feature_cols = [
                            'Volume', 'trend_macd_diff', 'trend_adx', 'momentum_rsi',
                            'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_roc',
                            'volatility_atr', 'volatility_bbw', 'volume_mfi'
                        ]
                        
                        X_latest = df_features[feature_cols].iloc[-1:].copy()
                        X_latest = X_latest.fillna(0).replace([np.inf, -np.inf], 0)
                        
                        # Scale and predict
                        X_scaled = reg_scaler_X.transform(X_latest)
                        pred_scaled = reg_model.predict(X_scaled)
                        predicted_price = reg_scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0][0]
                        
                        current = df_5m['Close'].iloc[-1]
                        change = predicted_price - current
                        change_pct = (change / current) * 100
                        
                        # Display prediction
                        direction_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        st.markdown(f"""
                        <div class='regression-card'>
                            <h2>{direction_emoji} ${predicted_price:.2f}</h2>
                            <h3>Predicted Price</h3>
                            <p style='font-size:1.1rem; margin-top:1rem;'>
                                Current: ${current:.2f}<br>
                                Change: ${change:+.2f} ({change_pct:+.2f}%)
                            </p>
                            <p style='font-size:0.9rem; margin-top:1rem;'>10 minutes ahead</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current", f"${current:.2f}")
                        with col2:
                            st.metric("Predicted", f"${predicted_price:.2f}", 
                                     delta=f"{change:+.2f}")
                        with col3:
                            st.metric("Change %", f"{change_pct:+.2f}%")
                        
                        # Model info
                        with st.expander("📊 Model Performance"):
                            st.write("**Training Performance:**")
                            st.write("• R² Score: ~97%")
                            st.write("• Trained on: 60d 5min bars")
                            st.warning("⚠️ High R² may not reflect real trading performance")
                            st.caption("Directional accuracy: ~57% (from comparison study)")
                    else:
                        st.error("Failed to engineer features")
                else:
                    st.error(f"No 5min data available for {symbol}")
        
        st.markdown("---")
        
        # Comparison insights
        if use_classifier and use_regression and clf_loaded and reg_loaded:
            st.subheader("🔍 Model Comparison Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Classifier (Recommended for Trading)**
                - ✅ Predicts actionable direction (UP/DOWN)
                - ✅ Realistic ~60% accuracy with CV
                - ✅ Class-aware metrics (recalls)
                - ✅ SMOTE balanced
                - ⏱️ 1-hour bars, 2h horizon
                """)
            
            with col2:
                st.markdown("""
                **💵 Regression (Price Forecasting)**
                - 🎯 Predicts exact price values
                - ⚠️ High R² (97%) but misleading
                - ⚠️ ~57% directional accuracy
                - ⚠️ No train/test split shown
                - ⏱️ 5-minute bars, 10min horizon
                """)
            
            st.info("""
            **💡 Key Insight:** High R² doesn't mean good trading performance! 
            The classifier's ~60% directional accuracy is more honest and actionable than regression's 97% R².
            """)

else:
    # Landing page
    st.info("👈 Select models and click **'🚀 Make Predictions'** to compare!")
    
    st.markdown("---")
    
    st.subheader("🎯 Dual Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Classifier Model
        
        **XGBoost Direction Prediction**
        - UP/DOWN with confidence
        - ~60% CV accuracy (realistic)
        - Trained on 1-hour AAPL data
        - Predicts 2 hours ahead
        - >0.2% threshold
        - SMOTE balanced
        
        **Use for:** Trading decisions
        """)
    
    with col2:
        st.markdown("""
        ### 💵 Regression Model
        
        **Price Value Prediction**
        - Exact price forecast
        - 97% R² (training)
        - Trained on 5-minute data
        - Predicts 10 minutes ahead
        - Basic technical indicators
        
        **Use for:** Short-term price levels
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Why Both Models?")
    
    st.markdown("""
    **Different perspectives on the market:**
    
    1. **Classifier** tells you the *direction* - essential for trading
    2. **Regression** tells you the *target price* - useful for limit orders
    3. **Together** they provide a complete picture:
       - If both agree (UP + higher price) → Strong signal
       - If they disagree → Caution, low confidence
    
    **Model Performance Reality:**
    - Classifier: ~60% directional accuracy (honest, with CV)
    - Regression: 97% R² but only ~57% directional (misleading metric)
    
    💡 **Pro Tip:** Trust the classifier for direction, use regression for magnitude
    """)

st.markdown("---")
st.caption("⚠️ **Disclaimer**: Not financial advice. For educational purposes only. Past performance does not guarantee future results.")
