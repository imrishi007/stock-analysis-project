import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
from datetime import datetime, timedelta

# Load the saved model and scalers
@st.cache_resource
def load_model():
    model = joblib.load('stock_predictor_model.joblib')
    scaler_X = joblib.load('stock_scaler_X.joblib')
    scaler_y = joblib.load('stock_scaler_y.joblib')
    return model, scaler_X, scaler_y

from data_importer import StockDataImporter

# Initialize the data importer
data_importer = StockDataImporter()

def get_stock_data(symbol, period="60d", interval="5m"):
    """Get stock data using the data importer"""
    if not symbol:
        raise ValueError("Please enter a valid stock symbol")
        
    try:
        # Get data from our importer
        df = data_importer.get_or_update_data(symbol)
        if df is None:
            raise ValueError(f"No data available for {symbol}")
        
        # Get company info
        stock = yf.Ticker(symbol)
        info = stock.info
        
        company_name = info.get('longName', symbol)
        current_price = info.get('regularMarketPrice', 0)
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'N/A')
        
        return df, {
            'company_name': company_name,
            'current_price': current_price,
            'market_cap': market_cap,
            'sector': sector
        }
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

def prepare_features(df):
    """Calculate technical indicators"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate log of volume first (this matches the training process)
    df['Volume_log'] = np.log1p(df['Volume'])
    
    # Calculate all technical indicators
    df = ta.add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )
    
    # Define features in exact order as training - matching the original training features
    features = [
        'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
        'trend_macd_diff', 'trend_adx', 'momentum_rsi', 'momentum_stoch_rsi_k',
        'momentum_stoch_rsi_d', 'momentum_roc', 'volatility_atr', 'volatility_bbw',
        'volume_obv', 'volume_vwap', 'volume_mfi', 'Volume_log'
    ]
    
    # Get only the last row (most recent data)
    return df[features].iloc[-1:].copy()

# Set up the Streamlit page
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

# Sidebar
st.sidebar.header("Settings")

# Popular stocks suggestions
popular_stocks = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "Electric Vehicles": ["TSLA", "RIVN", "LCID"],
    "E-commerce": ["AMZN", "BABA", "SHOP"],
    "Financial": ["JPM", "BAC", "GS", "V"],
    "Healthcare": ["JNJ", "PFE", "MRNA"]
}

# Add sector selection
selected_sector = st.sidebar.selectbox("Select Sector", list(popular_stocks.keys()))

# Add stock selection with both dropdown and manual input
use_suggested = st.sidebar.checkbox("Use suggested stocks", True)
if use_suggested:
    symbol = st.sidebar.selectbox("Select Stock", popular_stocks[selected_sector])
else:
    symbol = st.sidebar.text_input("Enter Stock Symbol:", "")

# Add timeframe selection
prediction_timeframe = st.sidebar.selectbox(
    "Select Prediction Timeframe",
    ["5 minutes", "10 minutes", "15 minutes"],
    index=1
)

prediction_button = st.sidebar.button("Make Prediction")

# Load model and scalers
try:
    model, scaler_X, scaler_y = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("Model files not found. Please ensure the model files are in the current directory.")
    model_loaded = False

if model_loaded and prediction_button:
    if not symbol:
        st.error("Please enter a valid stock symbol")
    else:
        with st.spinner("Fetching latest data and making prediction..."):
            try:
                # Get latest data and company info
                df, company_info = get_stock_data(symbol)
                
                # Display company information
                st.subheader(f"📊 {company_info['company_name']} ({symbol})")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sector", company_info['sector'])
                with col2:
                    st.metric("Current Market Price", f"${company_info['current_price']:,.2f}")
                with col3:
                    st.metric("Market Cap", f"${company_info['market_cap']:,.0f}")
                
                # Prepare and scale features
                X_new = prepare_features(df)
                X_new_scaled = scaler_X.transform(X_new)
                
                # Make prediction
                prediction_scaled = model.predict(X_new_scaled)
                final_prediction = scaler_y.inverse_transform(prediction_scaled.reshape(1, -1))[0][0]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Current Price")
                    current_price = df['Close'].iloc[-1]
                    st.metric(
                        "Price",
                        f"${current_price:.2f}",
                        f"{((final_prediction - current_price) / current_price * 100):.2f}%"
                    )
                
                with col2:
                    st.subheader("Predicted Price (10min ahead)")
                    st.metric(
                        "Price",
                        f"${final_prediction:.2f}",
                        f"${final_prediction - current_price:.2f}"
                    )
                
                # Show recent price chart
                st.subheader("Recent Price History")
                fig_data = pd.DataFrame({
                    'Time': df['Datetime'].iloc[-20:],
                    'Price': df['Close'].iloc[-20:]
                })
                st.line_chart(fig_data.set_index('Time'))
                
                # Additional information
                st.info("⚠️ Disclaimer: This is a prediction based on historical data and technical indicators. It should not be used as the sole basis for investment decisions.")
                
                # Show technical indicators
                st.subheader("Current Technical Indicators")
                indicators_df = pd.DataFrame(X_new, columns=X_new.columns)
                st.dataframe(indicators_df.round(4))
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

# Add information about the model
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About the Model
This model:
- Uses technical indicators for prediction
- Predicts price 10 minutes ahead
- Updates with latest market data
- Uses ML regression techniques
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")