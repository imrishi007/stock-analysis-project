import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_importer import StockDataImporter

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_model():
    """Load the ML model and scalers"""
    try:
        model = joblib.load('stock_predictor_model.joblib')
        scaler_X = joblib.load('stock_scaler_X.joblib')
        scaler_y = joblib.load('stock_scaler_y.joblib')
        return model, scaler_X, scaler_y, True
    except FileNotFoundError:
        return None, None, None, False

@st.cache_resource
def get_data_importer():
    """Initialize the data importer"""
    return StockDataImporter()

def prepare_features(df):
    """Prepare features for prediction"""
    df = df.copy()
    df['Volume'] = np.log1p(df['Volume'])
    
    features = [
        'Volume', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
        'trend_macd_diff', 'trend_adx', 'momentum_rsi', 'momentum_stoch_rsi_k',
        'momentum_stoch_rsi_d', 'momentum_roc', 'volatility_atr', 'volatility_bbw',
        'volume_obv', 'volume_vwap', 'volume_mfi'
    ]
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    return df[features].iloc[-1:].copy()

def create_price_chart(df, prediction=None, current_price=None):
    """Create an interactive price chart with Plotly"""
    fig = go.Figure()
    
    # Candlestick chart for recent data
    recent_df = df.tail(100)
    fig.add_trace(go.Candlestick(
        x=recent_df['Datetime'],
        open=recent_df['Open'],
        high=recent_df['High'],
        low=recent_df['Low'],
        close=recent_df['Close'],
        name='Price',
        showlegend=True
    ))
    
    # Add prediction point if available
    if prediction is not None and current_price is not None:
        last_time = df['Datetime'].iloc[-1]
        pred_time = last_time + timedelta(minutes=10)
        
        # Prediction color based on direction
        pred_color = '#ff6b6b' if prediction < current_price else '#51cf66'
        
        # Draw prediction line - simple and clean
        fig.add_trace(go.Scatter(
            x=[last_time, pred_time],
            y=[current_price, prediction],
            mode='lines+markers',
            name=f'Prediction: ${prediction:.2f}',
            line=dict(color=pred_color, width=4),
            marker=dict(size=[10, 15], color=pred_color, symbol='diamond'),
            showlegend=True
        ))
    
    fig.update_layout(
        title='Price Movement & Prediction',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

# Header
st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Stock selection
    st.subheader("Select Stock")
    
    popular_stocks = {
        "💻 Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA"],
        "🏦 Financial": ["JPM", "BAC", "GS", "V", "MA"],
        "🛒 E-commerce": ["AMZN", "BABA", "SHOP"],
        "🏥 Healthcare": ["JNJ", "PFE", "MRNA", "UNH"],
        "🚗 Automotive": ["TSLA", "F", "GM", "RIVN"]
    }
    
    selected_category = st.selectbox("Category", list(popular_stocks.keys()))
    
    use_preset = st.radio("Stock Selection", ["Preset", "Custom"], horizontal=True)
    
    if use_preset == "Preset":
        symbol = st.selectbox("Choose Stock", popular_stocks[selected_category])
    else:
        symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
    
    st.markdown("---")
    
    # Prediction button
    predict_button = st.button("🔮 Make Prediction", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Model info
    st.subheader("ℹ️ About")
    st.info("""
    **This model:**
    - Uses 16 technical indicators
    - Predicts price 10 minutes ahead
    - Trained on AAPL historical data
    - Updates with latest market data
    """)
    
    st.warning("⚠️ **Disclaimer**: Not financial advice. For educational purposes only.")

# Main content
model, scaler_X, scaler_y, model_loaded = load_model()

if not model_loaded:
    st.error("❌ **Error**: Model files not found!")
    st.info("Please ensure these files exist:\n- stock_predictor_model.joblib\n- stock_scaler_X.joblib\n- stock_scaler_y.joblib")
    st.stop()

if predict_button:
    if not symbol:
        st.error("Please enter a valid stock symbol")
        st.stop()
    
    with st.spinner(f"🔄 Fetching data and analyzing {symbol}..."):
        try:
            # Get data
            data_importer = get_data_importer()
            df = data_importer.get_or_update_data(symbol)
            
            if df is None or df.empty:
                st.error(f"❌ No data available for {symbol}. Please check the symbol.")
                st.stop()
            
            # Get company info
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'N/A')
                market_cap = info.get('marketCap', 0)
                current_market_price = info.get('regularMarketPrice', df['Close'].iloc[-1])
            except:
                company_name = symbol
                sector = "N/A"
                market_cap = 0
                current_market_price = df['Close'].iloc[-1]
            
            # Display company info
            st.subheader(f"📊 {company_name} ({symbol})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sector", sector)
            with col2:
                st.metric("Market Price", f"${current_market_price:,.2f}")
            with col3:
                if market_cap > 0:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
            with col4:
                latest_time = df['Datetime'].iloc[-1]
                st.metric("Latest Data", latest_time.strftime("%Y-%m-%d %H:%M"))
            
            st.markdown("---")
            
            # Prepare features and make prediction
            X_new = prepare_features(df)
            X_new_scaled = scaler_X.transform(X_new)
            prediction_scaled = model.predict(X_new_scaled)
            predicted_price = scaler_y.inverse_transform(prediction_scaled.reshape(1, -1))[0][0]
            
            current_price = df['Close'].iloc[-1]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Display prediction results
            st.subheader("🎯 AI Prediction Analysis")
            
            # Big prediction card
            if price_change > 0:
                card_color = "#28a745"  # Green
                trend_emoji = "📈"
                trend_text = "BULLISH"
                recommendation = "CONSIDER BUYING"
                rec_emoji = "✅"
            elif price_change < 0:
                card_color = "#dc3545"  # Red
                trend_emoji = "📉"
                trend_text = "BEARISH"
                recommendation = "CONSIDER SELLING"
                rec_emoji = "⚠️"
            else:
                card_color = "#6c757d"  # Gray
                trend_emoji = "➡️"
                trend_text = "NEUTRAL"
                recommendation = "HOLD POSITION"
                rec_emoji = "⏸️"
            
            # Create prediction card with custom HTML
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {card_color} 0%, {card_color}dd 100%); 
                        padding: 2rem; border-radius: 1rem; color: white; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
                <h2 style='margin:0; font-size: 2.5rem;'>{trend_emoji} {trend_text}</h2>
                <h1 style='margin: 1rem 0; font-size: 3.5rem; font-weight: bold;'>${predicted_price:.2f}</h1>
                <p style='font-size: 1.5rem; margin: 0.5rem 0;'>Predicted Price (10 minutes ahead)</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;'>
                <h3 style='margin: 1rem 0; font-size: 2rem;'>{rec_emoji} {recommendation}</h3>
                <p style='font-size: 1.2rem; margin: 0;'>Expected Change: <b>${price_change:+.2f}</b> ({price_change_pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "� Current Price",
                    f"${current_price:.2f}",
                    help="Latest 5-minute bar closing price"
                )
            
            with col2:
                st.metric(
                    "🎯 Predicted Price",
                    f"${predicted_price:.2f}",
                    delta=f"{price_change:+.2f}",
                    help="AI prediction for 10 minutes ahead"
                )
            
            with col3:
                st.metric(
                    "📊 Change Amount",
                    f"${abs(price_change):.2f}",
                    delta=f"{price_change_pct:+.2f}%",
                    help="Expected price movement"
                )
            
            with col4:
                magnitude = abs(price_change_pct)
                if magnitude < 0.1:
                    strength = "Very Weak"
                    strength_emoji = "⚪"
                elif magnitude < 0.3:
                    strength = "Weak"
                    strength_emoji = "🟡"
                elif magnitude < 0.5:
                    strength = "Moderate"
                    strength_emoji = "🟠"
                elif magnitude < 1.0:
                    strength = "Strong"
                    strength_emoji = "🔴"
                else:
                    strength = "Very Strong"
                    strength_emoji = "🔴🔴"
                
                st.metric(
                    "💪 Signal Strength",
                    f"{strength_emoji} {strength}",
                    help=f"{magnitude:.2f}% movement expected"
                )
            
            st.markdown("---")
            
            # Price chart
            st.subheader("📈 Price Chart with Prediction")
            fig = create_price_chart(df, predicted_price, current_price)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Key Technical Indicators Summary
            st.subheader("📊 Key Technical Indicators")
            
            # Get RSI value for sentiment
            rsi_value = X_new['momentum_rsi'].values[0]
            if rsi_value > 70:
                rsi_status = "🔴 Overbought"
                rsi_color = "red"
            elif rsi_value < 30:
                rsi_status = "🟢 Oversold"
                rsi_color = "green"
            else:
                rsi_status = "🟡 Neutral"
                rsi_color = "orange"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 2px solid #dee2e6;'>
                    <h4 style='color: #212529;'>RSI (Momentum)</h4>
                    <h2 style='color: {rsi_color}; margin: 0.5rem 0;'>{rsi_value:.1f}</h2>
                    <p style='margin: 0; font-weight: bold; color: #495057;'>{rsi_status}</p>
                    <small style='color: #6c757d;'>RSI > 70: Overbought | RSI < 30: Oversold</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                macd_value = X_new['trend_macd_diff'].values[0]
                macd_signal = "🟢 Bullish" if macd_value > 0 else "🔴 Bearish"
                macd_color = "green" if macd_value > 0 else "red"
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 2px solid #dee2e6;'>
                    <h4 style='color: #212529;'>MACD (Trend)</h4>
                    <h2 style='color: {macd_color}; margin: 0.5rem 0;'>{macd_value:.4f}</h2>
                    <p style='margin: 0; font-weight: bold; color: #495057;'>{macd_signal}</p>
                    <small style='color: #6c757d;'>Positive: Uptrend | Negative: Downtrend</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                adx_value = X_new['trend_adx'].values[0]
                if adx_value > 50:
                    adx_status = "� Very Strong"
                elif adx_value > 25:
                    adx_status = "🔶 Strong"
                else:
                    adx_status = "⚪ Weak"
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 2px solid #dee2e6;'>
                    <h4 style='color: #212529;'>ADX (Strength)</h4>
                    <h2 style='color: #212529; margin: 0.5rem 0;'>{adx_value:.1f}</h2>
                    <p style='margin: 0; font-weight: bold; color: #495057;'>{adx_status}</p>
                    <small style='color: #6c757d;'>ADX > 25: Strong Trend | ADX < 25: Weak Trend</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Trading Signal Summary
            st.subheader("🎯 Trading Signal Summary")
            
            # Calculate overall signal
            signals = []
            if price_change > 0:
                signals.append("✅ AI Prediction: Bullish")
            else:
                signals.append("⚠️ AI Prediction: Bearish")
            
            if rsi_value > 70:
                signals.append("⚠️ RSI: Overbought (caution)")
            elif rsi_value < 30:
                signals.append("✅ RSI: Oversold (potential buy)")
            else:
                signals.append("🟡 RSI: Neutral zone")
            
            if macd_value > 0:
                signals.append("✅ MACD: Bullish trend")
            else:
                signals.append("⚠️ MACD: Bearish trend")
            
            if adx_value > 25:
                signals.append("✅ ADX: Strong trend confirmation")
            else:
                signals.append("🟡 ADX: Weak trend (be cautious)")
            
            for signal in signals:
                st.markdown(f"- {signal}")
            
            st.markdown("---")
            
            # Technical indicators
            with st.expander("📊 All Technical Indicators (Advanced)"):
                indicators_df = X_new.copy()
                indicators_df = indicators_df.T
                indicators_df.columns = ['Value']
                indicators_df['Indicator'] = indicators_df.index
                indicators_df = indicators_df[['Indicator', 'Value']]
                indicators_df['Value'] = indicators_df['Value'].round(4)
                st.dataframe(indicators_df, use_container_width=True, hide_index=True)
            
            # Recent price data
            with st.expander("📋 Recent Price Data (Last 20 bars)"):
                recent_data = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(20).copy()
                recent_data = recent_data.sort_values('Datetime', ascending=False)
                st.dataframe(recent_data, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
            with st.expander("🔍 Debug Information"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ using Streamlit | ⚠️ For educational purposes only - Not financial advice</p>
</div>
""", unsafe_allow_html=True)
