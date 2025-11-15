import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AlphaVision - Dual AI Stock Predictor",
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .up-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .down-prediction {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .price-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
        border-radius: 8px;
    }
    .alert-box {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #ff4757;
    }
    .info-box {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #1e90ff;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        # Regression models (5-minute bars) - Price Prediction
        regression_model = joblib.load('stock_predictor_model.joblib')
        regression_scaler_X = joblib.load('stock_scaler_X.joblib')
        regression_scaler_y = joblib.load('stock_scaler_y.joblib')
        
        # Classification models (1-hour bars) - UP/DOWN Prediction
        classification_model = joblib.load('models/aapl_1h_xgboost_final.joblib')
        classification_scaler = joblib.load('models/aapl_1h_scaler_final.joblib')
        
        return {
            'regression_model': regression_model,
            'regression_scaler_X': regression_scaler_X,
            'regression_scaler_y': regression_scaler_y,
            'classification_model': classification_model,
            'classification_scaler': classification_scaler
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Fetch and engineer features for classification (1-hour data)
@st.cache_data(ttl=300)
def fetch_classification_features(ticker):
    try:
        df = yf.download(ticker, period="730d", interval="1h", progress=False)
        
        if df.empty:
            return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df.reset_index(inplace=True)
        
        # Add all technical indicators
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low", 
            close="Close", volume="Volume", fillna=True
        )
        
        latest = df.iloc[-1:].copy()
        current_price = latest['Close'].values[0]
        
        # Drop non-feature columns
        features_to_drop = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        feature_cols = [col for col in latest.columns if col not in features_to_drop]
        X_latest = latest[feature_cols]
        
        return X_latest, current_price
    
    except Exception as e:
        st.error(f"Error fetching classification data: {str(e)}")
        return None, None

# Fetch and engineer features for regression (5-minute data)
@st.cache_data(ttl=300)
def fetch_regression_features(ticker):
    try:
        df = yf.download(ticker, period="60d", interval="5m", progress=False)
        
        if df.empty:
            return None, None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df.reset_index(inplace=True)
        
        # Calculate technical indicators
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low",
            close="Close", volume="Volume", fillna=True
        )
        
        # Add Volume_log
        df['Volume_log'] = np.log1p(df['Volume'])
        
        # Select the 17 features
        feature_names = [
            'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
            'trend_macd_diff', 'trend_adx', 'momentum_rsi', 'momentum_stoch_rsi_k',
            'momentum_stoch_rsi_d', 'momentum_roc', 'volatility_atr', 'volatility_bbw',
            'volume_obv', 'volume_vwap', 'volume_mfi', 'Volume_log'
        ]
        
        latest = df.iloc[-1:].copy()
        current_price = latest['Close'].values[0]
        X_latest = latest[feature_names]
        
        return X_latest, current_price, df
    
    except Exception as e:
        st.error(f"Error fetching regression data: {str(e)}")
        return None, None, None

# Predictions
def predict_classification(X, models):
    try:
        X_scaled = models['classification_scaler'].transform(X)
        prediction = models['classification_model'].predict(X_scaled)[0]
        probabilities = models['classification_model'].predict_proba(X_scaled)[0]
        
        direction = "UP ⬆️" if prediction == 1 else "DOWN ⬇️"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        return {
            'direction': direction,
            'confidence': confidence * 100,
            'prob_up': probabilities[1] * 100,
            'prob_down': probabilities[0] * 100
        }
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return None

def predict_regression(X, models):
    try:
        X_scaled = models['regression_scaler_X'].transform(X)
        prediction_scaled = models['regression_model'].predict(X_scaled)
        prediction = models['regression_scaler_y'].inverse_transform(
            prediction_scaled.reshape(1, -1)
        )[0][0]
        return prediction
    except Exception as e:
        st.error(f"Regression error: {str(e)}")
        return None

# Visualization functions
def plot_price_chart_with_indicators(df, ticker, current_price, predicted_price):
    """Create comprehensive price chart with technical indicators"""
    # Get last 7 days of data for better visibility
    df_recent = df.tail(2000).copy()
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f'{ticker} Price & Moving Averages', 'RSI', 'MACD', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_recent['Datetime'],
        open=df_recent['Open'],
        high=df_recent['High'],
        low=df_recent['Low'],
        close=df_recent['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Add moving averages
    if 'trend_sma_fast' in df_recent.columns:
        fig.add_trace(go.Scatter(
            x=df_recent['Datetime'],
            y=df_recent['trend_sma_fast'],
            name='SMA Fast',
            line=dict(color='#2196F3', width=2.5)
        ), row=1, col=1)
    
    if 'trend_sma_slow' in df_recent.columns:
        fig.add_trace(go.Scatter(
            x=df_recent['Datetime'],
            y=df_recent['trend_sma_slow'],
            name='SMA Slow',
            line=dict(color='#FF9800', width=2.5)
        ), row=1, col=1)
    
    # Add prediction point
    last_time = df_recent['Datetime'].iloc[-1]
    predicted_time = last_time + timedelta(minutes=10)
    
    fig.add_trace(go.Scatter(
        x=[last_time, predicted_time],
        y=[current_price, predicted_price],
        mode='lines+markers',
        name='Prediction',
        line=dict(color='#E91E63', width=5, dash='dash'),
        marker=dict(size=18, symbol='star')
    ), row=1, col=1)
    
    # RSI
    if 'momentum_rsi' in df_recent.columns:
        fig.add_trace(go.Scatter(
            x=df_recent['Datetime'],
            y=df_recent['momentum_rsi'],
            name='RSI',
            line=dict(color='#9C27B0', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.1)'
        ), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # MACD
    if 'trend_macd_diff' in df_recent.columns:
        colors = ['green' if val >= 0 else 'red' for val in df_recent['trend_macd_diff']]
        fig.add_trace(go.Bar(
            x=df_recent['Datetime'],
            y=df_recent['trend_macd_diff'],
            name='MACD Histogram',
            marker_color=colors,
            showlegend=False
        ), row=3, col=1)
    
    # Volume
    volume_colors = ['green' if df_recent['Close'].iloc[i] >= df_recent['Open'].iloc[i] 
                     else 'red' for i in range(len(df_recent))]
    fig.add_trace(go.Bar(
        x=df_recent['Datetime'],
        y=df_recent['Volume'],
        name='Volume',
        marker_color=volume_colors,
        showlegend=False
    ), row=4, col=1)
    
    # Update layout with better interactivity
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        dragmode='pan',
        modebar_add=['drawline', 'drawopenpath', 'eraseshape'],
        margin=dict(l=70, r=70, t=90, b=70)
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def plot_confidence_gauge(confidence, direction):
    """Create a gauge chart for confidence level"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence - {direction}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea" if "UP" in direction else "#f5576c"},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def plot_prediction_comparison(class_result, price_change_pct):
    """Compare classification and regression predictions"""
    
    # Determine agreement
    class_direction = 1 if "UP" in class_result['direction'] else -1
    reg_direction = 1 if price_change_pct > 0 else -1
    
    agreement = "✅ AGREE" if class_direction == reg_direction else "⚠️ DISAGREE"
    agreement_color = "#4CAF50" if class_direction == reg_direction else "#FF9800"
    
    fig = go.Figure()
    
    # Classification prediction
    fig.add_trace(go.Bar(
        x=['Classification Model'],
        y=[class_result['confidence']],
        name=f"Direction: {class_result['direction']}",
        marker_color='#667eea',
        text=[f"{class_result['confidence']:.1f}%"],
        textposition='auto',
    ))
    
    # Regression prediction (convert to percentage)
    reg_confidence = min(abs(price_change_pct) * 10, 100)  # Scale to 0-100
    fig.add_trace(go.Bar(
        x=['Regression Model'],
        y=[reg_confidence],
        name=f"Change: {price_change_pct:+.2f}%",
        marker_color='#FF9800',
        text=[f"{price_change_pct:+.2f}%"],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Model Agreement: {agreement}",
        yaxis_title="Confidence / Change %",
        showlegend=True,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=100, b=60),
        dragmode='pan'
    )
    
    # Add agreement annotation
    fig.add_annotation(
        text=agreement,
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=20, color=agreement_color, family="Arial Black"),
        bgcolor="white",
        bordercolor=agreement_color,
        borderwidth=3,
        borderpad=10
    )
    
    return fig, (class_direction == reg_direction)

def plot_feature_importance_radar(X, feature_names):
    """Create radar chart showing current feature values"""
    # Normalize features to 0-100 scale for visualization
    values = X.iloc[0].values
    
    # Select top 8 features for cleaner visualization
    top_features = feature_names[:8]
    top_values = values[:8]
    
    # Normalize to 0-100
    normalized = []
    for val in top_values:
        if val < 0:
            normalized.append(max(0, 50 + val * 10))
        else:
            normalized.append(min(100, 50 + val * 10))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized,
        theta=top_features,
        fill='toself',
        name='Feature Values',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=500,
        title=dict(text="Key Technical Indicators", font=dict(size=18)),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 AlphaVision - Dual AI Stock Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classification (UP/DOWN) + Regression (Price) Analysis</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please ensure all .joblib files are present.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Ticker input
        st.subheader("📊 Select Stock")
        popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'DIS']
        selected_popular = st.selectbox("Popular Stocks:", [""] + popular_tickers)
        custom_ticker = st.text_input("Or enter custom ticker:", "").upper()
        ticker = custom_ticker if custom_ticker else selected_popular
        
        st.divider()
        predict_button = st.button("🔮 Generate Predictions", type="primary")
        
        st.divider()
        st.info("""
        **Model 1:** XGBoost Classifier  
        Predicts UP/DOWN using 1-hour bars
        
        **Model 2:** Ridge Regression  
        Predicts price 10 min ahead using 5-min bars
        """)
    
    # Main content
    if not ticker:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 👈 Select a stock ticker to begin")
            st.markdown("""
            **Two Powerful ML Models:**
            
            1. **XGBoost Classifier** - Predicts UP or DOWN movement
            2. **Ridge Regression** - Predicts exact price 10 minutes ahead
            
            Select a stock and click **Generate Predictions**!
            """)
    
    elif predict_button:
        with st.spinner(f'🔄 Fetching real-time data for {ticker}...'):
            X_class, current_price_class = fetch_classification_features(ticker)
            X_reg, current_price_reg, df_hist = fetch_regression_features(ticker)
            
            if X_class is None or X_reg is None:
                st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
                return
            
            current_price = current_price_reg
        
        st.success(f'✅ Data fetched successfully for {ticker}')
        
        # Current price header
        st.markdown(f"## 💰 Current Price: ${current_price:.2f}")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.divider()
        
        # Get predictions first
        with st.spinner('🔮 Generating predictions...'):
            class_result = predict_classification(X_class, models)
            predicted_price = predict_regression(X_reg, models)
        
        if not class_result or not predicted_price:
            st.error("Failed to generate predictions")
            return
        
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Check for prediction disagreement
        class_direction = 1 if "UP" in class_result['direction'] else -1
        reg_direction = 1 if price_change > 0 else -1
        predictions_agree = (class_direction == reg_direction)
        
        # Show alert if predictions disagree
        if not predictions_agree:
            st.markdown(f"""
            <div class="alert-box">
                <h3 style='margin: 0;'>⚠️ CONFLICTING PREDICTIONS DETECTED</h3>
                <p style='margin: 0.5rem 0 0 0;'>
                    The Classification model predicts <strong>{class_result['direction']}</strong> 
                    while the Regression model predicts a <strong>{'POSITIVE' if price_change > 0 else 'NEGATIVE'}</strong> change.
                    This suggests high market uncertainty. Trade with caution!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <h3 style='margin: 0;'>✅ MODELS AGREE</h3>
                <p style='margin: 0.5rem 0 0 0;'>
                    Both models predict a <strong>{class_result['direction'].split()[0]}</strong> movement.
                    Confidence level: <strong>{class_result['confidence']:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Two columns for predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Model 1: Direction Prediction")
            st.caption("1-Hour Bar Analysis (XGBoost)")
            
            is_up = "UP" in class_result['direction']
            card_class = "up-prediction" if is_up else "down-prediction"
            
            st.markdown(f"""
            <div class="prediction-box {card_class}">
                <h1 style='text-align: center; margin: 0;'>{class_result['direction']}</h1>
                <h3 style='text-align: center; margin-top: 1rem;'>Confidence: {class_result['confidence']:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(plot_confidence_gauge(class_result['confidence'], class_result['direction']), 
                          use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        
        with col2:
            st.markdown("### 📊 Model 2: Price Prediction")
            st.caption("5-Minute Bar Analysis (Ridge)")
            
            st.markdown(f"""
            <div class="price-box">
                <h3 style='text-align: center; margin: 0; color: #333;'>Predicted Price (10 min)</h3>
                <h1 style='text-align: center; margin: 0.5rem 0; color: #333;'>${predicted_price:.2f}</h1>
                <h4 style='text-align: center; margin: 0; color: {"#28a745" if price_change >= 0 else "#dc3545"};'>
                    {'+' if price_change >= 0 else ''}{price_change:.2f} ({'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}%)
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2b:
                st.metric("Expected Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        
        st.divider()
        
        # Model Agreement Chart
        st.markdown("### 🤝 Model Agreement Analysis")
        comparison_fig, agree = plot_prediction_comparison(class_result, price_change_pct)
        st.plotly_chart(comparison_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        
        st.divider()
        
        # Comprehensive Price Chart
        st.markdown("### 📈 Technical Analysis Chart")
        if df_hist is not None:
            price_chart = plot_price_chart_with_indicators(df_hist, ticker, current_price, predicted_price)
            st.plotly_chart(price_chart, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']})
        
        st.divider()
        
        # Feature Analysis
        col_feat1, col_feat2 = st.columns(2)
        
        with col_feat1:
            st.markdown("### 🎯 Technical Indicators Radar")
            feature_names = [
                'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                'trend_macd_diff', 'trend_adx', 'momentum_rsi', 'momentum_stoch_rsi_k'
            ]
            radar_fig = plot_feature_importance_radar(X_reg, feature_names)
            st.plotly_chart(radar_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        
        with col_feat2:
            st.markdown("### 📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Direction': ['UP ⬆️', 'DOWN ⬇️'],
                'Probability': [class_result['prob_up'], class_result['prob_down']]
            })
            
            fig_prob = go.Figure(data=[
                go.Bar(
                    x=prob_df['Direction'],
                    y=prob_df['Probability'],
                    marker_color=['#667eea', '#f5576c'],
                    text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto',
                )
            ])
            fig_prob.update_layout(
                height=500,
                yaxis_title="Probability (%)",
                showlegend=False,
                margin=dict(l=60, r=60, t=60, b=60),
                dragmode='pan'
            )
            st.plotly_chart(fig_prob, use_container_width=True, config={'scrollZoom': True})
        
        # Trading Insights
        st.divider()
        st.markdown("### 💡 Trading Insights")
        
        col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
        
        with col_insight1:
            st.metric("Trend Signal", 
                     class_result['direction'].split()[0],
                     f"{class_result['confidence']:.1f}% confidence")
        
        with col_insight2:
            st.metric("Price Target", 
                     f"${predicted_price:.2f}",
                     f"{price_change_pct:+.2f}%")
        
        with col_insight3:
            agreement_text = "✅ Agree" if predictions_agree else "⚠️ Disagree"
            st.metric("Model Agreement", agreement_text)
        
        with col_insight4:
            risk_level = "🟢 Low" if (predictions_agree and class_result['confidence'] > 70) else \
                        "🟡 Medium" if predictions_agree else "🔴 High"
            st.metric("Risk Level", risk_level)
        
        # Disclaimer
        st.divider()
        st.warning("""
        ⚠️ **Disclaimer:** AI predictions for educational purposes only. Not financial advice. 
        When models disagree, it indicates higher uncertainty and risk. Always do your own research.
        """)

if __name__ == "__main__":
    main()
