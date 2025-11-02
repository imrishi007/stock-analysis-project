# Stock Price Prediction System

A machine learning-based stock price predictor that forecasts prices 10 minutes ahead using technical indicators.

## ✅ Status: FULLY WORKING

All issues have been identified and fixed. The system is now operational.

## 🚀 Quick Start

### Predict a Stock Price

```bash
python predict_stock.py AAPL
```

### Try Other Stocks

```bash
python predict_stock.py MSFT
python predict_stock.py GOOGL
python predict_stock.py TSLA
python predict_stock.py NVDA
```

## 📋 Requirements

- Python 3.14
- yfinance
- pandas
- numpy
- ta (technical analysis library)
- scikit-learn
- joblib

## 🔧 Installation

```bash
# Install required packages
pip install yfinance pandas numpy ta scikit-learn joblib
```

## 📊 How It Works

1. **Data Collection**: Fetches latest 60 days of 5-minute interval data from Yahoo Finance
2. **Technical Indicators**: Calculates 83+ technical indicators using the `ta` library
3. **Feature Selection**: Uses 16 key features for prediction
4. **Prediction**: ML model (trained on AAPL) predicts price 10 minutes ahead
5. **Results**: Displays predicted price, change amount, direction, and magnitude

## 🎯 Features Used

The model uses these 16 features:

1. **Volume** (log-transformed)
2. **Trend Indicators**: SMA fast/slow, EMA fast/slow, MACD diff, ADX
3. **Momentum Indicators**: RSI, Stochastic RSI (K&D), ROC
4. **Volatility Indicators**: ATR, Bollinger Band Width
5. **Volume Indicators**: OBV, VWAP, MFI

## 📁 Project Files

- `predict_stock.py` - **CLI tool for predictions** (use this!)
- `app.py` - Streamlit web app (requires Python 3.11/3.12)
- `data_importer.py` - Handles data fetching and caching
- `test_prediction.py` - Comprehensive testing script
- `test_aapl.py` - AAPL-specific diagnostics
- `FIXES_SUMMARY.md` - Detailed documentation of issues and fixes

## 🎓 Model Details

- **Algorithm**: Machine Learning Regression (trained via GridSearchCV)
- **Training Data**: AAPL 60 days of 5-minute data
- **Prediction Horizon**: 10 minutes ahead (2 bars)
- **Features**: 16 technical indicators
- **Preprocessing**: StandardScaler for features and target

## ⚠️ Known Limitations

1. **Stock-Specific**: Model was trained on AAPL data
   - May not generalize perfectly to other stocks
   - Best results expected on similar large-cap tech stocks

2. **Short-Term Only**: Predicts 10 minutes ahead
   - Not suitable for long-term investing
   - Designed for intraday analysis

3. **Market Hours**: Uses 5-minute interval data
   - Best results during market hours
   - After-hours data may be sparse

## 🐛 Troubleshooting

### Issue: "No module named 'yfinance'"
```bash
pip install yfinance pandas numpy ta scikit-learn joblib
```

### Issue: "Model files not found"
Ensure these files are in the current directory:
- `stock_predictor_model.joblib`
- `stock_scaler_X.joblib`
- `stock_scaler_y.joblib`

### Issue: "No data available for symbol"
- Check if the stock symbol is correct
- Ensure you have internet connection
- Try a different stock symbol

### Issue: Streamlit won't install on Python 3.14
**Solution**: Use `predict_stock.py` CLI tool instead (no Streamlit needed)

## 📈 Sample Output

```
======================================================================
  Stock Price Prediction for AAPL
======================================================================

[1/5] Loading ML model...
      ✓ Model loaded successfully

[2/5] Fetching latest market data for AAPL...
      ✓ Data retrieved successfully
        Latest data: 2025-10-31 19:55:00
        Current price: $270.25

[3/5] Fetching company information...
      ✓ Company: Apple Inc.
        Sector: Technology
        Market Cap: $3,995,082,424,320

[4/5] Preparing technical indicators...
      ✓ 16 features prepared

[5/5] Making prediction...
      ✓ Prediction complete

======================================================================
  PREDICTION RESULTS
======================================================================

  Current Price (latest 5-min bar):  $    270.25
  Predicted Price (10 min ahead):    $    269.41
  ──────────────────────────────────────────────────
  Expected Change:                    $     -0.84  (-0.31%)

  Direction: 📉 BEARISH (Price expected to fall)
  Magnitude: Moderate Change (0.31%)
```

## 🔮 Future Improvements

1. Multi-stock training dataset
2. Ensemble models for better accuracy
3. Confidence intervals
4. Real-time streaming predictions
5. Backtesting framework
6. Alert system for significant predictions
7. Web dashboard with charts
8. Support for longer prediction horizons

## ⚖️ Disclaimer

**This tool is for educational and research purposes only.**

- NOT financial advice
- Do NOT use as sole basis for trading decisions
- Past performance does not guarantee future results
- Always consult with qualified financial advisors
- Use at your own risk

## 📝 License

For educational use only.

## 👨‍💻 Support

If you encounter issues:
1. Check `FIXES_SUMMARY.md` for common problems
2. Run `test_prediction.py` for diagnostics
3. Run `test_aapl.py` for detailed AAPL testing

---

**Made with ❤️ for stock market analysis**
