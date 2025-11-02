import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
from datetime import datetime, timedelta
import time

class StockDataImporter:
    def __init__(self, data_dir="stock_data"):
        """Initialize the data importer with in-memory caching via Streamlit"""
        # No database needed - all caching handled by Streamlit
        pass

    @st.cache_data(ttl=3600, show_spinner="Fetching stock data...")  # Cache for 1 hour
    def _fetch_yf_data(_self, symbol, period="60d", interval="5m"):
        """Fetch data from yfinance with retries and caching"""
        max_retries = 5  # Increased retries for cloud environment
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching {symbol} data (attempt {attempt + 1}/{max_retries})...")
                
                # Try with longer timeout
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval, auto_adjust=True)
                
                if not df.empty:
                    df.reset_index(inplace=True)
                    
                    # Handle column names
                    df.columns = [col if not isinstance(col, tuple) else col[0] for col in df.columns]
                    
                    # Verify we have the expected columns
                    required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        print(f"⚠️ Missing columns for {symbol}: {missing_columns}")
                        if attempt < max_retries - 1:
                            time.sleep(3)
                            continue
                    
                    print(f"✅ Successfully fetched {len(df)} rows for {symbol}")
                    return df
                else:
                    print(f"⚠️ Empty dataframe received for {symbol} on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed for {symbol}: {type(e).__name__}: {str(e)}")
                
            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1  # 1, 3, 5, 9, 17 seconds
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        print(f"❌ All {max_retries} attempts failed for {symbol}")
        return pd.DataFrame()

    def get_latest_data(self, symbol):
        """Get the latest data for a symbol with technical indicators"""
        try:
            # Fetch data from Yahoo Finance (cached)
            df = self._fetch_yf_data(symbol)
            
            if df.empty:
                return None
            
            # Calculate technical indicators with raw volume first
            # The volume-based indicators (OBV, VWAP, MFI, etc.) need raw volume values
            df = ta.add_all_ta_features(
                df,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                fillna=True
            )
            
            # After calculating all indicators, transform Volume column to log scale
            # This matches the training process where Volume was log-transformed in-place
            df['Volume'] = np.log1p(df['Volume'])
            
            return df
            
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {str(e)}")
            return None

    def get_or_update_data(self, symbol):
        """Get fresh data for a symbol (automatically cached by Streamlit)"""
        return self.get_latest_data(symbol)
