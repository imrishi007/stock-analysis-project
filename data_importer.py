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

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _fetch_yf_data(_self, symbol, period="60d", interval="5m"):
        """Fetch data from yfinance with retries and caching"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Set auto_adjust explicitly to handle the warning
                df = yf.download(
                    symbol, 
                    period=period, 
                    interval=interval, 
                    progress=False,
                    auto_adjust=True
                )
                
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    df.reset_index(inplace=True)
                    
                    # Verify we have the expected columns
                    required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        print(f"Missing columns for {symbol}: {missing_columns}")
                        continue
                        
                    return df
                else:
                    print(f"Empty dataframe received for {symbol}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
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
