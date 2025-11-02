import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
from datetime import datetime, timedelta
import time
import sqlite3
from pathlib import Path

class StockDataImporter:
    def __init__(self, data_dir="stock_data"):
        """Initialize the data importer with a data directory"""
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "stock_data.db")
        Path(data_dir).mkdir(exist_ok=True)
        self._initialize_database()

    def _initialize_database(self):
        """Create database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                datetime TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (datetime, symbol)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS last_update (
                symbol TEXT PRIMARY KEY,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def _fetch_yf_data(self, symbol, period="60d", interval="5m"):
        """Fetch data from yfinance with retries"""
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

    def update_stock_data(self, symbol):
        """Update stock data for a given symbol"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Check last update
            last_update = pd.read_sql(
                "SELECT last_updated FROM last_update WHERE symbol = ?",
                conn,
                params=(symbol,)
            )
            
            # Fetch new data
            df = self._fetch_yf_data(symbol)
            if df.empty:
                print(f"No data available for {symbol}")
                return None
            
            # Standardize column names
            df = df.rename(columns={
                'Datetime': 'datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Convert datetime to string for storage
            df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add symbol column to the dataframe
            df['symbol'] = symbol
            
            # Insert new data
            df.to_sql('stock_data', conn, if_exists='append', index=False)
            
            # Clean up old data for this symbol
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stock_data WHERE symbol = ? AND datetime NOT IN (SELECT datetime FROM stock_data WHERE symbol = ? ORDER BY datetime DESC LIMIT 8640)", (symbol, symbol))  # Keep last 60 days of 5-min data
            
            # Update last_update timestamp
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO last_update (symbol, last_updated) VALUES (?, ?)",
                (symbol, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            
            conn.commit()
            print(f"Successfully updated data for {symbol}")
            
            return self.get_latest_data(symbol)
            
        except Exception as e:
            print(f"Error updating {symbol}: {str(e)}")
            return None
        finally:
            conn.close()

    def get_latest_data(self, symbol):
        """Get the latest data for a symbol with technical indicators"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get the data
            df = pd.read_sql(
                "SELECT * FROM stock_data WHERE symbol = ? ORDER BY datetime",
                conn,
                params=(symbol,)
            )
            
            if df.empty:
                return None
            
            # Rename columns back to original format for consistency with model
            df = df.rename(columns={
                'datetime': 'Datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Convert string datetime back to datetime object
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
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
        finally:
            conn.close()

    def is_data_fresh(self, symbol, max_age_hours=24):
        """Check if the data for a symbol is fresh"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT last_updated FROM last_update WHERE symbol = ?",
                (symbol,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False
                
            last_update = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            age = datetime.now() - last_update
            
            return age.total_seconds() / 3600 < max_age_hours
            
        except Exception:
            return False
        finally:
            conn.close()

    def get_or_update_data(self, symbol):
        """Get fresh data for a symbol, updating if necessary"""
        if not self.is_data_fresh(symbol):
            return self.update_stock_data(symbol)
        return self.get_latest_data(symbol)