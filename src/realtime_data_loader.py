import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class BinanceRealtimeLoader:
    """
    Real-time data loader using Binance US API.
    Fetches actual market data for live trading scenarios.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "15m"):
        """
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
        """
        self.symbol = symbol
        self.interval = interval
        # Binance US API endpoint
        self.base_url = "https://api.binance.us/api/v3"
        self.timeout = 10
    
    def fetch_klines(
        self,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch klines data from Binance US API.
        
        Args:
            limit: Number of klines to fetch (max 1000)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            
        Returns:
            DataFrame with columns: [open_time, open, high, low, close, volume, ...]
        """
        endpoint = f"{self.base_url}/klines"
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': min(limit, 1000)  # Binance max is 1000
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning("No data returned from Binance API")
                return pd.DataFrame()
            
            # Parse klines data
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to appropriate types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Select relevant columns
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Successfully fetched {len(df)} klines from Binance")
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Binance: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error parsing Binance data: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_klines(self, limit: int = 500) -> pd.DataFrame:
        """
        Get latest klines data.
        
        Args:
            limit: Number of recent klines to fetch
            
        Returns:
            DataFrame with latest klines
        """
        return self.fetch_klines(limit=limit)
    
    def get_klines_range(
        self,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Get klines data for last N days.
        
        Args:
            days_back: Number of days to fetch
            
        Returns:
            DataFrame with klines data for specified period
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Fetch in batches to handle API limits
        all_data = []
        current_start = start_ms
        
        while current_start < end_ms:
            batch = self.fetch_klines(
                limit=1000,
                start_time=current_start,
                end_time=end_ms
            )
            
            if batch.empty:
                break
            
            all_data.append(batch)
            current_start = int(batch.iloc[-1]['open_time'].timestamp() * 1000) + 1
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.drop_duplicates(subset=['open_time'])
            df = df.sort_values('open_time').reset_index(drop=True)
            logger.info(f"Fetched {len(df)} klines for {days_back} days")
            return df
        else:
            logger.warning(f"No data fetched for {days_back} days")
            return pd.DataFrame()
    
    def get_current_price(self) -> Optional[float]:
        """
        Get current market price for the symbol.
        
        Returns:
            Current price or None if error
        """
        try:
            endpoint = f"{self.base_url}/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error fetching current price: {str(e)}")
            return None
    
    def get_ticker_info(self) -> Optional[dict]:
        """
        Get current 24h ticker information.
        
        Returns:
            Dict with ticker info (price, volume, change, etc.) or None
        """
        try:
            endpoint = f"{self.base_url}/ticker/24hr"
            params = {'symbol': self.symbol}
            
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return {
                'price': float(data['lastPrice']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'change_percent': float(data['priceChangePercent']),
                'bid': float(data['bidPrice']),
                'ask': float(data['askPrice'])
            }
        except Exception as e:
            logger.error(f"Error fetching ticker info: {str(e)}")
            return None


class YFinanceRealtimeLoader:
    """
    Alternative real-time data loader using yfinance library.
    Suitable as backup if Binance is unavailable.
    """
    
    def __init__(self, symbol: str = "BTC-USD", interval: str = "15m"):
        """
        Args:
            symbol: Yahoo Finance symbol (e.g., 'BTC-USD')
            interval: Interval (1m, 5m, 15m, 1h, 1d, etc.)
        """
        try:
            import yfinance as yf
            self.yf = yf
            self.symbol = symbol
            self.interval = interval
        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            self.yf = None
    
    def fetch_klines(
        self,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Fetch klines data using yfinance.
        
        Args:
            days_back: Number of days to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.yf:
            logger.error("yfinance not available")
            return pd.DataFrame()
        
        try:
            ticker = self.yf.Ticker(self.symbol)
            df = ticker.history(period=f"{days_back}d", interval=self.interval)
            
            if df.empty:
                logger.warning("No data returned from yfinance")
                return pd.DataFrame()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df = df[['open', 'high', 'low', 'close', 'volume']].reset_index()
            df = df.rename(columns={'Date': 'open_time', 'Datetime': 'open_time'})
            
            logger.info(f"Successfully fetched {len(df)} klines from yfinance")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self) -> Optional[float]:
        """
        Get current price using yfinance.
        
        Returns:
            Current price or None if error
        """
        if not self.yf:
            return None
        
        try:
            ticker = self.yf.Ticker(self.symbol)
            return ticker.info.get('currentPrice')
        except Exception as e:
            logger.error(f"Error fetching current price from yfinance: {str(e)}")
            return None
