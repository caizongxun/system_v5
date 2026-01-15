import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, repo_id: str, cache_dir: str = None):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir) if cache_dir else Path("test/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load K-line data from HuggingFace dataset
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            timeframe: '15m', '1h', or '1d'
            
        Returns:
            DataFrame with OHLCV data
        """
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            logger.info(f"Loading {symbol} {timeframe} data from HuggingFace...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                cache_dir=str(self.cache_dir)
            )
            df = pd.read_parquet(local_path)
            logger.info(f"Successfully loaded {len(df)} rows for {symbol} {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Error loading {symbol} {timeframe}: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity
        
        Returns:
            True if data is valid
        """
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Got: {df.columns.tolist()}")
            return False
        
        if df.isnull().any().any():
            logger.warning(f"Found null values: {df.isnull().sum().sum()} cells")
        
        if len(df) < 100:
            logger.error(f"Insufficient data: {len(df)} rows (minimum 100 required)")
            return False
        
        logger.info(f"Data validation passed. Total rows: {len(df)}")
        return True
