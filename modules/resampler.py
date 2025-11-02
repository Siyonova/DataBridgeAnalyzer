import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataResampler:
    TIMEFRAME_MAP = {
        '1s': '1S',
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '1h': '1H',
        '1d': '1D'
    }
    
    @staticmethod
    def resample_ticks_to_ohlcv(df: pd.DataFrame, timeframe: str = '1m') -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        
        try:
            df = df.copy()
            
            if 'timestamp' not in df.columns:
                raise ValueError("DataFrame must have 'timestamp' column")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            pandas_timeframe = DataResampler.TIMEFRAME_MAP.get(timeframe, timeframe)
            
            results = []
            
            if 'symbol' in df.columns:
                symbols = df['symbol'].unique()
                
                for symbol in symbols:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    symbol_df = symbol_df.sort_values('timestamp')
                    symbol_df.set_index('timestamp', inplace=True)
                    
                    ohlcv = symbol_df['price'].resample(pandas_timeframe).agg([
                        ('open', 'first'),
                        ('high', 'max'),
                        ('low', 'min'),
                        ('close', 'last')
                    ])
                    
                    if 'size' in symbol_df.columns:
                        volume = symbol_df['size'].resample(pandas_timeframe).sum()
                        ohlcv['volume'] = volume
                    else:
                        ohlcv['volume'] = 0
                    
                    ohlcv = ohlcv.dropna()
                    ohlcv = ohlcv.reset_index()
                    ohlcv['symbol'] = symbol
                    ohlcv['timeframe'] = timeframe
                    
                    results.append(ohlcv)
            else:
                df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)
                
                ohlcv = df['price'].resample(pandas_timeframe).agg([
                    ('open', 'first'),
                    ('high', 'max'),
                    ('low', 'min'),
                    ('close', 'last')
                ])
                
                if 'size' in df.columns:
                    volume = df['size'].resample(pandas_timeframe).sum()
                    ohlcv['volume'] = volume
                else:
                    ohlcv['volume'] = 0
                
                ohlcv = ohlcv.dropna()
                ohlcv = ohlcv.reset_index()
                ohlcv['timeframe'] = timeframe
                
                results.append(ohlcv)
            
            if results:
                final_df = pd.concat(results, ignore_index=True)
                logger.info(f"Resampled {len(df)} ticks to {len(final_df)} OHLCV bars ({timeframe})")
                return final_df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_latest_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        
        try:
            symbol_df = df[(df['symbol'] == symbol) & (df['timeframe'] == timeframe)].copy()
            symbol_df = symbol_df.sort_values('timestamp', ascending=False).head(limit)
            symbol_df = symbol_df.sort_values('timestamp')
            
            return symbol_df
        
        except Exception as e:
            logger.error(f"Error getting latest OHLCV: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def merge_pair_data(df: pd.DataFrame, symbol1: str, symbol2: str, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        
        try:
            df1 = df[(df['symbol'] == symbol1) & (df['timeframe'] == timeframe)].copy()
            df2 = df[(df['symbol'] == symbol2) & (df['timeframe'] == timeframe)].copy()
            
            if df1.empty or df2.empty:
                return pd.DataFrame()
            
            df1 = df1[['timestamp', 'close']].rename(columns={'close': f'{symbol1}_price'})
            df2 = df2[['timestamp', 'close']].rename(columns={'close': f'{symbol2}_price'})
            
            merged = pd.merge(df1, df2, on='timestamp', how='inner')
            merged = merged.sort_values('timestamp')
            
            return merged
        
        except Exception as e:
            logger.error(f"Error merging pair data: {e}")
            return pd.DataFrame()
