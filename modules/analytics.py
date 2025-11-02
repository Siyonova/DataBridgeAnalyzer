import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression, HuberRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsEngine:
    @staticmethod
    def calculate_price_stats(df: pd.DataFrame, symbol: str) -> Dict:
        if df.empty:
            return {}
        
        symbol_data = df[df['symbol'] == symbol]['price'] if 'symbol' in df.columns else df['price']
        
        if symbol_data.empty:
            return {}
        
        return {
            'mean': float(symbol_data.mean()),
            'median': float(symbol_data.median()),
            'std': float(symbol_data.std()),
            'min': float(symbol_data.min()),
            'max': float(symbol_data.max()),
            'current': float(symbol_data.iloc[-1]) if len(symbol_data) > 0 else None,
            'count': int(len(symbol_data))
        }
    
    @staticmethod
    def calculate_ols_regression(x_data: pd.Series, y_data: pd.Series) -> Dict:
        if len(x_data) < 2 or len(y_data) < 2:
            return {}
        
        try:
            X = x_data.values.reshape(-1, 1)
            y = y_data.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            hedge_ratio = float(model.coef_[0])
            intercept = float(model.intercept_)
            r_squared = float(model.score(X, y))
            
            return {
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'r_squared': r_squared
            }
        except Exception as e:
            logger.error(f"Error in OLS regression: {e}")
            return {}
    
    @staticmethod
    def calculate_robust_regression(x_data: pd.Series, y_data: pd.Series, method: str = 'huber') -> Dict:
        if len(x_data) < 2 or len(y_data) < 2:
            return {}
        
        try:
            X = x_data.values.reshape(-1, 1)
            y = y_data.values
            
            if method.lower() == 'huber':
                model = HuberRegressor()
                model.fit(X, y)
                
                hedge_ratio = float(model.coef_[0])
                intercept = float(model.intercept_)
                
                return {
                    'hedge_ratio': hedge_ratio,
                    'intercept': intercept,
                    'method': 'Huber'
                }
            else:
                return AnalyticsEngine.calculate_ols_regression(x_data, y_data)
        
        except Exception as e:
            logger.error(f"Error in robust regression: {e}")
            return {}
    
    @staticmethod
    def calculate_spread(x_data: pd.Series, y_data: pd.Series, hedge_ratio: float, intercept: float = 0) -> pd.Series:
        return y_data - (hedge_ratio * x_data + intercept)
    
    @staticmethod
    def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
        if len(spread) < window:
            window = max(2, len(spread))
        
        rolling_mean = spread.rolling(window=window, min_periods=1).mean()
        rolling_std = spread.rolling(window=window, min_periods=1).std()
        
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore.fillna(0)
    
    @staticmethod
    def calculate_rolling_correlation(x_data: pd.Series, y_data: pd.Series, window: int = 20) -> pd.Series:
        if len(x_data) < 2 or len(y_data) < 2:
            return pd.Series([0] * len(x_data))
        
        if len(x_data) < window:
            window = max(2, len(x_data))
        
        return x_data.rolling(window=window, min_periods=2).corr(y_data).fillna(0)
    
    @staticmethod
    def perform_adf_test(series: pd.Series) -> Dict:
        if len(series) < 3:
            return {
                'adf_statistic': None,
                'p_value': None,
                'is_stationary': False,
                'error': 'Insufficient data for ADF test'
            }
        
        try:
            series_clean = series.dropna()
            
            if len(series_clean) < 3:
                return {
                    'adf_statistic': None,
                    'p_value': None,
                    'is_stationary': False,
                    'error': 'Insufficient non-null data for ADF test'
                }
            
            result = adfuller(series_clean, autolag='AIC')
            
            return {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'is_stationary': result[1] < 0.05,
                'n_lags': int(result[2]),
                'n_obs': int(result[3])
            }
        except Exception as e:
            logger.error(f"Error in ADF test: {e}")
            return {
                'adf_statistic': None,
                'p_value': None,
                'is_stationary': False,
                'error': str(e)
            }
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().fillna(0)
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        returns = AnalyticsEngine.calculate_returns(prices)
        return returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return float(excess_returns.mean() / returns.std() * np.sqrt(252))
    
    @staticmethod
    def generate_summary_stats(df: pd.DataFrame, symbol: str, timeframe: str = '1min') -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        
        try:
            symbol_data = df[df['symbol'] == symbol].copy() if 'symbol' in df.columns else df.copy()
            
            if symbol_data.empty:
                return pd.DataFrame()
            
            symbol_data = symbol_data.sort_values('timestamp')
            symbol_data.set_index('timestamp', inplace=True)
            
            resampled = symbol_data['price'].resample(timeframe).agg([
                ('open', 'first'),
                ('high', 'max'),
                ('low', 'min'),
                ('close', 'last'),
                ('count', 'count')
            ])
            
            if 'size' in symbol_data.columns:
                volume = symbol_data['size'].resample(timeframe).sum()
                resampled['volume'] = volume
            else:
                resampled['volume'] = 0
            
            resampled['return'] = resampled['close'].pct_change() * 100
            resampled['volatility'] = resampled['return'].rolling(window=10, min_periods=1).std()
            
            resampled = resampled.reset_index()
            resampled['symbol'] = symbol
            
            return resampled
        
        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            return pd.DataFrame()
