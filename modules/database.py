import sqlite3
import pandas as pd
from datetime import datetime
import threading
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str = "crypto_analytics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    source TEXT DEFAULT 'websocket'
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_tick_symbol_timestamp 
                ON tick_data(symbol, timestamp)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp 
                ON ohlcv_data(symbol, timeframe, timestamp)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    symbol TEXT,
                    message TEXT NOT NULL,
                    value REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_path}")
    
    def insert_tick(self, symbol: str, timestamp: str, price: float, size: float, source: str = 'websocket'):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO tick_data (symbol, timestamp, price, size, source) VALUES (?, ?, ?, ?, ?)',
                (symbol, timestamp, price, size, source)
            )
            conn.commit()
            conn.close()
    
    def insert_ticks_batch(self, ticks: List[Dict]):
        if not ticks:
            return
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executemany(
                'INSERT INTO tick_data (symbol, timestamp, price, size, source) VALUES (?, ?, ?, ?, ?)',
                [(t['symbol'], t['ts'], t['price'], t['size'], t.get('source', 'websocket')) for t in ticks]
            )
            conn.commit()
            conn.close()
            logger.info(f"Inserted {len(ticks)} ticks into database")
    
    def insert_ohlcv(self, symbol: str, timestamp: str, timeframe: str, 
                     open_price: float, high: float, low: float, close: float, volume: float):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO ohlcv_data 
                (symbol, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timestamp, timeframe, open_price, high, low, close, volume))
            conn.commit()
            conn.close()
    
    def insert_ohlcv_batch(self, ohlcv_list: List[Dict]):
        if not ohlcv_list:
            return
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO ohlcv_data 
                (symbol, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(o['symbol'], o['timestamp'], o['timeframe'], o['open'], 
                   o['high'], o['low'], o['close'], o['volume']) for o in ohlcv_list])
            conn.commit()
            conn.close()
    
    def get_tick_data(self, symbol: Optional[str] = None, limit: Optional[int] = None, 
                      start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            query = 'SELECT symbol, timestamp, price, size, source FROM tick_data WHERE 1=1'
            params = []
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += f' LIMIT {limit}'
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce', format='mixed')
                df = df.dropna(subset=['timestamp'])
            
            return df
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
            '''
            
            params = [symbol, timeframe]
            
            if limit:
                query += f' LIMIT {limit}'
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce', format='mixed')
                df = df.dropna(subset=['timestamp'])
                df = df.sort_values('timestamp')
            
            return df
    
    def insert_alert(self, alert_type: str, message: str, symbol: Optional[str] = None, value: Optional[float] = None):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO alerts (timestamp, alert_type, symbol, message, value) VALUES (?, ?, ?, ?, ?)',
                (datetime.utcnow().isoformat(), alert_type, symbol, message, value)
            )
            conn.commit()
            conn.close()
            logger.info(f"Alert inserted: {alert_type} - {message}")
    
    def get_recent_alerts(self, limit: int = 50) -> pd.DataFrame:
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                'SELECT timestamp, alert_type, symbol, message, value FROM alerts ORDER BY timestamp DESC LIMIT ?',
                conn, params=[limit]
            )
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce', format='mixed')
                df = df.dropna(subset=['timestamp'])
            
            return df
    
    def get_data_summary(self) -> Dict:
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM tick_data')
            total_ticks = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM tick_data')
            unique_symbols = cursor.fetchone()[0]
            
            cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM tick_data')
            time_range = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM ohlcv_data')
            total_ohlcv = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM alerts')
            total_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_ticks': total_ticks,
                'unique_symbols': unique_symbols,
                'earliest_timestamp': time_range[0],
                'latest_timestamp': time_range[1],
                'total_ohlcv_bars': total_ohlcv,
                'total_alerts': total_alerts
            }
    
    def clear_all_data(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM tick_data')
            cursor.execute('DELETE FROM ohlcv_data')
            cursor.execute('DELETE FROM alerts')
            conn.commit()
            conn.close()
            logger.info("All data cleared from database")
