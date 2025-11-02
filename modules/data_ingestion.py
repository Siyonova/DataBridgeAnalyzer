import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Callable, Optional
import logging
import websocket
import pandas as pd
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    def __init__(self, symbols: List[str], on_message_callback: Callable):
        self.symbols = [s.lower().strip() for s in symbols]
        self.on_message_callback = on_message_callback
        self.ws_connections = []
        self.running = False
        self.threads = []
    
    def _normalize_message(self, data: Dict) -> Dict:
        timestamp = datetime.fromtimestamp(data.get('T', data.get('E', 0)) / 1000).isoformat()
        return {
            'symbol': data.get('s', '').upper(),
            'ts': timestamp,
            'price': float(data.get('p', 0)),
            'size': float(data.get('q', 0))
        }
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('e') == 'trade':
                normalized = self._normalize_message(data)
                self.on_message_callback(normalized)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def _on_open(self, ws):
        logger.info(f"WebSocket connection opened")
    
    def _run_websocket(self, symbol: str):
        url = f"wss://fstream.binance.com/ws/{symbol}@trade"
        ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.ws_connections.append(ws)
        ws.run_forever()
    
    def start(self):
        if self.running:
            logger.warning("WebSocket client already running")
            return
        
        self.running = True
        logger.info(f"Starting WebSocket client for symbols: {self.symbols}")
        
        for symbol in self.symbols:
            thread = threading.Thread(target=self._run_websocket, args=(symbol,), daemon=True)
            thread.start()
            self.threads.append(thread)
            time.sleep(0.1)
    
    def stop(self):
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping WebSocket client")
        
        for ws in self.ws_connections:
            try:
                ws.close()
            except:
                pass
        
        self.ws_connections = []
        self.threads = []


class FileDataLoader:
    @staticmethod
    def load_ndjson(file_content: bytes) -> pd.DataFrame:
        try:
            lines = file_content.decode('utf-8').strip().split('\n')
            data = [json.loads(line) for line in lines if line.strip()]
            
            df = pd.DataFrame(data)
            
            if 'ts' in df.columns:
                df['timestamp'] = pd.to_datetime(df['ts'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                raise ValueError("No timestamp field found in NDJSON data")
            
            required_fields = ['symbol', 'price', 'size']
            for field in required_fields:
                if field not in df.columns:
                    raise ValueError(f"Missing required field: {field}")
            
            df['symbol'] = df['symbol'].str.upper()
            df['price'] = pd.to_numeric(df['price'])
            df['size'] = pd.to_numeric(df['size'])
            
            logger.info(f"Loaded {len(df)} records from NDJSON file")
            return df[['symbol', 'timestamp', 'price', 'size']]
        
        except Exception as e:
            logger.error(f"Error loading NDJSON file: {e}")
            raise
    
    @staticmethod
    def load_csv(file_content: bytes) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            
            timestamp_col = None
            for col in ['timestamp', 'ts', 'time', 'datetime']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                raise ValueError("No timestamp column found in CSV file")
            
            df['timestamp'] = pd.to_datetime(df[timestamp_col])
            
            if 'symbol' not in df.columns:
                symbol_col = None
                for col in ['Symbol', 'SYMBOL', 'ticker', 'Ticker']:
                    if col in df.columns:
                        symbol_col = col
                        break
                
                if symbol_col:
                    df['symbol'] = df[symbol_col]
                else:
                    raise ValueError("No symbol column found in CSV file")
            
            price_col = None
            for col in ['price', 'Price', 'close', 'Close', 'CLOSE']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                raise ValueError("No price column found in CSV file")
            
            df['price'] = pd.to_numeric(df[price_col])
            
            size_col = None
            for col in ['size', 'Size', 'volume', 'Volume', 'qty', 'quantity']:
                if col in df.columns:
                    size_col = col
                    break
            
            if size_col:
                df['size'] = pd.to_numeric(df[size_col])
            else:
                df['size'] = 0.0
            
            df['symbol'] = df['symbol'].str.upper()
            
            logger.info(f"Loaded {len(df)} records from CSV file")
            return df[['symbol', 'timestamp', 'price', 'size']]
        
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    @staticmethod
    def detect_and_load(file_content: bytes, filename: str) -> pd.DataFrame:
        if filename.endswith('.ndjson') or filename.endswith('.jsonl'):
            return FileDataLoader.load_ndjson(file_content)
        elif filename.endswith('.csv'):
            return FileDataLoader.load_csv(file_content)
        else:
            try:
                return FileDataLoader.load_ndjson(file_content)
            except:
                try:
                    return FileDataLoader.load_csv(file_content)
                except:
                    raise ValueError("Could not detect file format. Please use .ndjson or .csv files")
