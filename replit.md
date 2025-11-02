# Cryptocurrency Quantitative Analytics Dashboard

## Overview

This is a real-time cryptocurrency analytics platform designed for quantitative trading analysis, specifically focusing on statistical arbitrage and pairs trading in cryptocurrency futures markets. The application provides live market data streaming, statistical analysis, and alerting capabilities for professional traders analyzing multi-frequency trading opportunities.

The system ingests real-time tick data from Binance Futures WebSocket API, stores it in a local SQLite database, performs various quantitative analyses (regression, correlation, stationarity tests), and visualizes the results through an interactive Streamlit dashboard.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework
- **Rationale**: Provides rapid development of data-intensive dashboards with minimal frontend code
- **Interactive Visualizations**: Plotly for candlestick charts, correlation heatmaps, and time-series plots
- **Session State Management**: Streamlit's built-in session state for managing WebSocket connections and user preferences
- **Auto-refresh Capability**: Real-time dashboard updates without page reloads

### Backend Architecture

**Modular Python Design** with separation of concerns:

1. **Data Ingestion Layer** (`modules/data_ingestion.py`)
   - WebSocket client for live Binance Futures streams
   - File upload support for NDJSON and CSV formats
   - Data normalization to unified schema
   - **Design Choice**: Callback-based architecture for real-time processing

2. **Analytics Engine** (`modules/analytics.py`)
   - Statistical computations: OLS regression, robust regression (Huber, Theil-Sen), correlation analysis
   - **Kalman Filter**: Dynamic hedge ratio estimation with time-varying parameters
   - Spread and z-score calculations for pairs trading
   - Augmented Dickey-Fuller (ADF) test for stationarity
   - **Liquidity Filters**: Volume-based filtering and liquidity scoring
   - **Libraries**: NumPy, Pandas, SciPy, scikit-learn, statsmodels
   - **Design Choice**: Stateless static methods for pure functional analytics

3. **Data Resampling** (`modules/resampler.py`)
   - Tick data aggregation to OHLCV (candlestick) data
   - Multi-timeframe support: 1s, 1m, 5m, 15m, 1h, 1d
   - **Design Choice**: Pandas resample for efficient time-series aggregation

4. **Alert System** (`modules/alerts.py`)
   - Rule-based alerting on metrics (z-score, price, spread, correlation)
   - **Multi-condition alerts**: Complex rules with AND/OR logic combining multiple metrics
   - **Webhook notifications**: Real-time alerts sent to external endpoints (Slack, Discord, custom)
   - Callback mechanism for extensible alert actions
   - Alert history tracking with timestamps

5. **Backtesting Engine** (`modules/backtesting.py`)
   - Mean-reversion strategy simulation with z-score entry/exit thresholds
   - **Capital-aware position sizing**: 95% allocation with hedge-adjusted exposure
   - Performance metrics: total return, Sharpe ratio, max drawdown, win rate
   - Trade history with timestamps and exit reasons
   - **Design Choice**: Proper spread-based PnL calculations prevent look-ahead bias

6. **WebSocket Server API** (`websocket_server.py`)
   - Flask-SocketIO server for external data ingestion on port 5001
   - REST endpoints: /api/tick (single), /api/ticks/batch (bulk)
   - WebSocket events for real-time data streaming
   - Integration with DatabaseManager for persistence

### Data Storage

**SQLite Database** (`modules/database.py`)
- **Primary Storage**: Local SQLite for tick data and OHLCV candles
- **Schema Design**:
  - `tick_data` table: Raw tick-level trades with symbol, timestamp, price, size
  - `ohlcv_data` table: Aggregated candlestick data with timeframe dimension
  - `alerts_log` table: Alert history with timestamps and triggered values
- **Indexing**: Composite indexes on (symbol, timestamp) for query performance
- **Thread Safety**: Threading locks for concurrent read/write operations
- **Rationale**: SQLite chosen for simplicity, zero-configuration, and sufficient performance for single-user analytics workloads

### Concurrency Model

**Threading-based Architecture**:
- WebSocket connections run in separate threads per symbol
- Queue-based communication between WebSocket threads and main Streamlit thread
- Thread-safe database writes with mutex locks
- **Design Choice**: Python threading (not asyncio) for simpler integration with Streamlit's synchronous model

### Data Flow

1. **Live Data Path**: Binance WebSocket → Normalization → Queue → Database → Resampling → Analytics → Visualization
2. **File Upload Path**: CSV/NDJSON Upload → Parsing → Database → Resampling → Analytics → Visualization
3. **Alert Path**: Analytics Engine → Alert Manager → Rule Evaluation → Callback Execution → Database Logging

## External Dependencies

### Third-Party APIs

**Binance Futures WebSocket API**
- **Endpoint**: `wss://fstream.binance.com/ws/{symbol}@trade`
- **Data Format**: JSON trade events with event type, symbol, price, quantity, timestamp
- **Purpose**: Real-time tick-by-tick trade data for cryptocurrency futures
- **Rate Limits**: None specified for WebSocket streams (connection-based)

### Python Libraries

**Data Processing**:
- `pandas`: Time-series data manipulation and resampling
- `numpy`: Numerical computations
- `scipy`: Statistical functions and hypothesis testing
- `statsmodels`: Econometric models (ADF test)
- `sklearn`: Machine learning models (LinearRegression, HuberRegressor)

**Web Framework**:
- `streamlit`: Dashboard framework and UI components

**Visualization**:
- `plotly`: Interactive charting library

**Networking**:
- `websocket-client`: WebSocket connections to Binance

**Database**:
- `sqlite3`: Standard library SQLite interface (no external dependency)

### File Format Support

**Input Formats**:
- NDJSON (Newline-Delimited JSON): Streaming JSON format for tick data
- CSV: Tabular format with headers for tick or OHLCV data

**Output Formats**:
- CSV exports for processed analytics and summary statistics

### Browser-Based Data Collection

The repository includes an HTML asset (`attached_assets/binance_browser_collector_save_test_*.html`) for browser-based WebSocket data collection, suggesting an alternative client-side data gathering approach independent of the Python backend.