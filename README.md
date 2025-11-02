# Cryptocurrency Quantitative Analytics Dashboard

A comprehensive real-time analytics platform for statistical arbitrage and pairs trading in cryptocurrency futures markets. Built with Python and Streamlit, this application provides professional-grade quantitative analysis tools for traders and researchers.

## 🎯 Overview

This application is designed for quantitative traders at multi-frequency trading (MFT) firms focusing on:
- Statistical arbitrage strategies
- Pairs trading opportunities
- Risk-premia harvesting
- Market microstructure analysis
- Real-time market analytics

## ✨ Features

### Data Ingestion
- **Live WebSocket Stream**: Direct connection to Binance Futures WebSocket API for real-time tick data
- **File Upload**: Support for NDJSON and CSV file formats
- **Automatic Format Detection**: Intelligent parsing of various data formats
- **Multi-symbol Support**: Simultaneous tracking of multiple trading pairs

### Analytics Engine
- **OLS Regression**: Calculate hedge ratios between trading pairs
- **Robust Regression**: Huber regression for outlier-resistant analysis
- **Spread Calculation**: Compute statistical spreads between correlated assets
- **Z-Score Analysis**: Real-time z-score computation for mean-reversion signals
- **Rolling Correlation**: Dynamic correlation tracking
- **ADF Stationarity Test**: Augmented Dickey-Fuller test for spread stationarity
- **Volatility Analysis**: Rolling volatility calculations
- **Price Statistics**: Comprehensive statistical summaries

### Visualization
- **Interactive Candlestick Charts**: OHLCV visualization with zoom and pan
- **Spread & Z-Score Plots**: Real-time tracking of trading signals
- **Correlation Heatmaps**: Multi-asset correlation analysis
- **Volume Analysis**: Trading volume patterns and trends
- **Normalized Price Comparison**: Side-by-side pair analysis

### Alert System
- **Custom Alert Rules**: Define threshold-based alerts
- **Multiple Metrics**: Alerts for z-score, price, spread, correlation
- **Alert History**: Track all triggered alerts with timestamps
- **Database Persistence**: All alerts saved for analysis

### Data Management
- **SQLite Database**: Efficient local storage for tick and OHLCV data
- **Multi-timeframe Resampling**: 1s, 1m, 5m, 15m, 1h intervals
- **CSV Export**: Download processed data and analytics
- **Summary Statistics**: Time-series aggregation and export

## 🏗️ Architecture

### Modular Design

The application follows a clean, modular architecture with clear separation of concerns:

```
├── app.py                      # Main Streamlit application
├── modules/
│   ├── database.py            # SQLite database layer
│   ├── data_ingestion.py      # WebSocket client & file loaders
│   ├── analytics.py           # Statistical analysis engine
│   ├── resampler.py           # Time-series resampling
│   └── alerts.py              # Alert management system
├── crypto_analytics.db        # SQLite database (created at runtime)
└── README.md
```

### Component Interactions

1. **Data Layer** (`database.py`)
   - Thread-safe SQLite operations
   - Tick data storage with indexing
   - OHLCV bar storage
   - Alert persistence

2. **Ingestion Layer** (`data_ingestion.py`)
   - WebSocket client for Binance Futures
   - Multi-threaded connection management
   - File parsers for NDJSON/CSV
   - Data normalization

3. **Analytics Layer** (`analytics.py`)
   - Statistical computations
   - Regression analysis (OLS, Robust)
   - Time-series tests (ADF)
   - Rolling metrics

4. **Resampling Layer** (`resampler.py`)
   - Tick-to-OHLCV aggregation
   - Multi-timeframe support
   - Symbol pair merging

5. **Alert Layer** (`alerts.py`)
   - Rule-based alerting
   - Threshold monitoring
   - Callback system

6. **Presentation Layer** (`app.py`)
   - Streamlit dashboard
   - Interactive widgets
   - Real-time updates
   - Data visualization

### Design Principles

#### Loose Coupling
- Each module has a well-defined interface
- Components communicate through data structures (DataFrames, dictionaries)
- Minimal inter-module dependencies

#### Scalability Considerations
- **Database**: SQLite chosen for simplicity; can be swapped for PostgreSQL/TimescaleDB for production
- **WebSocket**: Multi-threaded design supports multiple symbols; can be extended to connection pooling
- **Analytics**: Vectorized operations using pandas/numpy for performance
- **Storage**: Indexed queries for fast retrieval; partition strategy can be added for scale

#### Extensibility
- **New Data Sources**: Implement interface matching `BinanceWebSocketClient` pattern
- **New Analytics**: Add methods to `AnalyticsEngine` class
- **New Visualizations**: Add tabs in Streamlit layout
- **New Alerts**: Extend `AlertRule` class with custom logic

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Internet connection for live data streaming

### Installation

The project uses the Replit environment with pre-configured dependencies:

```bash
# All dependencies are managed through Replit's package system
# Required packages:
# - streamlit
# - pandas
# - numpy
# - plotly
# - scipy
# - statsmodels
# - scikit-learn
# - websocket-client
# - requests
```

### Running the Application

Single-command execution:

```bash
streamlit run app.py --server.port 5000
```

Or use the configured Replit workflow to start automatically.

## 📊 Usage Guide

### Getting Started

1. **Choose Data Source**
   - **Live WebSocket**: Enter symbols (e.g., `btcusdt,ethusdt`) and click "Start Stream"
   - **File Upload**: Upload NDJSON or CSV files with tick data
   - **View Stored Data**: Analyze previously collected data

2. **Pair Analysis**
   - Select two symbols for pairs trading analysis
   - Choose timeframe (1s, 1m, 5m, 15m)
   - Adjust rolling window for z-score calculations
   - View hedge ratio, spread, and z-score in real-time

3. **Set Alerts**
   - Navigate to Alerts tab
   - Define metric, condition, and threshold
   - Monitor alert triggers in real-time

4. **Export Data**
   - Download pair analysis as CSV
   - Export summary statistics
   - Save OHLCV data for backtesting

### Data Format Requirements

#### NDJSON Format
```json
{"symbol": "BTCUSDT", "ts": "2025-01-01T00:00:00Z", "price": 45000.0, "size": 0.5}
{"symbol": "ETHUSDT", "ts": "2025-01-01T00:00:01Z", "price": 3000.0, "size": 1.2}
```

#### CSV Format
```csv
timestamp,symbol,price,size
2025-01-01T00:00:00Z,BTCUSDT,45000.0,0.5
2025-01-01T00:00:01Z,ETHUSDT,3000.0,1.2
```

## 🔧 Configuration

### Timeframe Settings
- **1s**: High-frequency analysis (large data volume)
- **1m**: Intraday trading (recommended for most use cases)
- **5m**: Medium-term patterns
- **15m**: Longer-term trends

### Rolling Window
- Smaller windows (5-10): More sensitive to recent changes
- Medium windows (20-30): Balanced approach (default: 20)
- Larger windows (50-100): Smoother, less reactive

### Regression Types
- **OLS**: Standard linear regression, assumes normal distribution
- **Robust (Huber)**: Resistant to outliers, better for noisy data

## 📈 Analytics Methodology

### Hedge Ratio Calculation
Using Ordinary Least Squares regression:
```
Y = β₀ + β₁X + ε
```
Where β₁ is the hedge ratio for pairs (X, Y)

### Spread Calculation
```
Spread = Y - (β₁ × X + β₀)
```

### Z-Score Normalization
```
Z = (Spread - μ) / σ
```
Where μ and σ are rolling mean and standard deviation

### Trading Signals
- **Entry Signal**: |Z-Score| > 2 (spread is 2 standard deviations from mean)
- **Exit Signal**: Z-Score crosses 0 (mean reversion)
- **Stationary Test**: ADF p-value < 0.05 indicates stationary spread

## 🎯 Use Cases

### Statistical Arbitrage
1. Identify correlated pairs using correlation heatmap
2. Verify spread stationarity with ADF test
3. Monitor z-score for entry/exit signals
4. Set alerts for z-score thresholds

### Risk Management
1. Track rolling correlation for pair stability
2. Monitor volatility for position sizing
3. Use alerts for stop-loss conditions

### Market Research
1. Analyze microstructure with tick data
2. Study correlation dynamics
3. Export data for external analysis

## 🔮 Future Enhancements

### Planned Features (Phase 2)
- **Kalman Filter**: Dynamic hedge ratio estimation
- **Backtesting Engine**: Mean-reversion strategy testing
- **Liquidity Filters**: Volume-based trade filtering
- **Multiple Regression**: Multi-asset hedge ratios
- **REST API**: External data source integration
- **WebSocket Server**: Push notifications to external systems

### Scaling Considerations
- **Database**: Migrate to TimescaleDB for production
- **Caching**: Redis for real-time metrics
- **Load Balancing**: Multiple WebSocket connections
- **Distributed Computing**: Celery for async analytics
- **Cloud Deployment**: Containerization with Docker

## 🛠️ Technology Stack

- **Framework**: Streamlit 1.x
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy, Statsmodels, Scikit-learn
- **Visualization**: Plotly
- **Database**: SQLite3
- **WebSocket**: websocket-client
- **Language**: Python 3.11

## 📝 Development Notes

### AI Usage Transparency
This project was developed with assistance from AI tools for:
- Code structure and modular architecture design
- Statistical formula implementation
- Documentation and README composition
- Debugging and optimization suggestions

The AI was used as a collaborative development tool while maintaining full understanding and control over the codebase architecture and business logic.

## 🤝 Contributing

This is an evaluation project demonstrating quantitative development capabilities. For production use:

1. Implement proper error handling and logging
2. Add unit tests for analytics functions
3. Implement database migrations
4. Add authentication for multi-user deployment
5. Set up monitoring and alerting infrastructure

## ⚠️ Disclaimer

This software is for educational and research purposes. It is not financial advice. Cryptocurrency trading involves significant risk. Always perform your own due diligence and risk assessment.

## 📄 License

This project is part of a technical evaluation assignment.

---

**Built with ❤️ for quantitative traders and researchers**
