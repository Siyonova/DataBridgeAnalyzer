# Cryptocurrency Quantitative Analytics Dashboard

A comprehensive real-time analytics platform for statistical arbitrage and pairs trading in cryptocurrency futures markets. Built with Python and Streamlit, this application provides professional-grade quantitative analysis tools for traders and researchers.

## 🎯 Overview

This application is designed for quantitative traders at multi-frequency trading (MFT) firms focusing on:
- Statistical arbitrage strategies
- Pairs trading opportunities
- Risk-premia harvesting
- Market microstructure analysis
- Real-time market analytics
<img width="1913" height="760" alt="image" src="https://github.com/user-attachments/assets/022581a4-986c-4aa1-80ae-899c0f61176d" />


## ✨ Features

### Data Ingestion
- **Live WebSocket Stream**: Direct connection to Binance Futures WebSocket API for real-time tick data
- **File Upload**: Support for NDJSON and CSV file formats
- **Automatic Format Detection**: Intelligent parsing of various data formats
- **Multi-symbol Support**: Simultaneous tracking of multiple trading pairs
<img width="446" height="303" alt="image" src="https://github.com/user-attachments/assets/737af38a-d3bd-4b2a-bd58-2ff94b370ec9" />


### Analytics Engine
- **OLS Regression**: Calculate hedge ratios between trading pairs
- **Robust Regression**: Huber and Theil-Sen regression for outlier-resistant analysis
- **Kalman Filter**: Dynamic hedge ratio estimation with time-varying parameters
- **Spread Calculation**: Compute statistical spreads between correlated assets
- **Z-Score Analysis**: Real-time z-score computation for mean-reversion signals
- **Rolling Correlation**: Dynamic correlation tracking
- **ADF Stationarity Test**: Augmented Dickey-Fuller test for spread stationarity
- **Volatility Analysis**: Rolling volatility calculations
- **Price Statistics**: Comprehensive statistical summaries
- **Liquidity Filters**: Volume-based filtering to identify tradeable pairs
<img width="1908" height="866" alt="image" src="https://github.com/user-attachments/assets/c33c4a6c-dadd-49e0-961f-010e440fe71e" />
<img width="532" height="267" alt="image" src="https://github.com/user-attachments/assets/2eb1006f-9c73-47ee-a6af-126c5d8d5b3f" />


### Visualization
- **Interactive Candlestick Charts**: OHLCV visualization with zoom and pan
<img width="1447" height="733" alt="image" src="https://github.com/user-attachments/assets/8fcb9d0a-e0dc-4a7d-aa0d-22e5bb486ab5" />
<img width="1447" height="696" alt="image" src="https://github.com/user-attachments/assets/2336d8d9-143b-41de-9d2d-105fe96c14c7" />

- **Spread & Z-Score Plots**: Real-time tracking of trading signals
- **Correlation Heatmaps**: Multi-asset correlation analysis
- **Volume Analysis**: Trading volume patterns and trends
- **Normalized Price Comparison**: Side-by-side pair analysis
<img width="1467" height="866" alt="image" src="https://github.com/user-attachments/assets/5952b7da-b3ec-48b0-a923-be77d4bf96d2" />
<img width="1451" height="763" alt="image" src="https://github.com/user-attachments/assets/d9f3770b-7658-4fa9-b57c-4cb596629d71" />
<img width="1467" height="840" alt="image" src="https://github.com/user-attachments/assets/4a56dcd0-da34-4c61-9644-c78238222b54" />




### Backtesting Engine
- **Mean-Reversion Strategy**: Simulate pairs trading with configurable entry/exit z-score thresholds
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate, average win/loss
<img width="1457" height="847" alt="image" src="https://github.com/user-attachments/assets/61cc39d7-ea35-4725-b83d-c4416dcbcea9" />
<img width="1410" height="532" alt="image" src="https://github.com/user-attachments/assets/09f47e84-7468-4679-a6ea-dfbb5809316b" />


- **Equity Curve**: Visual representation of strategy performance over time
- **Trade History**: Detailed log of all executed trades with timestamps and PnL
- **Configurable Parameters**: Adjustable initial capital, entry/exit thresholds

### Alert System
- **Custom Alert Rules**: Define threshold-based alerts
<img width="1446" height="786" alt="image" src="https://github.com/user-attachments/assets/f446f407-6056-48b9-952e-9f9c612abd55" />

- **Multi-Condition Alerts**: Complex rules combining multiple metrics with AND/OR logic
- **Multiple Metrics**: Alerts for z-score, price, spread, correlation
- **Webhook Notifications**: Send real-time alerts to external services (Slack, Discord, custom endpoints)
- **Alert History**: Track all triggered alerts with timestamps
<img width="1432" height="406" alt="image" src="https://github.com/user-attachments/assets/c541694f-835e-459f-ba35-e0d31a939f49" />

- **Database Persistence**: All alerts saved for analysis

### Data Management
- **SQLite Database**: Efficient local storage for tick and OHLCV data
<img width="1447" height="853" alt="image" src="https://github.com/user-attachments/assets/fa8ea58b-a142-4069-9d42-c32a4341a6d5" />

- **Multi-timeframe Resampling**: 1s, 1m, 5m, 15m, 1h intervals
- **CSV Export**: Download processed data and analytics
<img width="1452" height="855" alt="image" src="https://github.com/user-attachments/assets/3366a7cd-7007-489e-b1ab-4d1c221df46c" />
<img width="1462" height="350" alt="image" src="https://github.com/user-attachments/assets/f49b2ced-0e95-4ae6-95b0-9a6f3fa640b2" />

<img width="1453" height="670" alt="image" src="https://github.com/user-attachments/assets/fdbbc894-e806-451f-a67c-73dbc23a406f" />

- **Summary Statistics**: Time-series aggregation and export



### External Integration
- **WebSocket Server API**: Flask-based server accepting tick data from external sources
- **REST API Endpoints**: HTTP endpoints for single tick and batch data ingestion
- **Real-time Broadcasting**: WebSocket events for connected clients
- **Multiple Data Sources**: Support for Binance WebSocket, file uploads, and external APIs

## 🏗️ Architecture

### Modular Design

The application follows a clean, modular architecture with clear separation of concerns:

```
├── app.py                      # Main Streamlit application
├── websocket_server.py         # External data ingestion server
├── modules/
│   ├── database.py            # SQLite database layer
│   ├── data_ingestion.py      # WebSocket client & file loaders
│   ├── analytics.py           # Statistical analysis engine
│   ├── resampler.py           # Time-series resampling
│   ├── alerts.py              # Alert management system
│   └── backtesting.py         # Mean-reversion backtesting engine
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
