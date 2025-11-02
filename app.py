import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
from datetime import datetime, timedelta
import io
import queue

from modules.database import DatabaseManager
from modules.data_ingestion import BinanceWebSocketClient, FileDataLoader
from modules.analytics import AnalyticsEngine
from modules.resampler import DataResampler
from modules.alerts import AlertManager, AlertRule
from modules.backtesting import MeanReversionBacktest

st.set_page_config(
    page_title="Crypto Quant Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_database():
    return DatabaseManager()

@st.cache_resource
def get_alert_manager():
    return AlertManager()

@st.cache_resource
def get_data_queue():
    return queue.Queue()

def initialize_session_state():
    if 'ws_client' not in st.session_state:
        st.session_state.ws_client = None
    if 'ws_running' not in st.session_state:
        st.session_state.ws_running = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.utcnow()
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = ['BTCUSDT', 'ETHUSDT']
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

initialize_session_state()

db = get_database()
alert_manager = get_alert_manager()
data_queue = get_data_queue()

def on_tick_received(tick_data):
    data_queue.put(tick_data)

def process_queued_data():
    ticks = []
    try:
        while True:
            tick = data_queue.get_nowait()
            ticks.append(tick)
            if len(ticks) >= 10:
                break
    except queue.Empty:
        pass
    
    if ticks:
        db.insert_ticks_batch(ticks)
        st.session_state.last_update = datetime.utcnow()
        return len(ticks)
    return 0

if st.session_state.ws_running:
    process_queued_data()

st.title("📈 Cryptocurrency Quantitative Analytics Dashboard")
st.markdown("Real-time analytics for statistical arbitrage and pairs trading")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    data_source = st.selectbox(
        "Data Source",
        ["Live WebSocket (Binance)", "Upload File (NDJSON/CSV)", "View Stored Data"],
        help="Choose your data source"
    )
    
    st.divider()
    
    if data_source == "Live WebSocket (Binance)":
        st.subheader("WebSocket Settings")
        
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="btcusdt,ethusdt",
            help="Enter crypto symbols in lowercase"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Stream", use_container_width=True):
                if not st.session_state.ws_running:
                    symbols = [s.strip().lower() for s in symbols_input.split(',')]
                    st.session_state.selected_symbols = [s.upper() for s in symbols]
                    
                    st.session_state.ws_client = BinanceWebSocketClient(
                        symbols=symbols,
                        on_message_callback=on_tick_received
                    )
                    st.session_state.ws_client.start()
                    st.session_state.ws_running = True
                    st.success(f"Started streaming {len(symbols)} symbols")
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Stream", use_container_width=True):
                if st.session_state.ws_running and st.session_state.ws_client:
                    st.session_state.ws_client.stop()
                    st.session_state.ws_running = False
                    st.session_state.ws_client = None
                    st.info("Stream stopped")
                    st.rerun()
        
        if st.session_state.ws_running:
            st.success("🟢 WebSocket Active")
        else:
            st.info("⚪ WebSocket Inactive")
    
    elif data_source == "Upload File (NDJSON/CSV)":
        st.subheader("File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['ndjson', 'jsonl', 'csv'],
            help="Upload tick data in NDJSON or CSV format"
        )
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read()
                df = FileDataLoader.detect_and_load(file_content, uploaded_file.name)
                
                st.success(f"Loaded {len(df)} records from {uploaded_file.name}")
                
                if st.button("💾 Save to Database", use_container_width=True):
                    ticks = []
                    for _, row in df.iterrows():
                        ticks.append({
                            'symbol': row['symbol'],
                            'ts': row['timestamp'].isoformat(),
                            'price': row['price'],
                            'size': row['size'],
                            'source': 'file_upload'
                        })
                    
                    db.insert_ticks_batch(ticks)
                    st.session_state.selected_symbols = df['symbol'].unique().tolist()
                    st.success(f"Saved {len(ticks)} ticks to database")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    st.divider()
    st.subheader("📊 Database Info")
    summary = db.get_data_summary()
    st.metric("Total Ticks", f"{summary['total_ticks']:,}")
    st.metric("Unique Symbols", summary['unique_symbols'])
    st.metric("OHLCV Bars", f"{summary['total_ohlcv_bars']:,}")
    st.metric("Alerts", summary['total_alerts'])
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        db.clear_all_data()
        st.success("All data cleared")
        st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Pair Analysis", 
    "📊 Price Charts", 
    "🔔 Alerts", 
    "📋 Summary Stats",
    "⚙️ Advanced Analytics"
])

with tab1:
    st.header("Pair Analysis & Statistical Arbitrage")
    
    col1, col2, col3 = st.columns(3)
    
    available_symbols = db.get_tick_data(limit=1000)['symbol'].unique().tolist() if not db.get_tick_data(limit=1).empty else st.session_state.selected_symbols
    
    with col1:
        symbol1 = st.selectbox("Symbol 1", available_symbols, index=0 if len(available_symbols) > 0 else 0)
    
    with col2:
        symbol2 = st.selectbox("Symbol 2", available_symbols, index=1 if len(available_symbols) > 1 else 0)
    
    with col3:
        timeframe = st.selectbox("Timeframe", ['1s', '1m', '5m', '15m'], index=1)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rolling_window = st.slider("Rolling Window", 5, 100, 20, help="Window for z-score and correlation")
    
    with col2:
        regression_type = st.selectbox("Regression Type", ["OLS", "Robust (Huber)", "Robust (Theil-Sen)", "Kalman Filter"])
    
    with col3:
        run_adf = st.checkbox("Run ADF Test", value=False)
    
    with col4:
        if regression_type == "Kalman Filter":
            kalman_delta = st.number_input("Kalman Δ", value=1e-5, format="%.1e", help="Transition covariance")
    
    if symbol1 and symbol2 and symbol1 != symbol2:
        tick_data = db.get_tick_data(limit=5000)
        
        if not tick_data.empty and symbol1 in tick_data['symbol'].values and symbol2 in tick_data['symbol'].values:
            resampled_df = DataResampler.resample_ticks_to_ohlcv(tick_data, timeframe)
            
            if not resampled_df.empty:
                db.insert_ohlcv_batch([{
                    'symbol': row['symbol'],
                    'timestamp': row['timestamp'].isoformat(),
                    'timeframe': row['timeframe'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                } for _, row in resampled_df.iterrows()])
            
            ohlcv1 = db.get_ohlcv_data(symbol1, timeframe, limit=500)
            ohlcv2 = db.get_ohlcv_data(symbol2, timeframe, limit=500)
            
            if not ohlcv1.empty and not ohlcv2.empty:
                merged = pd.merge(
                    ohlcv1[['timestamp', 'close']].rename(columns={'close': 'price1'}),
                    ohlcv2[['timestamp', 'close']].rename(columns={'close': 'price2'}),
                    on='timestamp',
                    how='inner'
                )
                
                if len(merged) >= 2:
                    if regression_type == "Kalman Filter":
                        kalman_df = AnalyticsEngine.calculate_kalman_hedge_ratio(
                            merged['price1'], merged['price2'], delta=kalman_delta if regression_type == "Kalman Filter" else 1e-5
                        )
                        
                        if not kalman_df.empty:
                            merged['hedge_ratio'] = kalman_df['hedge_ratio']
                            merged['intercept'] = kalman_df['intercept']
                            merged['spread'] = merged.apply(
                                lambda row: row['price2'] - (row['hedge_ratio'] * row['price1'] + row['intercept']),
                                axis=1
                            )
                            
                            hedge_ratio = kalman_df['hedge_ratio'].iloc[-1]
                            intercept = kalman_df['intercept'].iloc[-1]
                            regression_result = {
                                'hedge_ratio': hedge_ratio,
                                'intercept': intercept,
                                'method': 'Kalman Filter'
                            }
                        else:
                            regression_result = None
                    else:
                        if regression_type == "OLS":
                            regression_result = AnalyticsEngine.calculate_ols_regression(
                                merged['price1'], merged['price2']
                            )
                        elif regression_type == "Robust (Theil-Sen)":
                            regression_result = AnalyticsEngine.calculate_robust_regression(
                                merged['price1'], merged['price2'], method='theil-sen'
                            )
                        else:
                            regression_result = AnalyticsEngine.calculate_robust_regression(
                                merged['price1'], merged['price2'], method='huber'
                            )
                        
                        if regression_result:
                            hedge_ratio = regression_result['hedge_ratio']
                            intercept = regression_result.get('intercept', 0)
                            merged['spread'] = AnalyticsEngine.calculate_spread(
                                merged['price1'], merged['price2'], hedge_ratio, intercept
                            )
                    
                    if regression_result:
                        merged['zscore'] = AnalyticsEngine.calculate_zscore(merged['spread'], rolling_window)
                        merged['correlation'] = AnalyticsEngine.calculate_rolling_correlation(
                            merged['price1'], merged['price2'], rolling_window
                        )
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
                        
                        with col2:
                            st.metric("R²", f"{regression_result.get('r_squared', 0):.4f}")
                        
                        with col3:
                            current_zscore = merged['zscore'].iloc[-1] if not merged['zscore'].isna().all() else 0
                            st.metric("Current Z-Score", f"{current_zscore:.2f}")
                            alert_manager.check_all_rules('Z-Score', current_zscore, f"{symbol1}/{symbol2}")
                        
                        with col4:
                            current_corr = merged['correlation'].iloc[-1] if not merged['correlation'].isna().all() else 0
                            st.metric("Correlation", f"{current_corr:.4f}")
                            alert_manager.check_all_rules('Correlation', current_corr, f"{symbol1}/{symbol2}")
                        
                        current_spread = merged['spread'].iloc[-1] if not merged['spread'].isna().all() else 0
                        alert_manager.check_all_rules('Spread', current_spread, f"{symbol1}/{symbol2}")
                        
                        fig = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=("Normalized Prices", "Spread", "Z-Score"),
                            vertical_spacing=0.08,
                            row_heights=[0.4, 0.3, 0.3]
                        )
                        
                        norm_price1 = merged['price1'] / merged['price1'].iloc[0] * 100
                        norm_price2 = merged['price2'] / merged['price2'].iloc[0] * 100
                        
                        fig.add_trace(
                            go.Scatter(x=merged['timestamp'], y=norm_price1, name=symbol1, line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=merged['timestamp'], y=norm_price2, name=symbol2, line=dict(color='orange')),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=merged['timestamp'], y=merged['spread'], name='Spread', line=dict(color='green')),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=merged['timestamp'], y=merged['zscore'], name='Z-Score', line=dict(color='red')),
                            row=3, col=1
                        )
                        
                        fig.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                        fig.add_hline(y=-2, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
                        
                        fig.update_layout(height=800, showlegend=True, hovermode='x unified')
                        fig.update_xaxes(title_text="Time", row=3, col=1)
                        fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
                        fig.update_yaxes(title_text="Spread", row=2, col=1)
                        fig.update_yaxes(title_text="Z-Score", row=3, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if regression_type == "Kalman Filter" and 'hedge_ratio' in merged.columns:
                            st.subheader("Dynamic Hedge Ratio (Kalman Filter)")
                            
                            fig_kalman = go.Figure()
                            fig_kalman.add_trace(go.Scatter(
                                x=merged['timestamp'],
                                y=merged['hedge_ratio'],
                                mode='lines',
                                name='Hedge Ratio',
                                line=dict(color='purple', width=2)
                            ))
                            
                            fig_kalman.update_layout(
                                title="Time-Varying Hedge Ratio",
                                xaxis_title="Time",
                                yaxis_title="Hedge Ratio",
                                height=300,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_kalman, use_container_width=True)
                        
                        st.divider()
                        st.subheader("📊 Mean-Reversion Backtest")
                        
                        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                        
                        with bcol1:
                            entry_z = st.number_input("Entry Z-Score", value=2.0, min_value=0.5, max_value=5.0, step=0.1)
                        
                        with bcol2:
                            exit_z = st.number_input("Exit Z-Score", value=0.0, min_value=-1.0, max_value=1.0, step=0.1)
                        
                        with bcol3:
                            initial_cap = st.number_input("Initial Capital", value=100000, min_value=1000, step=10000)
                        
                        with bcol4:
                            run_backtest = st.button("▶️ Run Backtest", use_container_width=True)
                        
                        if run_backtest:
                            backtest_engine = MeanReversionBacktest(
                                entry_threshold=entry_z,
                                exit_threshold=exit_z
                            )
                            
                            results = backtest_engine.backtest(merged, initial_capital=initial_cap)
                            
                            if results and 'total_trades' in results and results['total_trades'] > 0:
                                st.success(f"Backtest Complete - {results['total_trades']} trades executed")
                                
                                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                                
                                with mcol1:
                                    st.metric("Total Return", f"{results['total_return']:.2f}%")
                                
                                with mcol2:
                                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                                
                                with mcol3:
                                    st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                                
                                with mcol4:
                                    st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                                
                                with mcol5:
                                    st.metric("Avg Win/Loss", f"{results['avg_win']:.2f} / {results['avg_loss']:.2f}")
                                
                                fig_equity = go.Figure()
                                fig_equity.add_trace(go.Scatter(
                                    x=merged['timestamp'],
                                    y=results['equity_curve'],
                                    mode='lines',
                                    name='Equity',
                                    line=dict(color='green', width=2)
                                ))
                                
                                fig_equity.update_layout(
                                    title="Equity Curve",
                                    xaxis_title="Time",
                                    yaxis_title="Capital ($)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_equity, use_container_width=True)
                                
                                trades_df = backtest_engine.get_trades_dataframe()
                                if not trades_df.empty:
                                    st.write("Trade History:")
                                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
                                    
                                    trades_csv = trades_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Trades (CSV)",
                                        data=trades_csv,
                                        file_name=f"backtest_trades_{symbol1}_{symbol2}.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.info(results.get('message', 'No trades executed with current parameters'))
                        
                        if run_adf:
                            st.subheader("Stationarity Test (ADF)")
                            adf_result = AnalyticsEngine.perform_adf_test(merged['spread'])
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("ADF Statistic", f"{adf_result.get('adf_statistic', 0):.4f}" if adf_result.get('adf_statistic') else "N/A")
                            
                            with col2:
                                st.metric("P-Value", f"{adf_result.get('p_value', 0):.4f}" if adf_result.get('p_value') else "N/A")
                            
                            with col3:
                                is_stationary = adf_result.get('is_stationary', False)
                                st.metric("Stationary", "✅ Yes" if is_stationary else "❌ No")
                            
                            if 'critical_values' in adf_result:
                                st.write("Critical Values:", adf_result['critical_values'])
                        
                        csv_data = merged.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Pair Analysis (CSV)",
                            data=csv_data,
                            file_name=f"pair_analysis_{symbol1}_{symbol2}_{timeframe}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Insufficient data for regression analysis")
                else:
                    st.info(f"Waiting for more data points (currently: {len(merged)})")
            else:
                st.info("Resampling tick data... Please wait or start the WebSocket stream")
        else:
            st.info("No tick data available. Please start the WebSocket stream or upload a file")

with tab2:
    st.header("Price Charts & Volume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_symbol = st.selectbox("Select Symbol", available_symbols, key='chart_symbol')
    
    with col2:
        chart_timeframe = st.selectbox("Chart Timeframe", ['1s', '1m', '5m', '15m'], index=1, key='chart_timeframe')
    
    if chart_symbol:
        ohlcv_data = db.get_ohlcv_data(chart_symbol, chart_timeframe, limit=500)
        
        if not ohlcv_data.empty:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f"{chart_symbol} Price (Candlestick)", "Volume"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(
                go.Candlestick(
                    x=ohlcv_data['timestamp'],
                    open=ohlcv_data['open'],
                    high=ohlcv_data['high'],
                    low=ohlcv_data['low'],
                    close=ohlcv_data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=ohlcv_data['timestamp'], y=ohlcv_data['volume'], name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
            
            fig.update_layout(height=700, showlegend=False, xaxis_rangeslider_visible=False)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            stats = AnalyticsEngine.calculate_price_stats(
                db.get_tick_data(symbol=chart_symbol, limit=1000), 
                chart_symbol
            )
            
            if stats:
                st.subheader("Price Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Mean", f"${stats.get('mean', 0):,.2f}")
                with col2:
                    st.metric("Median", f"${stats.get('median', 0):,.2f}")
                with col3:
                    st.metric("Std Dev", f"${stats.get('std', 0):,.2f}")
                with col4:
                    st.metric("Min", f"${stats.get('min', 0):,.2f}")
                with col5:
                    st.metric("Max", f"${stats.get('max', 0):,.2f}")
                
                if stats.get('current'):
                    alert_manager.check_all_rules('Price', stats['current'], chart_symbol)
        else:
            st.info("No OHLCV data available for this symbol and timeframe")

with tab3:
    st.header("Alert Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create New Alert")
        
        acol1, acol2, acol3, acol4 = st.columns(4)
        
        with acol1:
            alert_metric = st.selectbox("Metric", ["Z-Score", "Price", "Spread", "Correlation"])
        
        with acol2:
            alert_condition = st.selectbox("Condition", [">", "<", ">=", "<="])
        
        with acol3:
            alert_threshold = st.number_input("Threshold", value=2.0, step=0.1)
        
        with acol4:
            alert_symbol = st.text_input("Symbol (optional)", value="", placeholder="e.g., BTCUSDT")
        
        if st.button("➕ Add Alert", use_container_width=False):
            rule_id = f"alert_{len(alert_manager.rules)}_{int(time.time())}"
            
            def alert_callback(rule, value, symbol):
                db.insert_alert(
                    alert_type=rule.name,
                    message=f"{rule.name} {rule.condition} {rule.threshold} (value: {value:.4f})",
                    symbol=symbol,
                    value=value
                )
            
            rule = AlertRule(
                rule_id=rule_id,
                name=alert_metric,
                condition=alert_condition,
                threshold=alert_threshold,
                symbol=alert_symbol.upper() if alert_symbol else None,
                callback=alert_callback
            )
            
            alert_manager.add_rule(rule)
            st.success(f"Alert created: {alert_metric} {alert_condition} {alert_threshold}")
            st.rerun()
    
    with col2:
        st.subheader("Active Alerts")
        rules = alert_manager.get_all_rules()
        st.metric("Total Active", len(rules))
    
    st.divider()
    
    st.subheader("Active Alert Rules")
    if alert_manager.rules:
        rules_df = pd.DataFrame(alert_manager.get_all_rules())
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
    else:
        st.info("No active alert rules")
    
    st.subheader("Recent Alerts (Database)")
    recent_alerts = db.get_recent_alerts(limit=50)
    
    if not recent_alerts.empty:
        st.dataframe(recent_alerts, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts triggered yet")

with tab4:
    st.header("Summary Statistics & Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        summary_symbol = st.selectbox("Symbol", available_symbols, key='summary_symbol')
    
    with col2:
        summary_timeframe = st.selectbox("Aggregation", ['1min', '5min', '15min', '1H'], index=0)
    
    if summary_symbol:
        tick_data = db.get_tick_data(symbol=summary_symbol, limit=10000)
        
        if not tick_data.empty:
            summary_df = AnalyticsEngine.generate_summary_stats(tick_data, summary_symbol, summary_timeframe)
            
            if not summary_df.empty:
                st.subheader(f"Time-Series Statistics ({summary_symbol})")
                
                display_df = summary_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'return', 'volatility']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)
                
                csv_export = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary Statistics (CSV)",
                    data=csv_export,
                    file_name=f"summary_stats_{summary_symbol}_{summary_timeframe}.csv",
                    mime="text/csv"
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=summary_df['timestamp'],
                    y=summary_df['return'].fillna(0),
                    mode='lines',
                    name='Returns (%)',
                    line=dict(color='purple')
                ))
                
                fig.update_layout(
                    title=f"{summary_symbol} Returns Over Time",
                    xaxis_title="Time",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for summary statistics")
        else:
            st.info("No data available for this symbol")

with tab5:
    st.header("Advanced Analytics")
    
    st.subheader("Correlation Heatmap")
    
    heatmap_timeframe = st.selectbox("Timeframe for Heatmap", ['1m', '5m', '15m'], index=1)
    
    tick_data_all = db.get_tick_data(limit=10000)
    
    if not tick_data_all.empty and len(tick_data_all['symbol'].unique()) > 1:
        resampled_all = DataResampler.resample_ticks_to_ohlcv(tick_data_all, heatmap_timeframe)
        
        if not resampled_all.empty:
            pivot_data = resampled_all.pivot(index='timestamp', columns='symbol', values='close')
            
            if len(pivot_data.columns) > 1:
                corr_matrix = pivot_data.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title="Symbol Correlation Matrix",
                    xaxis_title="Symbol",
                    yaxis_title="Symbol",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 symbols for correlation analysis")
        else:
            st.info("Resampling data...")
    else:
        st.info("Need data from multiple symbols for correlation heatmap")
    
    st.divider()
    
    st.subheader("Volatility Analysis")
    
    vol_symbol = st.selectbox("Symbol for Volatility", available_symbols, key='vol_symbol')
    
    if vol_symbol:
        tick_data_vol = db.get_tick_data(symbol=vol_symbol, limit=5000)
        
        if not tick_data_vol.empty and len(tick_data_vol) > 20:
            tick_data_vol = tick_data_vol.sort_values('timestamp')
            
            volatility = AnalyticsEngine.calculate_volatility(tick_data_vol['price'], window=20)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tick_data_vol['timestamp'],
                y=volatility,
                mode='lines',
                name='Annualized Volatility',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"{vol_symbol} Rolling Volatility (20-period)",
                xaxis_title="Time",
                yaxis_title="Volatility",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            current_vol = volatility.iloc[-1] if not volatility.isna().all() else 0
            st.metric("Current Volatility", f"{current_vol:.4f}")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

with col2:
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

with col3:
    st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

if st.session_state.auto_refresh:
    time.sleep(5)
    st.rerun()
