"""
UI Helper Functions for Enhanced User Experience

This module provides utility functions for:
- Onboarding and tutorial system
- Alert notifications
- Error handling and user-friendly messages
- Export functionality
- Timezone handling
"""

import streamlit as st
from datetime import datetime
import pytz
from typing import Dict, List, Optional, Any
import base64
import io


class OnboardingSystem:
    """Interactive onboarding tutorial system for new users"""
    
    TUTORIAL_STEPS = [
        {
            'step': 1,
            'title': '🎯 Welcome to Crypto Quant Analytics!',
            'content': '''
            This dashboard helps you analyze cryptocurrency pairs for statistical arbitrage and pairs trading.
            
            **Key Features:**
            - Real-time data streaming from Binance
            - Statistical analysis (correlation, regression, stationarity)
            - Automated alerts and backtesting
            - Multiple visualization types
            
            Let's get you started with a quick tour!
            ''',
            'action': None
        },
        {
            'step': 2,
            'title': '📊 Step 1: Choose Your Data Source',
            'content': '''
            You have three options to get data into the system:
            
            1. **Live WebSocket (Binance)**: Stream real-time tick data directly
            2. **Browser Stream URL**: Use external WebSocket server on port 5001
            3. **Upload File**: Import historical data from NDJSON or CSV files
            
            **Tip**: For testing, try uploading a sample file first!
            ''',
            'action': 'data_source'
        },
        {
            'step': 3,
            'title': '🔍 Step 2: Select Trading Pairs',
            'content': '''
            For pairs trading analysis, you'll need at least 2 symbols:
            
            **Popular Pairs:**
            - BTC/ETH (Bitcoin vs Ethereum)
            - BTC/BNB (Bitcoin vs Binance Coin)
            - ETH/BNB (Ethereum vs Binance Coin)
            
            The system will analyze correlations and calculate spreads between your chosen pairs.
            ''',
            'action': 'symbols'
        },
        {
            'step': 4,
            'title': '📈 Step 3: Explore Analytics',
            'content': '''
            Navigate through different tabs to analyze your data:
            
            - **Dashboard**: Live metrics and quick overview
            - **Analytics**: Deep dive into regression and spreads
            - **Correlation Matrix**: Visualize pair relationships
            - **Alerts**: Set up automated notifications
            - **Data Management**: Export and manage your data
            
            Start with the Dashboard tab to see live updates!
            ''',
            'action': 'tabs'
        },
        {
            'step': 5,
            'title': '🔔 Step 4: Configure Alerts',
            'content': '''
            Set up alerts to monitor trading opportunities:
            
            1. Go to the **Alerts** tab
            2. Define conditions (z-score, spread, correlation)
            3. Add webhooks for external notifications (optional)
            
            **Example**: Alert when z-score > 2 (indicating potential mean reversion)
            ''',
            'action': 'alerts'
        },
        {
            'step': 6,
            'title': '✅ You\'re All Set!',
            'content': '''
            **Quick Tips:**
            - Use presets for common trading pairs (see sidebar)
            - Enable dark mode for comfortable viewing
            - Export reports for offline analysis
            - Check timezone settings for accurate timestamps
            
            You can restart this tutorial anytime from the sidebar.
            
            Happy trading! 🚀
            ''',
            'action': 'finish'
        }
    ]
    
    @staticmethod
    def initialize_onboarding():
        """Initialize onboarding session state"""
        if 'onboarding_complete' not in st.session_state:
            st.session_state.onboarding_complete = False
        if 'onboarding_step' not in st.session_state:
            st.session_state.onboarding_step = 1
        if 'show_onboarding' not in st.session_state:
            st.session_state.show_onboarding = not st.session_state.onboarding_complete
    
    @staticmethod
    def show_tutorial():
        """Display the onboarding tutorial modal"""
        if not st.session_state.show_onboarding:
            return
        
        current_step = st.session_state.onboarding_step
        step_data = OnboardingSystem.TUTORIAL_STEPS[current_step - 1]
        
        with st.container():
            st.markdown(f"### {step_data['title']}")
            st.markdown(step_data['content'])
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if current_step > 1:
                    if st.button("⬅️ Previous", key=f"prev_{current_step}"):
                        st.session_state.onboarding_step -= 1
                        st.rerun()
            
            with col2:
                if st.button("Skip Tutorial", key=f"skip_{current_step}"):
                    st.session_state.show_onboarding = False
                    st.session_state.onboarding_complete = True
                    st.rerun()
            
            with col3:
                if current_step < len(OnboardingSystem.TUTORIAL_STEPS):
                    if st.button("Next ➡️", key=f"next_{current_step}"):
                        st.session_state.onboarding_step += 1
                        st.rerun()
                else:
                    if st.button("Finish ✅", key=f"finish_{current_step}"):
                        st.session_state.show_onboarding = False
                        st.session_state.onboarding_complete = True
                        st.rerun()
            
            st.progress(current_step / len(OnboardingSystem.TUTORIAL_STEPS))
            st.caption(f"Step {current_step} of {len(OnboardingSystem.TUTORIAL_STEPS)}")


class AlertNotificationSystem:
    """Real-time alert notification system with badges and toasts"""
    
    @staticmethod
    def initialize_notifications():
        """Initialize notification session state"""
        if 'active_alerts' not in st.session_state:
            st.session_state.active_alerts = []
        if 'alert_count' not in st.session_state:
            st.session_state.alert_count = 0
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = datetime.utcnow()
    
    @staticmethod
    def add_alert(alert_message: str, alert_type: str = "info", metric: str = ""):
        """Add a new alert notification"""
        alert = {
            'message': alert_message,
            'type': alert_type,
            'metric': metric,
            'timestamp': datetime.utcnow()
        }
        st.session_state.active_alerts.append(alert)
        st.session_state.alert_count = len(st.session_state.active_alerts)
    
    @staticmethod
    def show_alert_badge():
        """Display alert count badge in sidebar"""
        count = st.session_state.get('alert_count', 0)
        if count > 0:
            st.sidebar.markdown(
                f"""
                <div style='background-color: #ff4b4b; color: white; padding: 8px; 
                            border-radius: 5px; text-align: center; font-weight: bold;'>
                    🔔 {count} Active Alert{'s' if count != 1 else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    @staticmethod
    def show_recent_alerts(limit: int = 5):
        """Display recent alerts in the UI"""
        alerts = st.session_state.get('active_alerts', [])
        
        if not alerts:
            st.info("No active alerts")
            return
        
        st.subheader(f"🔔 Recent Alerts ({len(alerts)})")
        
        for alert in alerts[-limit:]:
            alert_type = alert.get('type', 'info')
            icon = "🔴" if alert_type == "error" else "🟡" if alert_type == "warning" else "🟢"
            
            with st.expander(f"{icon} {alert['message'][:50]}...", expanded=False):
                st.write(f"**Message:** {alert['message']}")
                st.write(f"**Metric:** {alert.get('metric', 'N/A')}")
                st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        if st.button("Clear All Alerts"):
            st.session_state.active_alerts = []
            st.session_state.alert_count = 0
            st.rerun()
    
    @staticmethod
    def show_toast_notification(message: str, icon: str = "🔔"):
        """Show a toast notification"""
        st.toast(f"{icon} {message}", icon=icon)


class TimezoneManager:
    """Timezone management for localized timestamps"""
    
    COMMON_TIMEZONES = [
        'UTC',
        'US/Eastern',
        'US/Pacific',
        'Europe/London',
        'Europe/Paris',
        'Asia/Tokyo',
        'Asia/Shanghai',
        'Asia/Hong_Kong',
        'Asia/Singapore',
        'Australia/Sydney'
    ]
    
    @staticmethod
    def initialize_timezone():
        """Initialize timezone in session state"""
        if 'user_timezone' not in st.session_state:
            st.session_state.user_timezone = 'UTC'
    
    @staticmethod
    def convert_to_user_tz(dt: datetime, from_tz: str = 'UTC') -> datetime:
        """Convert datetime to user's timezone"""
        user_tz = st.session_state.get('user_timezone', 'UTC')
        
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        
        if dt.tzinfo is None:
            dt = pytz.timezone(from_tz).localize(dt)
        
        return dt.astimezone(pytz.timezone(user_tz))
    
    @staticmethod
    def format_timestamp(dt: datetime, include_tz: bool = True) -> str:
        """Format timestamp with user's timezone"""
        converted = TimezoneManager.convert_to_user_tz(dt)
        if include_tz:
            return converted.strftime('%Y-%m-%d %H:%M:%S %Z')
        return converted.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def show_timezone_selector():
        """Display timezone selector in sidebar"""
        selected_tz = st.selectbox(
            "🌍 Timezone",
            options=TimezoneManager.COMMON_TIMEZONES,
            index=TimezoneManager.COMMON_TIMEZONES.index(st.session_state.get('user_timezone', 'UTC')),
            help="Select your preferred timezone for all timestamps"
        )
        st.session_state.user_timezone = selected_tz
        return selected_tz


class ErrorHandler:
    """Enhanced error handling with user-friendly messages"""
    
    ERROR_MESSAGES = {
        'websocket_connection': {
            'title': '🔌 WebSocket Connection Error',
            'message': 'Unable to connect to Binance WebSocket. Please check your internet connection and try again.',
            'troubleshooting': [
                'Check your internet connection',
                'Verify the symbol name is correct (e.g., btcusdt)',
                'Try refreshing the page',
                'Check if Binance is experiencing downtime'
            ]
        },
        'file_upload': {
            'title': '📁 File Upload Error',
            'message': 'There was a problem uploading your file. Please check the file format and try again.',
            'troubleshooting': [
                'Ensure file is in NDJSON or CSV format',
                'Check that required columns exist (timestamp, symbol, price, size)',
                'Verify file size is under 200MB',
                'Make sure timestamps are in valid ISO8601 format'
            ]
        },
        'data_processing': {
            'title': '⚙️ Data Processing Error',
            'message': 'An error occurred while processing your data. The data may be incomplete or malformed.',
            'troubleshooting': [
                'Check that your data has valid timestamps',
                'Ensure numeric fields (price, size) contain numbers',
                'Verify there are at least 2 data points',
                'Try resampling to a larger timeframe'
            ]
        },
        'insufficient_data': {
            'title': '📊 Insufficient Data',
            'message': 'Not enough data points to perform analysis. Please add more data or adjust your filters.',
            'troubleshooting': [
                'Upload more historical data',
                'Extend the live stream duration',
                'Use a larger timeframe for analysis',
                'Check your symbol filter settings'
            ]
        }
    }
    
    @staticmethod
    def show_error(error_type: str, details: str = ""):
        """Display user-friendly error message"""
        error_info = ErrorHandler.ERROR_MESSAGES.get(error_type, {
            'title': '⚠️ Error',
            'message': 'An unexpected error occurred.',
            'troubleshooting': ['Please try again or contact support']
        })
        
        st.error(f"### {error_info['title']}\n\n{error_info['message']}")
        
        if details:
            with st.expander("🔍 Technical Details"):
                st.code(details)
        
        with st.expander("🛠️ Troubleshooting Steps"):
            for step in error_info['troubleshooting']:
                st.write(f"• {step}")
    
    @staticmethod
    def handle_exception(func):
        """Decorator for graceful error handling"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                ErrorHandler.show_error('file_upload', str(e))
            except ValueError as e:
                ErrorHandler.show_error('data_processing', str(e))
            except ConnectionError as e:
                ErrorHandler.show_error('websocket_connection', str(e))
            except Exception as e:
                st.error(f"⚠️ Unexpected Error: {str(e)}")
                with st.expander("🔍 Full Error Details"):
                    st.exception(e)
        return wrapper


class ReportExporter:
    """Export analytics reports in HTML format"""
    
    @staticmethod
    def generate_html_report(
        summary_stats: Dict[str, Any],
        charts: List[Any],
        alerts: List[Dict],
        metadata: Dict[str, str]
    ) -> str:
        """Generate comprehensive HTML report"""
        
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Crypto Analytics Report - {metadata.get('title', 'Summary')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
                h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
                h2 {{ color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                .metric-box {{ display: inline-block; background: #f0f8ff; padding: 15px; margin: 10px; border-radius: 5px; border-left: 4px solid #1f77b4; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ background: #1f77b4; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background: #f5f5f5; }}
                .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .alert-warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .alert-info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Cryptocurrency Quantitative Analytics Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Analysis Period:</strong> {metadata.get('period', 'N/A')}</p>
                
                <h2>📊 Summary Statistics</h2>
                <div>
        """
        
        for key, value in summary_stats.items():
            html_content += f"""
                <div class="metric-box">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """
        
        html_content += """
                </div>
                
                <h2>🔔 Active Alerts</h2>
        """
        
        if alerts:
            for alert in alerts:
                html_content += f"""
                <div class="alert alert-warning">
                    <strong>{alert.get('metric', 'Alert')}:</strong> {alert.get('message', '')}
                    <br><small>{alert.get('timestamp', '')}</small>
                </div>
                """
        else:
            html_content += "<p>No active alerts</p>"
        
        html_content += """
                <h2>📋 Detailed Analysis</h2>
                <p>Charts and detailed analytics are available in the interactive dashboard.</p>
                
                <div class="footer">
                    <p>This report was generated by Crypto Quant Analytics Dashboard</p>
                    <p>For real-time updates and interactive analysis, please visit the dashboard.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    @staticmethod
    def create_download_link(html_content: str, filename: str = "crypto_analytics_report.html") -> str:
        """Create downloadable link for HTML report"""
        b64 = base64.b64encode(html_content.encode()).decode()
        return f'<a href="data:text/html;base64,{b64}" download="{filename}">📥 Download Report</a>'


class AnalysisPresets:
    """Default analysis presets for common trading pairs"""
    
    PRESETS = {
        'BTC/ETH Correlation': {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframe': '1m',
            'z_score_entry': 2.0,
            'z_score_exit': 0.5,
            'correlation_threshold': 0.7,
            'lookback_period': 100,
            'description': 'Classic BTC/ETH pairs trading with mean reversion'
        },
        'BTC/BNB Spread': {
            'symbols': ['BTCUSDT', 'BNBUSDT'],
            'timeframe': '5m',
            'z_score_entry': 2.5,
            'z_score_exit': 0.3,
            'correlation_threshold': 0.65,
            'lookback_period': 200,
            'description': 'BTC/BNB spread trading with wider z-score bands'
        },
        'ETH/BNB Momentum': {
            'symbols': ['ETHUSDT', 'BNBUSDT'],
            'timeframe': '1m',
            'z_score_entry': 1.8,
            'z_score_exit': 0.7,
            'correlation_threshold': 0.75,
            'lookback_period': 50,
            'description': 'Fast momentum strategy for ETH/BNB pair'
        },
        'Multi-Pair Basket': {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'timeframe': '15m',
            'z_score_entry': 2.2,
            'z_score_exit': 0.5,
            'correlation_threshold': 0.6,
            'lookback_period': 150,
            'description': 'Diversified basket approach with 3 major pairs'
        }
    }
    
    @staticmethod
    def get_preset_names() -> List[str]:
        """Get list of preset names"""
        return list(AnalysisPresets.PRESETS.keys())
    
    @staticmethod
    def apply_preset(preset_name: str) -> Dict[str, Any]:
        """Apply a preset configuration"""
        return AnalysisPresets.PRESETS.get(preset_name, {})
    
    @staticmethod
    def show_preset_selector():
        """Display preset selector in UI"""
        st.subheader("🎯 Quick Start Presets")
        
        preset_name = st.selectbox(
            "Choose a preset configuration",
            options=['Custom'] + AnalysisPresets.get_preset_names(),
            help="Select a pre-configured analysis template"
        )
        
        if preset_name != 'Custom':
            preset = AnalysisPresets.apply_preset(preset_name)
            st.info(f"ℹ️ {preset.get('description', '')}")
            
            if st.button("📋 Apply This Preset"):
                st.session_state.selected_symbols = preset['symbols']
                st.session_state.preset_config = preset
                st.success(f"✅ Applied preset: {preset_name}")
                st.rerun()
            
            with st.expander("📊 Preset Details"):
                st.json(preset)
        
        return preset_name
