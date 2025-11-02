import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeanReversionBacktest:
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.0, 
                 stop_loss: float = None, take_profit: float = None):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades: List[Dict] = []
        self.positions: List[Dict] = []
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict:
        if df.empty or 'zscore' not in df.columns:
            return {}
        
        try:
            df = df.copy().reset_index(drop=True)
            
            capital = initial_capital
            position = 0
            entry_price_x = 0
            entry_price_y = 0
            entry_time = None
            entry_zscore = 0
            entry_capital = 0
            position_size = 0
            
            equity_curve = []
            trade_pnl = []
            
            for i in range(len(df)):
                zscore = df.loc[i, 'zscore']
                price_x = df.loc[i, 'price1']
                price_y = df.loc[i, 'price2']
                timestamp = df.loc[i, 'timestamp']
                hedge_ratio = df.loc[i, 'hedge_ratio'] if 'hedge_ratio' in df.columns else 1.0
                
                if np.isnan(zscore) or np.isnan(price_x) or np.isnan(price_y):
                    equity_curve.append(capital)
                    continue
                
                current_pnl = 0
                if position != 0:
                    if position == 1:
                        current_pnl = position_size * ((price_y - entry_price_y) - hedge_ratio * (price_x - entry_price_x))
                    else:
                        current_pnl = position_size * (-(price_y - entry_price_y) + hedge_ratio * (price_x - entry_price_x))
                
                if position == 0:
                    if zscore > self.entry_threshold:
                        position = -1
                        entry_price_x = price_x
                        entry_price_y = price_y
                        entry_time = timestamp
                        entry_zscore = zscore
                        entry_capital = capital
                        
                        position_value = entry_capital * 0.95
                        position_size = position_value / (entry_price_x * hedge_ratio + entry_price_y)
                        
                        self.positions.append({
                            'timestamp': timestamp,
                            'action': 'ENTER_SHORT',
                            'zscore': zscore,
                            'price_x': price_x,
                            'price_y': price_y
                        })
                    
                    elif zscore < -self.entry_threshold:
                        position = 1
                        entry_price_x = price_x
                        entry_price_y = price_y
                        entry_time = timestamp
                        entry_zscore = zscore
                        entry_capital = capital
                        
                        position_value = entry_capital * 0.95
                        position_size = position_value / (entry_price_x * hedge_ratio + entry_price_y)
                        
                        self.positions.append({
                            'timestamp': timestamp,
                            'action': 'ENTER_LONG',
                            'zscore': zscore,
                            'price_x': price_x,
                            'price_y': price_y
                        })
                
                elif position != 0:
                    exit_signal = False
                    exit_reason = None
                    
                    if (position == 1 and zscore >= self.exit_threshold) or \
                       (position == -1 and zscore <= self.exit_threshold):
                        exit_signal = True
                        exit_reason = 'MEAN_REVERSION'
                    
                    if self.stop_loss and current_pnl < -self.stop_loss:
                        exit_signal = True
                        exit_reason = 'STOP_LOSS'
                    
                    if self.take_profit and current_pnl > self.take_profit:
                        exit_signal = True
                        exit_reason = 'TAKE_PROFIT'
                    
                    if exit_signal:
                        capital += current_pnl
                        trade_pnl.append(current_pnl)
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'entry_zscore': entry_zscore,
                            'exit_zscore': zscore,
                            'direction': 'LONG' if position == 1 else 'SHORT',
                            'pnl': current_pnl,
                            'exit_reason': exit_reason
                        })
                        
                        self.positions.append({
                            'timestamp': timestamp,
                            'action': 'EXIT',
                            'zscore': zscore,
                            'price_x': price_x,
                            'price_y': price_y,
                            'pnl': current_pnl,
                            'reason': exit_reason
                        })
                        
                        position = 0
                        entry_price_x = 0
                        entry_price_y = 0
                        entry_time = None
                        entry_zscore = 0
                        entry_capital = 0
                        position_size = 0
                
                equity_curve.append(capital + (current_pnl if position != 0 else 0))
            
            if position != 0:
                final_price_x = df.loc[len(df)-1, 'price1']
                final_price_y = df.loc[len(df)-1, 'price2']
                hedge_ratio = df.loc[len(df)-1, 'hedge_ratio'] if 'hedge_ratio' in df.columns else 1.0
                
                if position == 1:
                    final_pnl = position_size * ((final_price_y - entry_price_y) - hedge_ratio * (final_price_x - entry_price_x))
                else:
                    final_pnl = position_size * (-(final_price_y - entry_price_y) + hedge_ratio * (final_price_x - entry_price_x))
                
                capital += final_pnl
                trade_pnl.append(final_pnl)
                
                self.trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.loc[len(df)-1, 'timestamp'],
                    'entry_zscore': entry_zscore,
                    'exit_zscore': df.loc[len(df)-1, 'zscore'],
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': final_pnl,
                    'exit_reason': 'FORCED_EXIT'
                })
            
            if len(trade_pnl) == 0:
                return {
                    'total_trades': 0,
                    'final_capital': initial_capital,
                    'total_return': 0.0,
                    'message': 'No trades executed'
                }
            
            returns = pd.Series(equity_curve).pct_change().dropna()
            
            total_return = (capital - initial_capital) / initial_capital * 100
            winning_trades = [t for t in trade_pnl if t > 0]
            losing_trades = [t for t in trade_pnl if t < 0]
            
            win_rate = len(winning_trades) / len(trade_pnl) * 100 if trade_pnl else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            return {
                'total_trades': len(trade_pnl),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_return': total_return,
                'final_capital': capital,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'equity_curve': equity_curve,
                'trade_pnl': trade_pnl
            }
        
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {}
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_positions_dataframe(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame(self.positions)
