from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertRule:
    def __init__(self, rule_id: str, name: str, condition: str, threshold: float, 
                 symbol: Optional[str] = None, callback: Optional[Callable] = None):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.symbol = symbol
        self.callback = callback
        self.triggered_count = 0
        self.last_triggered = None
    
    def check(self, value: float, symbol: Optional[str] = None) -> bool:
        if self.symbol and symbol and self.symbol != symbol:
            return False
        
        triggered = False
        
        if self.condition == '>':
            triggered = value > self.threshold
        elif self.condition == '<':
            triggered = value < self.threshold
        elif self.condition == '>=':
            triggered = value >= self.threshold
        elif self.condition == '<=':
            triggered = value <= self.threshold
        elif self.condition == '==':
            triggered = abs(value - self.threshold) < 1e-6
        
        if triggered:
            self.triggered_count += 1
            self.last_triggered = datetime.utcnow()
            
            if self.callback:
                try:
                    self.callback(self, value, symbol)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
        
        return triggered
    
    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'condition': self.condition,
            'threshold': self.threshold,
            'symbol': self.symbol,
            'triggered_count': self.triggered_count,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }


class AlertManager:
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Dict] = []
    
    def add_rule(self, rule: AlertRule):
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str):
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def check_all_rules(self, metric_name: str, value: float, symbol: Optional[str] = None):
        for rule in self.rules.values():
            if rule.name == metric_name:
                if rule.check(value, symbol):
                    alert_msg = f"{metric_name} {rule.condition} {rule.threshold} (current: {value:.4f})"
                    if symbol:
                        alert_msg = f"[{symbol}] " + alert_msg
                    
                    self.alert_history.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'message': alert_msg,
                        'value': value,
                        'symbol': symbol
                    })
                    
                    logger.info(f"ALERT TRIGGERED: {alert_msg}")
    
    def get_all_rules(self) -> List[Dict]:
        return [rule.to_dict() for rule in self.rules.values()]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        return self.alert_history[-limit:]
    
    def clear_history(self):
        self.alert_history = []
        logger.info("Alert history cleared")
