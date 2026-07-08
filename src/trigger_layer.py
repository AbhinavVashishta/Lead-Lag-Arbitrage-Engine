import time
from typing import Dict, Any, Optional
import numpy as np

class SignalTriggerLayer:
    """
    Component 3: Signal Evaluation & Trigger Layer.
    Acts as a quantitative risk gatekeeper that fires directional execution payloads
    only when structural confidence and leader momentum conditions are met.
    """
    def __init__(self, min_confidence: float = 0.80, min_velocity_threshold: float = 0.0005,
                 min_lag_seconds: float = 0.0):
        # Default minimum velocity threshold is set to 0.05% price move per bucket
        self.min_confidence = min_confidence
        self.min_velocity_threshold = min_velocity_threshold
        self.min_lag_seconds = min_lag_seconds
        self.total_signals_fired = 0

    def compute_leader_velocity(self, x: np.ndarray) -> float:
        """
        Calculates the fractional price velocity (return) of the Leader array:
        (x[t] - x[t - dt]) / x[t - dt]
        """
        if len(x) < 2:
            return 0.0
        
        p_current = x[-1]
        p_previous = x[-2]
        
        if p_previous == 0.0 or np.isnan(p_previous) or np.isnan(p_current):
            return 0.0
            
        return float((p_current - p_previous) / p_previous)

    def evaluate_signal(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        tau_max: float, 
        rho: float
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates real-time market state against gating thresholds.
        Returns an execution payload dictionary if all conditions are met, else None.
        """
        if tau_max <= self.min_lag_seconds:
            return None

        # Gate 1: Check structural correlation confidence
        if rho <= self.min_confidence:
            return None
            
        # Gate 2: Check Leader price velocity
        delta_p_leader = self.compute_leader_velocity(x)
        
        if abs(delta_p_leader) <= self.min_velocity_threshold:
            return None
            
        # Gate 3: Directional arbitrage determination
        lagger_current_price = y[-1]
        if np.isnan(lagger_current_price) or lagger_current_price == 0.0:
            return None

        # Anticipated correction target for the Lagger exchange
        anticipated_target_price = lagger_current_price * (1.0 + delta_p_leader)
        
        direction = "BUY" if delta_p_leader > 0 else "SELL"
        self.total_signals_fired += 1
        
        execution_payload = {
            "signal_id": self.total_signals_fired,
            "timestamp_perf": time.perf_counter(),
            "timestamp_utc": time.time(),
            "action": direction,
            "target_exchange": "Coinbase (Lagger)",
            "symbol": "BTC-USD",
            "confidence_rho": round(float(rho), 4),
            "detected_lag_ms": round(float(tau_max * 1000.0), 2),
            "leader_velocity_pct": round(float(delta_p_leader * 100.0), 4),
            "lagger_reference_price": round(float(lagger_current_price), 2),
            "anticipated_target_price": round(float(anticipated_target_price), 2)
        }
        
        return execution_payload