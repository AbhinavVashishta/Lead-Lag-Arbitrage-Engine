import time
from typing import Dict, Any, Optional
import numpy as np

class SignalTriggerLayer:
    """
    Fee-Aware Signal Trigger Layer.
    Acts as a quantitative risk gatekeeper that fires directional execution payloads
    only when structural confidence and momentum strictly exceed round-trip transaction costs.
    """
    def __init__(
        self, 
        min_confidence: float = 0.80, 
        taker_fee_rate: float = 0.0005, 
        alpha_buffer: float = 0.0005,
        min_lag_seconds: float = 0.0
    ):
        self.min_confidence = min_confidence
        self.taker_fee_rate = taker_fee_rate
        self.alpha_buffer = alpha_buffer
        self.min_lag_seconds = min_lag_seconds
        self.total_signals_fired = 0
        
        # Calculate the hard mathematical hurdle rate required to beat transaction costs
        self.round_trip_fee = 2.0 * self.taker_fee_rate
        self.dynamic_velocity_hurdle = self.round_trip_fee + self.alpha_buffer
        
        print(f"[Trigger Layer Init] Taker Fee: {self.taker_fee_rate*100:.2f}% | Round-Trip: {self.round_trip_fee*100:.2f}%")
        print(f"[Trigger Layer Init] Alpha Buffer: {self.alpha_buffer*100:.2f}% -> Hard Hurdle Gate: {self.dynamic_velocity_hurdle*100:.4f}%")

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
        Evaluates real-time market state against structural hurdles and fee gates.
        Returns an execution payload dictionary only if the gross momentum beats the fee barrier.
        """
        # Gate 0: Ensure detected lag is positive and meaningful
        if tau_max <= self.min_lag_seconds:
            return None

        # Gate 1: Check structural correlation confidence
        if rho <= self.min_confidence:
            return None
            
        # Gate 2: Fee-Aware Velocity Hurdle Check
        delta_p_leader = self.compute_leader_velocity(x)
        
        # We strictly check against the dynamic hurdle rate (Round-Trip Fees + Buffer)
        if abs(delta_p_leader) <= self.dynamic_velocity_hurdle:
            return None
            
        # Gate 3: Directional arbitrage determination and reference checks
        lagger_current_price = y[-1]
        if np.isnan(lagger_current_price) or lagger_current_price == 0.0:
            return None

        # Anticipated correction target for the Lagger exchange
        anticipated_target_price = lagger_current_price * (1.0 + delta_p_leader)
        expected_gross_return = abs(delta_p_leader)
        expected_net_return = expected_gross_return - self.round_trip_fee
        
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
            "required_hurdle_pct": round(float(self.dynamic_velocity_hurdle * 100.0), 4),
            "expected_net_alpha_pct": round(float(expected_net_return * 100.0), 4),
            "lagger_reference_price": round(float(lagger_current_price), 2),
            "anticipated_target_price": round(float(anticipated_target_price), 2)
        }
        
        return execution_payload