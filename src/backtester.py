import bisect
from typing import List, Dict, Any, Optional
import numpy as np

from src.zoh_resampler import ZOHResampler
from src.dsp_engine import WienerKhinchinDSP
from src.trigger_layer import SignalTriggerLayer


class LatencyAwareBacktester:
    def __init__(
        self,
        dt_ms: float = 5.0,
        buffer_capacity: int = 4096,
        min_confidence: float = 0.80,
        taker_fee_rate: float = 0.0005,
        alpha_buffer: float = 0.0005,
        min_lag_seconds: float = 0.0,
        latency_ms: float = 30.0,
        batch_interval_sec: float = 1.0
    ):
        self.resampler = ZOHResampler(dt_ms=dt_ms, buffer_capacity=buffer_capacity)
        self.dsp_engine = WienerKhinchinDSP(dt_ms=dt_ms)
        self.trigger_layer = SignalTriggerLayer(
            min_confidence=min_confidence,
            taker_fee_rate=taker_fee_rate,
            alpha_buffer=alpha_buffer,
            min_lag_seconds=min_lag_seconds
        )

        self.latency_sec = latency_ms / 1000.0
        self.round_trip_fee = 2.0 * taker_fee_rate
        self.batch_interval_sec = batch_interval_sec

        self.signals: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    @staticmethod
    def _price_at(ticks: List[Dict[str, float]], timestamps: List[float], target_time: float) -> Optional[float]:
        # last trade at or before target_time - ZOH logic, same as the resampler uses live
        idx = bisect.bisect_right(timestamps, target_time) - 1
        if idx < 0:
            return None
        return ticks[idx]['price']

    def run(self, leader_ticks: List[Dict[str, float]], lagger_ticks: List[Dict[str, float]]) -> Dict[str, Any]:
        leader_ticks = sorted(leader_ticks, key=lambda t: t['timestamp'])
        lagger_ticks = sorted(lagger_ticks, key=lambda t: t['timestamp'])

        if not leader_ticks or not lagger_ticks:
            raise ValueError("Need tick data on both legs to backtest")

        lagger_timestamps = [t['timestamp'] for t in lagger_ticks]

        start = max(leader_ticks[0]['timestamp'], lagger_ticks[0]['timestamp'])
        end = min(leader_ticks[-1]['timestamp'], lagger_ticks[-1]['timestamp'])

        window_start = start
        while window_start + self.batch_interval_sec <= end:
            window_end = window_start + self.batch_interval_sec

            l_batch = [t for t in leader_ticks if window_start <= t['timestamp'] <= window_end]
            r_batch = [t for t in lagger_ticks if window_start <= t['timestamp'] <= window_end]

            x, y = self.resampler.resample_stream(l_batch, r_batch, window_start, window_end)

            if len(x) > 100 and len(x) == len(y):
                tau_max, rho, _ = self.dsp_engine.compute_cross_correlation(x, y)
                signal = self.trigger_layer.evaluate_signal(x, y, tau_max, rho)

                self.signals.append({
                    'tau_max': tau_max,
                    'rho': rho,
                    'signal_time': window_end,
                    'fired': signal is not None
                })

                if signal is not None:
                    trade = self._simulate_fill(signal, window_end, tau_max, lagger_ticks, lagger_timestamps)
                    if trade is not None:
                        self.trades.append(trade)

            window_start = window_end

        return self._summarize()

    def _simulate_fill(
        self,
        signal: Dict[str, Any],
        signal_time: float,
        tau_max: float,
        lagger_ticks: List[Dict[str, float]],
        lagger_timestamps: List[float]
    ) -> Optional[Dict[str, Any]]:
        # we don't get filled the instant the signal fires - the order has to cross the wire
        entry_time = signal_time + self.latency_sec
        # exit at the point the correction was supposed to have happened by (tau_max after signal),
        # itself delayed by the same round-trip latency on the way out
        exit_time = entry_time + max(tau_max, self.latency_sec)

        entry_price = self._price_at(lagger_ticks, lagger_timestamps, entry_time)
        exit_price = self._price_at(lagger_ticks, lagger_timestamps, exit_time)

        # arb window closed before we could get filled, or we've run off the end of the data
        if entry_price is None or exit_price is None or entry_price == 0.0:
            return None

        direction = 1.0 if signal['action'] == 'BUY' else -1.0
        gross_return = direction * (exit_price - entry_price) / entry_price
        net_return = gross_return - self.round_trip_fee

        return {
            'signal_id': signal['signal_id'],
            'action': signal['action'],
            'signal_time': signal_time,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_return_pct': round(gross_return * 100.0, 4),
            'net_return_pct': round(net_return * 100.0, 4),
            'tau_max_ms': signal['detected_lag_ms'],
            'confidence_rho': signal['confidence_rho']
        }

    def _summarize(self) -> Dict[str, Any]:
        returns = np.array([t['net_return_pct'] / 100.0 for t in self.trades])
        n_trades = len(returns)
        n_fired = sum(1 for s in self.signals if s['fired'])

        if n_trades == 0:
            return {
                'n_windows_evaluated': len(self.signals),
                'n_signals_fired': n_fired,
                'n_trades_filled': 0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'total_return_pct': 0.0,
                'win_rate_pct': 0.0,
                'trades': [],
                'tau_max_distribution_ms': [s['tau_max'] * 1000.0 for s in self.signals if s['fired']]
            }

        equity_curve = np.cumsum(returns)
        running_peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve - running_peak
        max_drawdown = float(drawdown.min())

        sharpe = 0.0
        if returns.std() > 0:
            # per-trade Sharpe scaled by sqrt(N) - not a calendar annualization, this is an
            # HFT strategy so "per year" is meaningless without knowing the true signal rate
            sharpe = float(returns.mean() / returns.std() * np.sqrt(n_trades))

        return {
            'n_windows_evaluated': len(self.signals),
            'n_signals_fired': n_fired,
            'n_trades_filled': n_trades,
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown_pct': round(max_drawdown * 100.0, 4),
            'total_return_pct': round(float(equity_curve[-1]) * 100.0, 4),
            'win_rate_pct': round(float(np.mean(returns > 0) * 100.0), 2),
            'trades': self.trades,
            'tau_max_distribution_ms': [s['tau_max'] * 1000.0 for s in self.signals if s['fired']]
        }

    def plot_tau_histogram(self, save_path: str = "tau_max_distribution.png") -> Optional[str]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fired_taus = [s['tau_max'] * 1000.0 for s in self.signals if s['fired']]
        if not fired_taus:
            print("[Backtester] No signals fired, nothing to plot.")
            return None

        plt.figure(figsize=(8, 5))
        plt.hist(fired_taus, bins=30, edgecolor='black')
        plt.xlabel("Detected Lag tau_max (ms)")
        plt.ylabel("Frequency")
        plt.title("Distribution of tau_max at Signal Fire")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 3:
        print("Usage: python backtester.py <leader_ticks.json> <lagger_ticks.json>")
        print("Each file: JSON list of {'timestamp': epoch_seconds, 'price': float}")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        leader_ticks = json.load(f)
    with open(sys.argv[2]) as f:
        lagger_ticks = json.load(f)

    bt = LatencyAwareBacktester(latency_ms=30.0)
    results = bt.run(leader_ticks, lagger_ticks)

    print(f"\nWindows evaluated: {results['n_windows_evaluated']}")
    print(f"Signals fired:     {results['n_signals_fired']}")
    print(f"Trades filled:     {results['n_trades_filled']}")
    print(f"Sharpe ratio:      {results['sharpe_ratio']}")
    print(f"Max drawdown:      {results['max_drawdown_pct']}%")
    print(f"Total return:      {results['total_return_pct']}%")
    print(f"Win rate:          {results['win_rate_pct']}%")

    bt.plot_tau_histogram("tau_max_distribution.png")