import asyncio
import json
import time
from datetime import datetime, timezone
from typing import List, Dict
import websockets
import numpy as np
from src.dsp_engine import WienerKhinchinDSP
from src.zoh_resampler import ZOHResampler
from src.trigger_layer import SignalTriggerLayer

class LiveExchangeStreamer:
    """
    Connects to live data streams, resamples ticks into uniform grids,
    computes Wiener-Khinchin cross-correlation, and evaluates execution triggers.
    """
    def __init__(self, dt_ms: float = 5.0, buffer_capacity: int = 4096):
        self.resampler = ZOHResampler(dt_ms=dt_ms, buffer_capacity=buffer_capacity)
        self.dsp_engine = WienerKhinchinDSP(dt_ms=dt_ms)
        
        self.trigger_layer = SignalTriggerLayer(
            min_confidence=0.80, 
            taker_fee_rate=0.0005, 
            alpha_buffer=0.0005,
            min_lag_seconds=0.0
        )
        
        self.leader_ticks: List[Dict[str, float]] = []
        self.lagger_ticks: List[Dict[str, float]] = []
        self.is_running = False

    async def consume_binance(self, symbol: str = "btcusdt"):
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        retry_delay = 1
        while self.is_running:
            try:
                async with websockets.connect(uri) as ws:
                    print(f"[Leader] Connected to Binance ({symbol.upper()})")
                    retry_delay = 1
                    while self.is_running:
                        message = await ws.recv()
                        data = json.loads(message)
                        tick = {
                            'timestamp': data['T'] / 1000.0,
                            'price': float(data['p'])
                        }
                        self.leader_ticks.append(tick)
            except Exception as e:
                print(f"[Leader Error] {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    async def consume_coinbase(self, product_id: str = "BTC-USD"):
        uri = "wss://ws-feed.exchange.coinbase.com"
        retry_delay = 1
        while self.is_running:
            try:
                async with websockets.connect(uri) as ws:
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": [product_id],
                        "channels": ["matches"]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    print(f"[Lagger] Connected to Coinbase ({product_id})")
                    retry_delay = 1

                    while self.is_running:
                        message = await ws.recv()
                        data = json.loads(message)
                        if data.get('type') == 'match':
                            trade_time = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                            tick = {
                                'timestamp': trade_time.timestamp(),
                                'price': float(data['price'])
                            }
                            self.lagger_ticks.append(tick)
            except Exception as e:
                print(f"[Lagger Error] {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    async def run_resampler_clock(self, batch_interval_sec: float = 1.0):
        print("[Clock] Waiting 3 seconds for initial exchange liquidity...")
        await asyncio.sleep(3.0)
        
        last_time = time.time() - batch_interval_sec
        
        while self.is_running:
            await asyncio.sleep(batch_interval_sec)
            current_time = time.time()
            
            l_batch = [t for t in self.leader_ticks if last_time <= t['timestamp'] <= current_time]
            r_batch = [t for t in self.lagger_ticks if last_time <= t['timestamp'] <= current_time]
            
            self.leader_ticks = [t for t in self.leader_ticks if t['timestamp'] > current_time]
            self.lagger_ticks = [t for t in self.lagger_ticks if t['timestamp'] > current_time]
            
            x, y = self.resampler.resample_stream(l_batch, r_batch, last_time, current_time)
            
            print(f"\n[Live Window: {batch_interval_sec}s | Δt = {self.resampler.dt*1000}ms]")
            print(f"Leader (Binance)  Ticks Ingested: {len(l_batch)} | Array Size: {len(x)} | Latest: ${x[-1]:.2f}" if len(x) > 0 else "Waiting for initial valid price...")
            print(f"Lagger (Coinbase) Ticks Ingested: {len(r_batch)} | Array Size: {len(y)} | Latest: ${y[-1]:.2f}" if len(y) > 0 else "Waiting for initial valid price...")
            
            if len(x) > 100 and len(x) == len(y):
                tau_max, rho, _ = self.dsp_engine.compute_cross_correlation(x, y)
                print(f"Detected Lag: {tau_max*1000:.2f} ms | Confidence: {rho:.4f}")
                
                # Component 3: Evaluate execution conditions
                signal_payload = self.trigger_layer.evaluate_signal(x, y, tau_max, rho)
                if signal_payload:
                    print("\n" + "="*70)
                    print(f"🚨 [EXECUTION SIGNAL FIRED] - ID #{signal_payload['signal_id']}")
                    print(f"Action:      {signal_payload['action']} {signal_payload['symbol']} @ {signal_payload['target_exchange']}")
                    print(f"Lag Locked:  {signal_payload['detected_lag_ms']} ms | Confidence: {signal_payload['confidence_rho']}")
                    print(f"Hurdle Gate: {signal_payload['required_hurdle_pct']}% | Velocity: {signal_payload['leader_velocity_pct']}%")
                    print(f"Target Px:   ${signal_payload['anticipated_target_price']} | Ref: ${signal_payload['lagger_reference_price']}")
                    print("="*70 + "\n")
                    
            last_time = current_time

    async def start(self):
        self.is_running = True
        await asyncio.gather(
            self.consume_binance("btcusdt"),
            self.consume_coinbase("BTC-USD"),
            self.run_resampler_clock(batch_interval_sec=1.0)
        )

if __name__ == "__main__":
    streamer = LiveExchangeStreamer(dt_ms=5.0, buffer_capacity=4096)
    try:
        asyncio.run(streamer.start())
    except KeyboardInterrupt:
        print("\n[Shutdown] Disconnecting from live exchanges cleanly.")