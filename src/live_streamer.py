import asyncio
import json
import time
from typing import List, Dict
import websockets
from src.zoh_resampler import ZOHResampler

class LiveExchangeStreamer:
    #Connects to the live data stream and feeds it into the resampler
    def __init__(self, dt_ms: float = 5.0, buffer_capacity: int = 4096):
        self.resampler = ZOHResampler(dt_ms=dt_ms, buffer_capacity=buffer_capacity)
        self.leader_ticks: List[Dict[str, float]] = []
        self.lagger_ticks: List[Dict[str, float]] = []
        self.is_running = False

    async def consume_binance(self, symbol: str = "btcusdt"):
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        async with websockets.connect(uri) as ws:
            print(f"[Leader] Connected to Binance ({symbol.upper()})")
            while self.is_running:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    tick = {
                        'timestamp': float(data['T']) / 1000.0,
                        'price': float(data['p'])
                    }
                    self.leader_ticks.append(tick)
                except Exception as e:
                    print(f"[Leader Error] {e}")
                    await asyncio.sleep(1)

    async def consume_coinbase(self, product_id: str = "BTC-USD"):
        uri = "wss://ws-feed.exchange.coinbase.com"
        async with websockets.connect(uri) as ws:
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channels": ["matches"]
            }
            await ws.send(json.dumps(subscribe_msg))
            print(f"[Lagger] Connected to Coinbase ({product_id})")
            
            while self.is_running:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    if data.get('type') == 'match':
                        # Convert Coinbase ISO timestamp or system time to epoch seconds
                        tick = {
                            'timestamp': time.time(), 
                            'price': float(data['price'])
                        }
                        self.lagger_ticks.append(tick)
                except Exception as e:
                    print(f"[Lagger Error] {e}")
                    await asyncio.sleep(1)

    async def run_resampler_clock(self, batch_interval_sec: float = 1.0):
        #flushes ingested ticks through the ZOH Resampler,
        #outputs the synchronized vectors x[n] and y[n] .
        print("[Clock] Waiting 3 seconds for initial exchange liquidity...")
        await asyncio.sleep(3.0)
        
        last_time = time.time() - batch_interval_sec
        
        while self.is_running:
            await asyncio.sleep(batch_interval_sec)
            current_time = time.time()
            
            l_batch = [t for t in self.leader_ticks if last_time <= t['timestamp'] <= current_time]
            r_batch = [t for t in self.lagger_ticks if last_time <= t['timestamp'] <= current_time]
            
            # Prune processed ticks from memory to prevent RAM bloat
            self.leader_ticks = [t for t in self.leader_ticks if t['timestamp'] > current_time]
            self.lagger_ticks = [t for t in self.lagger_ticks if t['timestamp'] > current_time]
            
            x, y = self.resampler.resample_stream(l_batch, r_batch, last_time, current_time)
            
            print(f"\n--- [Live Window: {batch_interval_sec}s | Δt = {self.resampler.dt*1000}ms] ---")
            print(f"Leader (Binance)  Ticks Ingested: {len(l_batch)} | Array Size: {len(x)} | Latest: ${x[-1]:.2f}" if len(x) > 0 else "Waiting for initial valid price...")
            print(f"Lagger (Coinbase) Ticks Ingested: {len(r_batch)} | Array Size: {len(y)} | Latest: ${y[-1]:.2f}" if len(y) > 0 else "Waiting for initial valid price...")
            
            last_time = current_time

    async def start(self):
        #Launches WebSocket listeners and the ZOH clock concurrently.
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