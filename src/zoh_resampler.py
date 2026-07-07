"""
FFT works on discrete time jumps, but markets aren't that kind, this zero order resampler
adjusts our timeline to be more kin
"""

from typing import List, Tuple, Dict
import numpy as np
from src.ring_buffer import RingBuffer

class ZOHResampler:
    def __init__(self, dt_ms: float = 5.0, buffer_capacity: int = 4096):
        self.dt = dt_ms/1000.0 #converting to seconds to match timestaps
        self.leader_buffer = RingBuffer(capacity=buffer_capacity)
        self.lagger_buffer = RingBuffer(capacity=buffer_capacity)

        #tracking the state for the carry forward logic ;)
        self.last_leader_price = np.nan
        self.last_lagger_price = np.nan
    
    def resample_stream(self, leader_ticks: List[Dict[str, float]], lagger_ticks: List[Dict[str, float]],
                        start_time: float, end_time:float)-> Tuple[np.ndarray, np.ndarray]:
        time_grid = np.arange(start_time + self.dt, end_time + self.dt, self.dt)

        #pointers to store our position
        left_index, right_index = 0, 0
        num_leader_ticks, num_lagger_ticks = len(leader_ticks), len(lagger_ticks)

        for t_end in time_grid:
            #The current window is [t_end - dt, t_end], we're processing our stuff in this window

            while left_index<num_leader_ticks and leader_ticks[left_index]['timestamp']<=t_end:
                self.last_leader_price = leader_ticks[left_index]['price']
                left_index+=1
            
            #if no trades keep the old value (we have to make sure something was there before to begin with tho)
            if not np.isnan(self.last_leader_price):
                self.leader_buffer.append(self.last_leader_price)
            
            #same thing for lagging
            while right_index<num_lagger_ticks and lagger_ticks[right_index]['timestamp']<=t_end:
                self.last_lagger_price = lagger_ticks[right_index]['price']
                right_index+=1
            
            if not np.isnan(self.last_lagger_price):
                self.lagger_buffer.append(self.last_lagger_price)
                
        return self.leader_buffer.get_window(), self.lagger_buffer.get_window()