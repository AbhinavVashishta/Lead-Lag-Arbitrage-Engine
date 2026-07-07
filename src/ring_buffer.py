"""
Circular buffer that holds onto whatever prices we get from the websocket in appropriate time "buckets" 
Circular is good cus then we don't have a ceiling on how much data we can hold, we just keep erasing old data we dont care about anymore (hopefully)
"""
import numpy as np

class RingBuffer:

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.buffer = np.full(self.capacity, np.nan, dtype=np.float64)
        self.index = 0
        self.size = 0

    def append(self, value: float)->None:
        self.buffer[self.index] = value
        self.index = (self.index+1)%(self.capacity)
        if(self.size<self.capacity):
            self.size+=1
    
    def get_window(self) -> np.ndarray:
        if(self.size<self.capacity):
            return self.buffer[:self.size].copy()
        
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))