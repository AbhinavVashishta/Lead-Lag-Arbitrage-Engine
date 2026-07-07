import numpy as np
from typing import Tuple

class WienerKhinchinDSP:
    #goated engine that computes the real time cross correlation between the leader and lagger
    #arrays to extract the temporal lag and signal lag in O(nlogn) time
    def __init__(self, dt_ms: float = 5.0):
        self.dt = dt_ms/(1000.0)
        self.max_lag_buckets = int(500.0 / dt_ms)

    def compute_cross_correlation(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        x is leader, y is lagger
        tau_max = detected lag in seconds
        rho = confidence
        r_xy = The full time-domain cross correlation array
        """
        n = len(x)

        if n!=len(y) or n==0:
            raise ValueError("Input arrays are off")
        
        #remove dc bias (Analog electronics truly coming in clutch) & remove macro drifiting of the prices
        x_vel = np.diff(x)
        y_vel = np.diff(y)

        x_clean = x_vel - np.mean(x_vel)
        y_clean = y_vel - np.mean(y_vel)

        #check for dead market
        var_x = np.sum(x_clean**2)
        var_y = np.sum(y_clean**2)
        if(var_x == 0.0 or var_y == 0.0):
            return 0.0, 0.0, np.zeroes(n)
        
        #Since we have circle around logic, we want to avoid abrupt jumps creating false frequencies, so we use a hann window
        #removing hann window to check something
        n = len(x_clean)
        #hann_window = np.hanning(n)
        x_win = x_clean
        y_win = y_clean

        pad_len = 2*n-1

        #fast fourier transform with the padded 2n-1 length
        X_f = np.fft.fft(x_win, n = pad_len)
        Y_f = np.fft.fft(y_win, n=pad_len)

        #Cross Power Spectral Density (its somehow cooler than it sounds)
        S_xy = np.conj(X_f) * Y_f

        r_xy_padded = np.fft.ifft(S_xy).real #freq to time to domain

        r_xy_shifted = np.fft.fftshift(r_xy_padded)

        max_idx = np.argmax(r_xy_shifted)

        center_idx = pad_len//2

        search_start = max(0, center_idx - self.max_lag_buckets)
        search_end = min(pad_len, center_idx + self.max_lag_buckets + 1)

        valid_correlation_window = r_xy_shifted[search_start:search_end]
        local_max_idx = np.argmax(valid_correlation_window)

        lag_idx = (search_start + local_max_idx) - center_idx

        tau_max = lag_idx*self.dt

        auto_x = np.sum(x_win ** 2)
        auto_y = np.sum(y_win ** 2)
        normalization_denom = np.sqrt(auto_x * auto_y)

        if normalization_denom == 0.0:
            rho = 0.0
        else:
            rho = float(r_xy_shifted[search_start + local_max_idx] / normalization_denom)
            # Bound confidence strictly within [0.0, 1.0]
            rho = min(max(rho, 0.0), 1.0)

        return tau_max, rho, r_xy_shifted
