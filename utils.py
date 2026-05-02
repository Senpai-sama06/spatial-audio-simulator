import numpy as np
import scipy.signal
from state import MasterState

def apply_mic_jitter(locations: list, variance_m: float = 0.002) -> list:
    """
    Applies a randomized Gaussian variance (jitter) to the theoretical x,y,z coordinates
    of the microphones to prevent the ML model from memorizing array geometry.
    """
    MasterState.check_seed()
    locs_array = np.array(locations, dtype=np.float64)
    jitter = np.random.normal(locs_array, variance_m)
    # The requirement says variance, which technically is standard deviation in np.random.normal if passed as scale. 
    # "Gaussian variance (e.g., +/- 2mm)" -> We use scale = 0.002
    jitter = np.random.normal(0, variance_m, locs_array.shape)
    jittered_locs = locs_array + jitter
    return jittered_locs.tolist()

def fractional_delay_sinc(signal: np.ndarray, delay_samples: float, num_taps: int = 31) -> np.ndarray:
    """
    Implements fractional delay using Windowed Sinc interpolation.
    Linear interpolation is banned due to zippering artifacts.
    """
    if delay_samples == 0:
        return signal.copy()

    # Create the windowed sinc filter
    int_delay = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay
    
    n = np.arange(-num_taps // 2, num_taps // 2 + 1)
    sinc_filter = np.sinc(n - frac_delay)
    
    # Apply Blackman window
    window = np.blackman(len(sinc_filter))
    sinc_filter *= window
    
    # Normalize filter
    sinc_filter /= np.sum(sinc_filter)
    
    # Convolve signal with the fractional delay filter
    delayed_frac = scipy.signal.convolve(signal, sinc_filter, mode='same')
    
    # Apply integer delay
    delayed_sig = np.zeros_like(signal, dtype=np.float64)
    if int_delay > 0:
        if int_delay < len(signal):
            delayed_sig[int_delay:] = delayed_frac[:-int_delay]
    elif int_delay < 0:
        if -int_delay < len(signal):
            delayed_sig[:int_delay] = delayed_frac[-int_delay:]
    else:
        delayed_sig = delayed_frac

    return delayed_sig
