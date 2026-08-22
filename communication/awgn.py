import numpy as np

def awgn_channel(symbols: np.ndarray, snr_db: float) -> tuple:
    """
    Simulates an Additive White Gaussian Noise (AWGN) channel.
    
    Args:
    - symbols: Transmitted symbols (e.g., BPSK modulated)
    - snr_db: Signal-to-Noise Ratio in dB
    
    Returns:
    - tuple: (received_symbols, noise_variance)
    """
    # Assuming signal power is 1 (for BPSK where symbols are -1, 1)
    signal_power = 1.0
    
    # Calculate noise variance based on SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_variance), size=symbols.shape)
    
    # Add noise to signal
    received_symbols = symbols + noise
    
    return received_symbols, noise_variance
