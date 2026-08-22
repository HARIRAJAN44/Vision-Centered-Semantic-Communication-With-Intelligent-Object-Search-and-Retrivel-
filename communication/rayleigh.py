import numpy as np

def jakes_fading_model(num_symbols: int, f_d: float = 10.0, num_scatterers: int = 16, t_s: float = 1e-3) -> np.ndarray:
    """
    Generates time-varying Rayleigh fading coefficients using the Jakes model.
    As per Eq (20): h(t) = (1/sqrt(N)) * sum_{n=1}^N exp(j * (2 * pi * f_D * t * cos(theta_n) + phi_n))
    """
    # Time vector
    t = np.arange(num_symbols) * t_s
    
    # Initialize fading coefficients
    h = np.zeros(num_symbols, dtype=np.complex128)
    
    # Generate scatterer parameters (fixed for the duration of the packet/frame for simplicity, or random)
    theta_n = np.random.uniform(0, 2*np.pi, num_scatterers)
    phi_n = np.random.uniform(0, 2*np.pi, num_scatterers)
    
    # Sum over all scatterers
    for n in range(num_scatterers):
        phase = 2 * np.pi * f_d * t * np.cos(theta_n[n]) + phi_n[n]
        h += np.exp(1j * phase)
        
    h = h / np.sqrt(num_scatterers)
    
    return h

def rayleigh_fading_channel(symbols: np.ndarray, snr_db: float, f_d: float = 10.0, num_scatterers: int = 16) -> tuple:
    """
    Simulates a time-varying Rayleigh fading channel with additive Gaussian noise.
    As per Eq (19): q_tilde(t) = h(t) * q(t) + eta(t)
    """
    num_symbols = len(symbols)
    
    # Generate fading coefficients
    # t_s is symbol duration. We can assume an arbitrary reasonable value like 1ms.
    h = jakes_fading_model(num_symbols, f_d=f_d, num_scatterers=num_scatterers, t_s=1e-3)
    
    # Apply fading to symbols
    # Since symbols are real (BPSK), the received signal becomes complex
    faded_symbols = h * symbols
    
    # Calculate noise variance based on SNR
    # The average power of Rayleigh fading |h|^2 is 1. Signal power is 1.
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    
    # Generate complex Gaussian noise
    noise_real = np.random.normal(0, np.sqrt(noise_variance / 2.0), num_symbols)
    noise_imag = np.random.normal(0, np.sqrt(noise_variance / 2.0), num_symbols)
    noise = noise_real + 1j * noise_imag
    
    # Add noise
    received_symbols = faded_symbols + noise
    
    return received_symbols, h, noise_variance
