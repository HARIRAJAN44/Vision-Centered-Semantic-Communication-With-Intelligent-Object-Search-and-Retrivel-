import numpy as np

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Modulates bits (0/1) to BPSK symbols (-1/+1).
    0 -> -1
    1 -> +1
    """
    symbols = 2.0 * bits.astype(float) - 1.0
    return symbols

def bpsk_demodulate(received_symbols: np.ndarray, noise_variance: float) -> np.ndarray:
    """
    Demodulates BPSK symbols to Log-Likelihood Ratios (LLRs).
    LLR = log(P(b=0|y) / P(b=1|y))
    For AWGN, LLR = - (2 * y) / noise_variance
    Actually, if 0 -> -1 and 1 -> +1:
    LLR = log(P(b=0|y) / P(b=1|y)) = ( - (y - (-1))^2 + (y - 1)^2 ) / (2 * noise_var)
    LLR = ( - (y^2 + 2y + 1) + (y^2 - 2y + 1) ) / (2 * noise_var)
    LLR = -4y / (2 * noise_var) = -2y / noise_var
    
    If LLR < 0, probability of bit 1 is higher.
    """
    llrs = -2.0 * received_symbols / noise_variance
    return llrs
