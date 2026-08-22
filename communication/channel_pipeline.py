import torch
import numpy as np
from communication.serialization import serialize_to_bits, deserialize_from_bits
from communication.turbo import TurboCodec
from communication.bpsk import bpsk_modulate, bpsk_demodulate
from communication.awgn import awgn_channel
from communication.rayleigh import rayleigh_fading_channel

class ChannelPipeline:
    def __init__(self, use_turbo: bool = True, channel_type: str = 'awgn'):
        self.use_turbo = use_turbo
        self.channel_type = channel_type.lower()
        if self.use_turbo:
            self.codec = TurboCodec(iterations=5)
            
    def transmit(self, q: torch.Tensor, snr_db: float, metadata: dict) -> torch.Tensor:
        """
        Simulates the entire communication pipeline:
        Serialization -> Turbo Encode -> BPSK -> Channel -> Demodulate -> Turbo Decode -> Deserialization
        """
        # 1. Serialization
        bits = serialize_to_bits(q)
        
        # 2. Turbo Encoding
        if self.use_turbo:
            encoded_bits = self.codec.encode(bits)
        else:
            encoded_bits = bits
            
        # 3. BPSK Modulation
        symbols = bpsk_modulate(encoded_bits)
        
        # 4. Channel
        if self.channel_type == 'awgn':
            received_symbols, noise_variance = awgn_channel(symbols, snr_db)
            
            # 5. Demodulation
            llrs = bpsk_demodulate(received_symbols, noise_variance)
            
        elif self.channel_type == 'rayleigh':
            received_symbols, h, noise_variance = rayleigh_fading_channel(symbols, snr_db)
            
            # 5. Demodulation with fading compensation (Zero Forcing / Equalization)
            # y = h*x + n => y*conj(h) = |h|^2 * x + n_prime
            # Equalized signal: y_eq = y * conj(h) / |h|^2 = y / h
            # LLR for BPSK over Rayleigh: LLR = - (2 * real(y * conj(h))) / noise_variance 
            # (Assuming 0->-1, 1->1, actually LLR = 4 * real(y * conj(h)) / N0, but with our 0->-1 it's negative)
            # Let's do simple zero-forcing equalization first to get it back to AWGN-like symbols
            # y_eq = received_symbols * np.conj(h) / (np.abs(h)**2)
            # We can use the AWGN LLR formula on the equalized symbols, scaled by SNR
            # A more precise LLR:
            llrs = -2.0 * np.real(received_symbols * np.conj(h)) / noise_variance
            
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")
            
        # 6. Turbo Decoding
        if self.use_turbo:
            decoded_bits = self.codec.decode(llrs)
            # Trim any padding/extra bits if turbo code size differs
            decoded_bits = decoded_bits[:len(bits)]
        else:
            decoded_bits = (llrs < 0).astype(int)
            
        # 7. Deserialization
        q_tilde = deserialize_from_bits(decoded_bits, metadata['shape'])
        
        return q_tilde.to(q.device)
