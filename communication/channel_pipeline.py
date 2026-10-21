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
        
        Handles both standard uint8 tensors and pre-packed multi-bit quantized data.
        """
        # Check if data is already packed (multi-bit quantization modes)
        is_packed = metadata.get('mode', '').endswith('bit-uniform') or 'region_metadata' in metadata
        
        if is_packed:
            # For pre-packed data, q is already a 1D byte array
            # Convert to bits directly from packed bytes
            q_np = q.cpu().numpy() if isinstance(q, torch.Tensor) else np.array(q)
            bits = np.unpackbits(q_np.view(np.uint8))
        else:
            # 1. Standard serialization for 8bit, 4bit-attn
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
        if is_packed:
            # For pre-packed data, pack bits back to bytes
            bytes_arr = np.packbits(decoded_bits)
            q_tilde = torch.from_numpy(bytes_arr)
        else:
            q_tilde = deserialize_from_bits(decoded_bits, metadata['shape'])
        
        return q_tilde.to(q.device)
