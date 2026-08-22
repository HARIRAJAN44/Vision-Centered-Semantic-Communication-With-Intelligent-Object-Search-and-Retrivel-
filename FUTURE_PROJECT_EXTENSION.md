# Future Project Extension Concept

**From:** *Wireless VCSC for Smart City*  
**To:** *Vision-Centered Semantic Communication for Wireless Surveillance*

## Conceptual Evolution

The current base paper focuses on general "Smart City Environment" scenarios, transmitting images over generic AWGN and Rayleigh channels. The final project will pivot this architecture specifically towards **Wireless Surveillance**.

### Key Extension Points

1. **Live Surveillance Camera Integration:**
   - Transition from static datasets to processing a live stream from a wireless surveillance camera (e.g., smartphone or IP camera).
   - Implement frame sampling to adapt to the semantic encoder's processing speed (processing keyframes rather than every frame).

2. **Adaptive Quantization based on Region of Interest (ROI):**
   - The base paper's attention module relies on the latent activations. We can extend this to use a lightweight object detector (e.g., YOLO) *before* encoding to explicitly force the highest quantization precision on faces, vehicles, or suspicious activities, which are critical for surveillance.

3. **Real-World Network Measurements:**
   - Instead of purely simulated AWGN/Rayleigh channels, the extension could log actual packet loss, latency, and throughput of the Wi-Fi or 5G connection and dynamically adjust the quantization mode (8-bit vs 4-bit) in real-time.

4. **Privacy-Preserving Semantics:**
   - Surveillance introduces privacy concerns. The semantic encoder could be fine-tuned or modified to deliberately discard identifying features (like faces) while retaining behavioral semantics, transmitting a privacy-safe latent code.

*Note: These extensions are strictly conceptual for the final project and are not implemented in the current faithful base-paper reproduction.*
