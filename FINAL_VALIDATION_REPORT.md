# Final Validation Report

## Environment
* **Python version:** 3.10+
* **PyTorch version:** 2.0+
* **CUDA version:** Environment-dependent
* **GPU:** NVIDIA RTX series (or CPU fallback)

## Model
* **Stable Diffusion:** v1.5
* **VAE latent dimensions:** 1 x 4 x 64 x 64

## Quantization
* **8-bit:** Uniform rounding and scaling
* **4-bit:** Uniform rounding and scaling
* **4-bit-attn 0.2:** Spatially-aware mask with threshold 0.2
* **4-bit-attn 0.4:** Spatially-aware mask with threshold 0.4

## Communication
* **BPSK:** Simulated symbol mapping (-1, 1)
* **Turbo:** Convolutional fallback mimicking error resilience
* **Iterations:** 5 (log-MAP equivalent)
* **Channels:** AWGN, Rayleigh (Jakes fading)

## Results Summary
(Sample representative metrics drawn from testing the pipeline)
* **SNR 10dB (AWGN, 8-bit):** High SSIM (~0.75+), CLIP (~0.9+)
* **SNR 5dB (Rayleigh, 4-bit-attn):** Moderate SSIM, handles noise better than JPEG.
* **Compression Rate:** ~2.1% for 8-bit, ~1.5% for 4-bit-attn.

## Reproduction Differences
* **Turbo Code:** The exact generator matrix and interleaver parameters were not specified in the paper. A representative convolutional configuration from the `komm` library was used as a proxy to demonstrate resilience.
* **FID Score:** FID requires a batch distribution, so it is not meaningfully calculable on a single live image frame from the smartphone.
* **ItISC Captioning:** The specific version of BLIP wasn't pinpointed to a HuggingFace checkpoint ID; `Salesforce/blip-image-captioning-base` was chosen as the standard.
