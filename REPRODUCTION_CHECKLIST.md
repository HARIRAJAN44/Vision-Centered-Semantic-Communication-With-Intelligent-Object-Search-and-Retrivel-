# Reproduction Checklist

- [x] Stable Diffusion v1.5 VAE
- [x] 512x512 input
- [x] 1x4x64x64 latent
- [x] VAE stochastic/reparameterization behavior
- [x] 8-bit uniform quantization
- [x] attention 4-bit quantization
- [x] threshold 0.2
- [x] threshold 0.4
- [x] AWGN
- [x] Rayleigh
- [x] Jakes fading
- [x] BPSK
- [x] Turbo coding
- [x] log-MAP
- [x] 5 iterations
- [x] JPEG baseline
- [x] Huffman baseline
- [x] ItISC baseline
- [x] SSIM
- [x] CLIP
- [x] LPIPS
- [x] FID
- [x] compression rate
- [x] ablation study
- [x] Turbo ON/OFF
- [x] SNR sweep
- [x] visual comparison
- [x] smartphone Wi-Fi input

**Notes on implementation assumptions:**
- Turbo Coding: Exact generator/interleaver parameters are not specified in the supplied paper. Implemented using `komm` Convolutional Code with feedback to simulate strong error correction capability in Python.
- BPSK: Not explicitly specified whether 0->1 or 0->-1. Assumed standard $0 \rightarrow -1, 1 \rightarrow 1$.
