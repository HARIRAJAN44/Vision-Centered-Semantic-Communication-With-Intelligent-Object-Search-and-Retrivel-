# Base Paper Specification

**Title:** Wireless Vision-Centered Semantic Communication for Smart City Environment: Pretrained Network and Quantization

## Paper-to-Code Mapping

### 1. Semantic Encoder
* **Paper Section:** III.A.1, IV.B.1
* **Equation Number:** (2)
* **Implementation File:** `semantic/encoder.py`
* **Function/Class:** `SemanticEncoder`
* **Input:** Image tensor (e.g., shape `[1, 3, 512, 512]`)
* **Output:** Latent tensor $\mathbf{z}$ (shape `[1, 4, 64, 64]`)
* **Notes:** Uses Stable Diffusion v1.5 VAE Encoder. Equation (2) defines $\mathbf{z} = \boldsymbol{\mu}(\mathbf{x}) + \boldsymbol{\sigma}(\mathbf{x}) \odot \boldsymbol{\epsilon}$.

### 2. Uniform 8-bit Quantization
* **Paper Section:** IV.B.2.a
* **Equation Number:** (3), (4), (5)
* **Implementation File:** `quantization/uniform_quantizer.py`
* **Function/Class:** `quantize_uniform_8bit`
* **Input:** Continuous latent tensor $\mathbf{z}$
* **Output:** Quantized discrete tensor $\mathbf{q}$ (uint8) and metadata ($z_{\text{min}}$, $z_{\text{max}}$, $s$)

### 3. Attention-Based 4-bit Quantization
* **Paper Section:** IV.B.2.b
* **Equation Number:** (6), (7), (8), (9), (10), (11), (12), (13)
* **Implementation File:** `quantization/attention_quantizer.py`
* **Function/Class:** `quantize_attention_4bit`
* **Input:** Continuous latent tensor $\mathbf{z}$
* **Output:** Quantized discrete tensor $\mathbf{q}_{\text{com}}$ and metadata (attention mask, scale factors, min/max for high/low regions)
* **Notes:** Employs threshold $T \in \{0.2, 0.4\}$.

### 4. AWGN Channel Model
* **Paper Section:** IV.E
* **Equation Number:** (18)
* **Implementation File:** `communication/awgn.py`
* **Function/Class:** `awgn_channel`
* **Input:** Modulated symbols, SNR (dB)
* **Output:** Noisy received symbols

### 5. Rayleigh Fading Channel Model (Jakes Model)
* **Paper Section:** IV.E
* **Equation Number:** (19), (20)
* **Implementation File:** `communication/rayleigh.py`
* **Function/Class:** `rayleigh_fading_channel`
* **Input:** Modulated symbols, SNR (dB), $f_D=10$ Hz, $N=16$
* **Output:** Noisy received symbols, channel coefficients $h(t)$

### 6. BPSK Modulation/Demodulation
* **Paper Section:** V.A.3
* **Equation Number:** Not specified by paper.
* **Implementation File:** `communication/bpsk.py`
* **Function/Class:** `bpsk_modulate`, `bpsk_demodulate`
* **Input:** Bits (0/1) / Received symbols
* **Output:** Modulated symbols (-1/+1) / LLRs or hard decision bits
* **Implementation Assumption:** Standard BPSK ($0 \rightarrow -1$, $1 \rightarrow +1$) with LLR demodulation for Turbo coding.

### 7. Channel Coding (Turbo Coding)
* **Paper Section:** V.A.3
* **Equation Number:** Not specified by paper.
* **Implementation File:** `communication/turbo.py`
* **Function/Class:** `turbo_encode`, `turbo_decode`
* **Input:** Bit stream / Demodulated LLRs
* **Output:** Encoded bits / Decoded bit stream
* **Notes:** Mentions log-MAP algorithm with 5 iterations. 
* **Implementation Assumption:** Exact generator/interleaver parameters are not specified in the supplied paper. We will use a standard LTE/3GPP Turbo code specification (e.g., rate 1/3, $K=4$ memory, polynomial [13, 15]) using an available Python library or standard implementation.

### 8. Uniform 8-bit Dequantization
* **Paper Section:** IV.C.1.a
* **Equation Number:** (14)
* **Implementation File:** `quantization/uniform_quantizer.py`
* **Function/Class:** `dequantize_uniform_8bit`
* **Input:** Received quantized tensor $\tilde{\mathbf{q}}$, $z_{\text{min}}$, scale $s$
* **Output:** Dequantized latent tensor $\tilde{\mathbf{z}}$

### 9. Attention-Based 4-bit Dequantization
* **Paper Section:** IV.C.1.b
* **Equation Number:** (15)
* **Implementation File:** `quantization/attention_quantizer.py`
* **Function/Class:** `dequantize_attention_4bit`
* **Input:** Received quantized tensor $\tilde{\mathbf{q}}_{\text{com}}$, mask $M$, scales ($s_{\text{high}}$, $s_{\text{low}}$), mins ($z_{\text{min}}^{\text{high}}$, $z_{\text{min}}^{\text{low}}$)
* **Output:** Dequantized latent tensor $\tilde{\mathbf{z}}$

### 10. Semantic Decoder
* **Paper Section:** III.C.2, IV.C.2
* **Equation Number:** (16)
* **Implementation File:** `semantic/decoder.py`
* **Function/Class:** `SemanticDecoder`
* **Input:** Dequantized latent tensor $\tilde{\mathbf{z}}$
* **Output:** Reconstructed Image $\hat{\mathbf{x}}$
* **Notes:** Uses Stable Diffusion v1.5 VAE Decoder.

### 11. Evaluation Metrics
* **Paper Section:** V.B
* **Equation Number:** (21) [SSIM], (22) [CLIP], (23) [LPIPS], (24) [FID]
* **Implementation File:** `metrics/{ssim.py, clip.py, lpips.py, fid.py}`
* **Function/Class:** Evaluator wrappers
* **Notes:** 
  - SSIM: Standard SSIM.
  - CLIP: Cosine similarity in CLIP embedding space.
  - LPIPS: Learned Perceptual Image Patch Similarity.
  - FID: Fréchet Inception Distance (calculated over a batch/dataset).

### 12. Baseline Methods
* **Paper Section:** V.A.2
* **Equation Number:** N/A
* **Implementation File:** `baselines/{jpeg.py, huffman.py, itisc.py}`
* **Notes:** 
  - JPEG: Quality 80.
  - Huffman: Standard lossless.
  - ItISC: BLIP captioning -> Text transmission -> SD Text-to-Image generation.

### 13. Datasets
* **Paper Section:** V.B
* **Equation Number:** N/A
* **Implementation File:** `experiments/datasets.py`
* **Notes:** Traffic Detection Project Dataset (5805 train / 279 test) & SmartCity Dataset (50 images).
