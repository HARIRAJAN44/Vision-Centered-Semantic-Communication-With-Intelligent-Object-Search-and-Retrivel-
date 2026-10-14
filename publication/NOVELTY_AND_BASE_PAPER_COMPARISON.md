# Novelty and Base Paper Comparison

**Base Paper:**  
Y. Gong, Z. Chu, Z. Zhu, P. Xiao, M. Zeng, Y. Wang, H. Pandey, and J. Hou, "Wireless Vision-Centered Semantic Communication for Smart City Environment: Pretrained Network and Quantization," *IEEE Transactions on Consumer Electronics*, vol. 72, no. 1, pp. 1–12, Feb. 2026.

---

## 1. Five-Pillar Architectural Comparison

| Dimension | Base Paper (Gong et al., 2026) | Our Proposed & Implemented System | Technical Distinction & Impact |
|---|---|---|---|
| **1. Semantic Latent Compression** | Pretrained Stable Diffusion v1.5 VAE encoder and decoder ($512\times 512 \to 1\times 4\times 64\times 64$). | **Retained identical VAE backbone** for reproducible latent representation fidelity. | Preserves base paper's foundation while ensuring backward compatibility with original VCSC pipelines. |
| **2. Latent Quantization Schemes** | Uniform 8-bit quantization and fixed two-tier attention-based 4-bit quantization ($T \in \{0.2, 0.4\}$). | **Adaptive multi-level quantization ($\mathcal{B} \in \{2, 4, 6, 8\}$ bits)** with explicit bit-packing and dynamic spatial map serialization. | Extends quantization flexibility beyond binary (8-bit vs 4-bit) partitioning to fine-grained 4-level quantization (4, 16, 64, 256 levels). |
| **3. Channel Awareness & Adaptation** | Static bit allocation independent of instantaneous SNR or channel state. | **Channel-aware allocation controller** with 5 condition tiers (Excellent, Good, Fair, Poor, Severe) and closed-form theoretical BER estimates. | Directly modulates base bit-depth and importance boost based on channel state and bandwidth budgets. |
| **4. Object Detection & ROI Semantic Guidance** | None. Attention is derived purely from latent magnitude activations: $A(i,j) = \frac{1}{C}\sum_c \|z(c,i,j)\|$. | **Integrated YOLOv8-nano detector** that extracts bounding boxes, classes, confidences, and maps them to latent space with $8\times$ factor. | Replaces unguided magnitude heuristics with semantically grounded object saliency for critical surveillance classes. |
| **5. Downstream Intelligent Content Retrieval** | None. System terminates at pixel-level image reconstruction at receiver. | **Persistent FAISS vector store (`IndexFlatIP`)** coupled with multimodal CLIP (ViT-B/32) embeddings and structured metadata. | Enables zero-shot natural language search, image-to-image similarity, crop search, and degradation tracking across noisy channels. |
| **6. Downstream Task Adaptation** | Task-agnostic reconstruction. Evaluated purely on SSIM, LPIPS, CLIP, and FID. | **Task-aware allocation framework** with configurable task profiles (Object Identification, Retrieval, Scene Understanding, Custom). | Optimizes bit allocation to maximize downstream task performance rather than uniform perceptual loss. |
| **7. Real-Time Physical Testbed** | Simulated offline dataset evaluation over SmartCity and Traffic Detection datasets. | **Interactive Flask-SocketIO live demonstration platform** streaming real camera frames from smartphone to receiver dashboard. | Provides a tangible, interactive edge-to-server demonstration of semantic communication resilience under noise. |

---

## 2. Granular Novelty Breakdown

### A. Original Architecture Retained
1. **VAE Autoencoder:** `runwayml/stable-diffusion-v1-5` VAE encoder $\mathcal{E}$ and decoder $\mathcal{D}$.
2. **Latent Dimension:** $\mathbf{z} \in \mathbb{R}^{1 \times 4 \times 64 \times 64}$ for $512 \times 512 \times 3$ input frames.
3. **Uniform 8-bit Quantization Baseline:** Equation (3)–(5) and (14) preserved in `quantization/uniform_quantizer.py`.
4. **Attention-based 4-bit Quantization Baseline:** Equation (6)–(13) and (15) preserved in `quantization/attention_quantizer.py` for thresholds $T=0.2$ and $T=0.4$.
5. **Channel Models:** AWGN and Rayleigh fading with Jakes Doppler model ($N=16$ scatterers, $f_D = 10\text{ Hz}$).
6. **Modulation:** BPSK ($0 \to -1, 1 \to +1$).
7. **Baselines:** JPEG (quality 80), Huffman/zlib, and ItISC (BLIP captioning + SD text-to-image).

### B. Implemented Technical Modifications
1. **Multi-Bit Quantization Engine (`adaptive_quantizer.py`):**
   - Full support for 2-bit (4 levels), 4-bit (16 levels), 6-bit (64 levels), and 8-bit (256 levels).
   - Non-byte-aligned bit packing (`_pack_bits` and `_unpack_bits`) allowing true bit-level transmission payloads.
   - Dynamic allocation map serialization that encodes unique active bit-depths and packed indices with negligible overhead ($< 1\%$ of payload).
2. **Channel-Aware Decision Controller (`allocation_controller.py`):**
   - Five channel regimes: Excellent ($>20\text{ dB}$), Good ($10-20\text{ dB}$), Fair ($5-10\text{ dB}$), Poor ($0-5\text{ dB}$), Severe ($<0\text{ dB}$).
   - Closed-form BER estimation:
     $$P_b^{\text{AWGN}} = \frac{1}{2} \text{erfc}\left(\sqrt{10^{\text{SNR}_{\text{dB}}/10}}\right), \quad P_b^{\text{Rayleigh}} = \frac{1}{2} \left(1 - \sqrt{\frac{10^{\text{SNR}_{\text{dB}}/10}}{1 + 10^{\text{SNR}_{\text{dB}}/10}}}\right)$$
   - Greedy budget-enforcing bit reallocation algorithm that demotes non-salient regions when bandwidth limits are exceeded.
3. **Turbo Code Proxy Realization (`communication/turbo.py`):**
   - Replaced theoretical, unparameterized 5-iteration log-MAP Turbo specification with an explicit Rate-1/3 repetition coding proxy with soft-decision majority voting to enable reproducible real-time execution in Python.

### C. New Integrated Modules
1. **Edge Object Detection (`object_detection/detector.py`):**
   - YOLOv8-nano integration with local cached weights (`yolov8n.pt`, 3.2M parameters).
   - Exact spatial downsampling mapping from pixel space $(512\times 512)$ to latent coordinate space $(64\times 64)$:
     $$x_1^{(z)} = \max\left(0, \left\lfloor \frac{x_1}{8} \right\rfloor\right), \quad x_2^{(z)} = \min\left(64, \left\lceil \frac{x_2}{8} \right\rceil\right)$$
   - Object bounding box visualization, confidence filtering, and localized crop extraction.
2. **Vector Indexing & Retrieval Engine (`retrieval/`):**
   - CLIP ViT-B/32 multimodal feature extraction ($D=512$, $L_2$-normalized).
   - FAISS `IndexFlatIP` vector index supporting sub-millisecond Top-$K$ retrieval.
   - Persistent metadata store (`metadata.json` / `mapping.pkl`) tracking transmission conditions, object labels, and image paths.
   - Unified search interface supporting natural language zero-shot text queries, image-to-image search, and bounding box crop queries.
3. **Task-Aware Semantic Allocation (`semantic/task_aware.py`):**
   - Formal task profiles: Object Identification, Object Retrieval, Scene Understanding, and Custom.
   - Unified priority weighting formula combining attention, detection confidence, class priority, and task relevance:
     $$P(i,j) = w_{\text{attn}} A_{\text{norm}}(i,j) + w_{\text{det}} C_{\text{norm}}(i,j) + w_{\text{class}} K_{\text{norm}}(i,j) + w_{\text{task}} R_{\text{norm}}(i,j)$$
   - Comprehensive task evaluation metrics (mAP, Precision@K, Recall@K).

---

## 3. Comparison of Scientific Capabilities

| Capability | Base Paper | Our Proposed System | Practical Advantage |
|---|---|---|---|
| Latent Representation | Fixed 8-bit or 4-bit attention | Dynamic 2, 4, 6, 8-bit multi-level | Up to $191.8\times$ raw compression at 2-bit; granular rate-distortion control |
| Channel Adaptivity | Static / Manual configuration | Automatic SNR-driven adaptation | Avoids catastrophic corruption under severe noise by shifting to low bit-depth |
| Saliency Detection | Heuristic latent magnitude | Supervised YOLOv8-nano object detection | Accurately isolates high-value surveillance targets (persons, vehicles) |
| Post-Transmission Utility | Visual reconstruction only | Vector indexing & natural language search | Enables instant surveillance query and forensic analysis at receiver |
| Downstream Optimization | Reconstruction-only loss | Task-oriented bit prioritization | Balances bandwidth against task performance (detection mAP, retrieval P@K) |
| End-to-End Testbed | Offline dataset evaluation | Real-time smartphone-to-server dashboard | Real-world interactive validation over Wi-Fi and simulated physical channels |
