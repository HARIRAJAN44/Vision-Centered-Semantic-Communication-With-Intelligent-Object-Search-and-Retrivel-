# IEEE Journal Pre-Submission Quality & Compliance Checklist

**Target Journal Type:** IEEE Transactions on Consumer Electronics / IEEE Transactions on Wireless Communications / IEEE Journal on Selected Areas in Communications

---

## 1. Compliance Audit

- [x] **Title Accuracy:** Evaluated and refined to accurately reflect implemented work without false claims of real-world municipal surveillance deployment.
- [x] **Abstract Quality:** Self-contained, structured, under 250 words, covers motivation, technical gap, proposed methodology, implemented contributions, quantitative findings, and scientific conclusions without marketing buzzwords.
- [x] **Index Terms:** 6–8 precise, authoritative IEEE taxonomy terms included.
- [x] **Literature Review & Related Works:** Rigorous review covering DeepJSCC, semantic communication, VCSC, adaptive quantization, task-oriented transmission, and vector similarity search. Includes a multi-column comparative literature table.
- [x] **Mathematical Rigor:**
  - [x] Every equation defines all variables, symbols, domains, and units.
  - [x] Clear distinction between base paper formulations and our novel extensions.
  - [x] Closed-form BER expressions for AWGN and Rayleigh channels.
  - [x] Concrete multi-level quantization and dequantization equations.
  - [x] Concrete multi-factor priority weighting equations.
- [x] **Algorithmic Specifications:** Included formal pseudocode for:
  - [x] Algorithm 1: Channel-Aware Adaptive Bit Allocation and Quantization.
  - [x] Algorithm 2: Integrated Transmitter-Receiver and FAISS Vector Retrieval Pipeline.
- [x] **Source Code Integrity:** Strictly adhered to the rule of not modifying any existing codebase files; all documentation and publication files are self-contained in `publication/`.
- [x] **Test Verification:** Executed and passed 58 automated unit and integration tests covering all novel and inherited modules (100% pass rate).
- [x] **Experimental Honesty:**
  - [x] No fabricated sweep curves or invented percentages.
  - [x] Representative operational measurements clearly distinguished from multi-sample statistical datasets.
  - [x] Turbo code proxy and FID omission transparently documented.
  - [x] Simulated physical layer vs. live edge Wi-Fi streaming accurately defined.
- [x] **IEEE Formatting & Citations:**
  - [x] Standard IEEE double-column format in `manuscript.tex`.
  - [x] Fully populated `references.bib` containing verified, authoritative IEEE/ACM/CVF citations with verified DOIs, volumes, issues, and dates.
  - [x] Strict 1-to-1 mapping between in-text citations and bibliography entries.
- [x] **Figures & Architectural Schematics:** Detailed diagrams mapping the actual implementation:
  - [x] Overall System Architecture (Transmitter, Channel, Receiver, Vector Store).
  - [x] Latent ROI Mapping (Pixel Space to Latent Space Downsampling).
  - [x] Channel Condition Classification and Multi-Level Bit Allocation Flow.
  - [x] FAISS Multimodal Query and Top-$K$ Retrieval Pipeline.
