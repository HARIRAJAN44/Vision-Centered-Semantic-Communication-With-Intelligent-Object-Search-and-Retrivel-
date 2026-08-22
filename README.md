# Vision-Centered Semantic Communication With Intelligent Object Search and Retrieval

This project implements a **Real-Time Interactive Demonstration** of the paper *Wireless Vision-Centered Semantic Communication for Smart City Environment: Pretrained Network and Quantization*. 

It provides an end-to-end framework allowing a smartphone (transmitter) to stream captured images over a simulated noisy wireless channel (AWGN / Rayleigh) to a laptop (receiver). The system compares a Neural VAE-based Semantic Communication method (VCSC) against traditional baselines (JPEG and Huffman) under harsh channel conditions (e.g., 0 dB SNR).

## Key Features
*   **Semantic VAE Autoencoder:** Uses `runwayml/stable-diffusion-v1-5` VAE to compress a 512x512 image into a highly robust 64x64 latent semantic representation.
*   **Attention-based Feature Quantization:** Isolates and protects critical latent features (4-bit Attention) to survive extremely high noise environments where traditional 8-bit uniform quantization shatters.
*   **Turbo Coding & BPSK Modulation:** Full simulated physical layer including Forward Error Correction (FEC).
*   **Real-time Metrics:** Live evaluation of Structural Similarity (SSIM), LPIPS (Perceptual Loss), and CLIP cosine similarity.
*   **Interactive Web Dashboards:** A modern, WebRTC-style camera capture interface for smartphones and a live statistics dashboard for laptops.

---

## 🛠 Installation

1. Clone this repository to your machine.
2. Ensure you have Python 3.10+ installed.
3. Install the required dependencies:
   ```bash
   pip install torch torchvision numpy Pillow flask flask-socketio scikit-image lpips transformers
   ```

---

## 🚀 How to Run the Real-Time Demo

The system consists of a Flask-SocketIO backend that orchestrates the neural networks and simulates the wireless channel. 

1. Open your terminal and start the server:
   ```bash
   python live/server.py
   ```
2. Wait for the Neural Networks (VAE, LPIPS, CLIP) to initialize. It will print `[Server] Ready.`

### Step 1: Open the Receiver (Laptop)
On the machine running the server, open your web browser and go to:
**`http://localhost:5000/receiver`**
This is your main dashboard. You can use the sliders here to change the channel Noise (SNR) and Bandwidth in real-time.

### Step 2: Open the Transmitter (Smartphone)
Due to strict mobile browser security, accessing the camera (`getUserMedia`) requires a secure context. Because you are connecting via a local IP address (e.g., `192.168.1.x`), mobile browsers will naturally block the camera. 

You have **two methods** to bypass this and connect your phone:

#### Method A: Pure Local Wi-Fi (Android Only - Recommended)
This method keeps all traffic strictly on your local Wi-Fi router.
1. Make sure your Android phone is on the **same Wi-Fi** as your laptop.
2. Open Google Chrome on your phone.
3. Type `chrome://flags/#unsafely-treat-insecure-origin-as-secure` in the URL bar.
4. In the text box, enter your laptop's local IP address with the port (e.g., `http://192.168.1.9:5000`).
5. Change the dropdown to **Enabled** and tap **Relaunch**.
6. Now, go to `http://192.168.1.9:5000/transmitter` on your phone. The camera will instantly work!

#### Method B: Localtunnel / Ngrok (iOS & Android)
If you have an iPhone or cannot change browser flags, use a tunnel to create a temporary, valid `https://` domain.
1. Leave the python server running. Open a **new** terminal window.
2. Run Localtunnel:
   ```bash
   npx localtunnel --port 5000
   ```
3. It will generate a link like `https://some-random-words.loca.lt`.
4. On your smartphone browser, open that link and add `/transmitter` to the end:
   **`https://some-random-words.loca.lt/transmitter`**
5. Click "Click to continue" on the Localtunnel warning page, and the camera will open securely!

---

## 🔬 Demonstration Guide

To clearly demonstrate why VCSC is superior to traditional codecs, follow these steps during your presentation:

1. **Perfect Conditions (High SNR):** 
   - Set the Receiver to **JPEG Baseline** and **SNR to 25 dB**. 
   - Tap *Take Photo & Transmit* on the phone. The image arrives perfectly.
2. **Harsh Conditions (JPEG Failure):**
   - Lower the SNR to **0 dB** and tap transmit again. 
   - The JPEG file header will be completely corrupted by the noise, causing the receiver to display a black box (transmission failed).
3. **Semantic Communication (VCSC Success):**
   - Change the method to **VCSC (Neural)**.
   - Crucially, set Quantization to **4-bit Attention (Base Paper)**.
   - Leave SNR at **0 dB** and tap transmit.
   - Despite the extreme noise, the Neural Decoder successfully reconstructs the semantic meaning of the image!

---

## 🐳 Running with Docker (Portable)

You can run this entire project on any system (Windows, Mac, Linux) without installing Python dependencies by using Docker.

1. **Build the Docker Image** (Run this once in the project folder):
   ```bash
   docker build -t vcsc-server .
   ```
2. **Run the Container**:
   ```bash
   docker run -p 5000:5000 vcsc-server
   ```
3. The server is now running! You can access the receiver at `http://localhost:5000/receiver` and use the same mobile connection methods as above. 
