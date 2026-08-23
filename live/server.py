import os
import sys
import base64
from io import BytesIO
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image
import time
import math
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_vcsc import VCSCPipeline
from metrics.evaluator import Evaluator
from baselines.jpeg import JPEGBaseline
from baselines.huffman import HuffmanBaseline

app = Flask(__name__)
# Allow CORS for WebSockets, use threading so we can use Werkzeug's adhoc SSL
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state to ensure transmitter and receiver are synchronized
config_state = {
    'method': 'vcsc',
    'quantization': '8bit',
    'channel': 'awgn',
    'snr': 10.0,
    'threshold': 0.2,
    'turbo': True,
    'bandwidth': 5.0
}

# Concurrency lock to drop frames if the server is busy processing the previous one
is_processing = False

print("[Server] Initializing Neural Networks. This may take a moment...")
vcsc_pipeline = VCSCPipeline(use_turbo=True, channel_type='awgn')
jpeg_pipeline = JPEGBaseline(quality=80, use_turbo=True, channel_type='awgn')
huffman_pipeline = HuffmanBaseline(use_turbo=True, channel_type='awgn')
evaluator = Evaluator(device=vcsc_pipeline.device)
print("[Server] Ready.")

@app.route('/transmitter')
def transmitter():
    return render_template('transmitter.html')

@app.route('/receiver')
def receiver():
    return render_template('receiver.html')

@app.route('/')
def index():
    # Redirect root to receiver for convenience
    return render_template('receiver.html')

@socketio.on('connect')
def handle_connect():
    # Send the current configuration to the newly connected client
    emit('config_updated', config_state)

@socketio.on('update_config')
def handle_config_update(data):
    global config_state
    config_state.update(data)
    # Broadcast the new configuration to all connected clients (so transmitter updates UI)
    socketio.emit('config_updated', config_state, broadcast=True)

@socketio.on('video_frame')
def handle_video_frame(data):
    global is_processing
    
    # If the server is currently processing a frame, drop this incoming frame
    # to maintain real-time low-latency and prevent a massive queue backup.
    if is_processing:
        return
        
    is_processing = True
    try:
        # Decode base64 image
        header, encoded = data['image'].split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Ensure image is 512x512 for VCSC
        if image.size != (512, 512):
            image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
            
        method = config_state['method']
        snr = float(config_state['snr'])
        use_turbo = str(config_state['turbo']).lower() == 'true'
        channel_type = config_state['channel']
        bandwidth_mhz = float(config_state['bandwidth'])
        quantization = config_state['quantization']
        threshold = float(config_state['threshold'])
        
        # Setup intermediate downloads directory
        tx_id = str(int(time.time() * 1000))
        save_dir = os.path.join(app.static_folder, 'downloads', tx_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save original
        image.save(os.path.join(save_dir, 'original.png'))
        orig_url = f"/static/downloads/{tx_id}/original.png"
        
        t0 = time.time()
        
        if method == 'jpeg':
            jpeg_pipeline.channel.use_turbo = use_turbo
            jpeg_pipeline.channel.channel_type = channel_type
            recon_img, comp_bits, orig_bits = jpeg_pipeline.process(image, snr, save_dir=save_dir)
            t1 = time.time()
            stats = {'compression_rate': (comp_bits / orig_bits) * 100, 'enc_time': t1-t0, 'dec_time': 0, 'comp_bits': comp_bits}
            comp_url = f"/static/downloads/{tx_id}/compressed.jpeg"
            logs = [
                f"Original Image -> <a href='{orig_url}' target='_blank'>Download</a>",
                f"Using JPEG transmission... -> <a href='{comp_url}' target='_blank'>Download Payload</a>",
                f"{channel_type.upper()} channel applied (SNR: {snr}dB)"
            ]
            
        elif method == 'huffman':
            huffman_pipeline.channel.use_turbo = use_turbo
            huffman_pipeline.channel.channel_type = channel_type
            recon_img, comp_bits, orig_bits = huffman_pipeline.process(image, snr, save_dir=save_dir)
            t1 = time.time()
            stats = {'compression_rate': (comp_bits / orig_bits) * 100, 'enc_time': t1-t0, 'dec_time': 0, 'comp_bits': comp_bits}
            comp_url = f"/static/downloads/{tx_id}/compressed.bin"
            logs = [
                f"Original Image -> <a href='{orig_url}' target='_blank'>Download</a>",
                f"Using Huffman (zlib) transmission... -> <a href='{comp_url}' target='_blank'>Download Payload</a>",
                f"{channel_type.upper()} channel applied (SNR: {snr}dB)"
            ]
            
        else:
            # VCSC
            vcsc_pipeline.channel.use_turbo = use_turbo
            vcsc_pipeline.channel.channel_type = channel_type
            recon_img, stats = vcsc_pipeline.process_image(image, quantization, snr, threshold, save_dir=save_dir)
            stats['comp_bits'] = stats['compressed_bits']
            
            latent_url = f"/static/downloads/{tx_id}/latent.pt"
            comp_url = f"/static/downloads/{tx_id}/quantized_stream.bin"
            
            logs = [
                f"Original Image -> <a href='{orig_url}' target='_blank'>Download</a>",
                f"VAE encoding ({stats['enc_time']:.2f}s) -> Latent 1x4x64x64 -> <a href='{latent_url}' target='_blank'>Download Latent</a>",
                f"Quantization ({quantization}) & Serialization -> <a href='{comp_url}' target='_blank'>Download Payload</a>",
                "Turbo encoding" if use_turbo else "Turbo bypassed",
                f"BPSK Mod -> {channel_type.upper()} Channel (SNR: {snr}dB) -> BPSK Demod",
                "Turbo decoding" if use_turbo else "Turbo bypassed",
                f"Dequantization -> VAE decoding ({stats['dec_time']:.2f}s)"
            ]
            
        # Simulated Bandwidth processing
        snr_linear = 10 ** (snr / 10.0)
        capacity_bps = (bandwidth_mhz * 1e6) * math.log2(1 + snr_linear)
        transmission_time_ms = (stats['comp_bits'] / capacity_bps) * 1000
        stats['transmission_time_ms'] = transmission_time_ms
        
        # In a strict simulation, if transmission_time_ms > frame_interval (e.g. 100ms for 10fps),
        # we would artificially drop frames. 
        # We simulate the delay visually in the receiver by limiting the push rate if needed,
        # but here we just pass the metric to the frontend to handle.
        
        # Evaluate
        if recon_img.size != (512, 512):
            recon_img = recon_img.resize((512, 512), resample=Image.Resampling.LANCZOS)
            
        metrics = evaluator.evaluate(image, recon_img)
        
        # Save reconstructed image
        recon_path = os.path.join(save_dir, 'reconstructed.png')
        recon_img.save(recon_path)
        recon_url = f"/static/downloads/{tx_id}/reconstructed.png"
        logs.append(f"Reconstructed Image -> <a href='{recon_url}' target='_blank'>Download</a>")
        
        # Convert images to base64
        buffered_orig = BytesIO()
        image.save(buffered_orig, format="JPEG", quality=80)
        orig_b64 = base64.b64encode(buffered_orig.getvalue()).decode('utf-8')
        
        buffered_recon = BytesIO()
        recon_img.save(buffered_recon, format="JPEG", quality=80)
        recon_b64 = base64.b64encode(buffered_recon.getvalue()).decode('utf-8')
        
        # Send data back to the receiver specifically
        emit('reconstructed_frame', {
            'original': orig_b64,
            'reconstructed': recon_b64,
            'metrics': metrics,
            'stats': stats,
            'logs': logs
        }, broadcast=True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
    finally:
        is_processing = False

if __name__ == '__main__':
    print("="*60)
    print("VCSC WIRELESS SURVEILLANCE SERVER")
    print("="*60)
    print("LAN access (HTTP):    http://0.0.0.0:5000")
    print("Use Ngrok or Localtunnel to expose this port for your smartphone camera.")
    print("="*60)
    
    # Run standard HTTP on port 5000 so localtunnel can proxy it perfectly
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
