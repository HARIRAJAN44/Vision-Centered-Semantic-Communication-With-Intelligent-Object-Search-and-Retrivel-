import os
import sys
import shutil
import base64
from io import BytesIO
<<<<<<< HEAD
from flask import Flask, render_template, request
=======
from flask import Flask, render_template, request, jsonify, send_from_directory
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
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

app = Flask(__name__, static_folder='static')
# Allow CORS for WebSockets
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Serve Socket.IO client library
@app.route('/socket.io/socket.io.js')
def socket_io_client():
    """Serve the Socket.IO client library."""
    try:
        import flask_socketio
        flask_socketio_path = os.path.dirname(flask_socketio.__file__)
        # Try to find the client library
        for root, dirs, files in os.walk(flask_socketio_path):
            if 'socket.io.js' in files:
                return send_from_directory(root, 'socket.io.js')
            if 'socket.io.min.js' in files:
                return send_from_directory(root, 'socket.io.min.js')
    except:
        pass
    # Fallback: serve from static folder if available
    try:
        return send_from_directory(app.static_folder, 'js/socket.io.min.js')
    except:
        return "// Socket.IO client not found - use CDN", 404

# Global state to ensure transmitter and receiver are synchronized
config_state = {
    'method': 'vcsc',
    'quantization': '8bit',
    'channel': 'awgn',
    'snr': 10.0,
    'threshold': 0.2,
    'turbo': True,
<<<<<<< HEAD
    'bandwidth': 5.0
=======
    'bandwidth': 5.0,
    'enable_detection': False,
    'detection_conf': 0.25
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
}

# Concurrency lock to drop frames if the server is busy processing the previous one
is_processing = False

print("[Server] Initializing Neural Networks. This may take a moment...")
<<<<<<< HEAD
vcsc_pipeline = VCSCPipeline(use_turbo=True, channel_type='awgn')
=======
vcsc_pipeline = VCSCPipeline(
    use_turbo=True, 
    channel_type='awgn',
    enable_object_detection=False
)
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
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

<<<<<<< HEAD
=======
@app.route('/api/detection/classes', methods=['GET'])
def api_detection_classes():
    """API endpoint to get available COCO classes."""
    if vcsc_pipeline.object_detector:
        classes = vcsc_pipeline.object_detector.COCO_CLASSES
        return jsonify({'classes': classes})
    return jsonify({'classes': []})

>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
@socketio.on('connect')
def handle_connect():
    print("[Server] Client connected, sending config")
    # Send the current configuration to the newly connected client
    emit('config_updated', config_state)

@socketio.on('update_config')
def handle_config_update(data):
<<<<<<< HEAD
    global config_state
    config_state.update(data)
    # Broadcast the new configuration to all connected clients (so transmitter updates UI)
    socketio.emit('config_updated', config_state, broadcast=True)
=======
    global config_state, vcsc_pipeline
    old_method = config_state['method']
    old_detection = config_state.get('enable_detection', False)
    
    config_state.update(data)
    
    # Log the config changes for debugging
    print(f"[Server] Config update: method={data.get('method')}, quant={data.get('quantization')}, detect={data.get('enable_detection')}")
    
    # Recreate pipeline if object detection toggled
    if config_state.get('enable_detection', False) != old_detection:
        print(f"[Server] Recreating pipeline: detection={config_state.get('enable_detection')}")
        vcsc_pipeline = VCSCPipeline(
            use_turbo=str(config_state['turbo']).lower() == 'true',
            channel_type=config_state['channel'],
            enable_object_detection=config_state.get('enable_detection', False)
        )
    
    # Handle method/quantization changes - reset pipeline
    new_method = config_state.get('method', old_method)
    if new_method != old_method:
        print(f"[Server] Method changed from '{old_method}' to '{new_method}', resetting pipeline")
        vcsc_pipeline = VCSCPipeline(
            use_turbo=str(config_state['turbo']).lower() == 'true',
            channel_type=config_state['channel'],
            enable_object_detection=config_state.get('enable_detection', False)
        )
    
    # Broadcast the new configuration to all connected clients
    socketio.emit('config_updated', config_state)
    print(f"[Server] config_updated emitted to all clients")
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)

@socketio.on('clear_logs')
def handle_clear_logs():
    # Delete contents of downloads folder
    downloads_dir = os.path.join(app.static_folder, 'downloads')
    if os.path.exists(downloads_dir):
        for item in os.listdir(downloads_dir):
            item_path = os.path.join(downloads_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
    # Tell clients to clear their UI logs
    socketio.emit('logs_cleared')

def process_frame_async(data):
    """Process a frame in background thread and emit result via socketio."""
    global is_processing
<<<<<<< HEAD
    
    # If the server is currently processing a frame, drop this incoming frame
    # to maintain real-time low-latency and prevent a massive queue backup.
    if is_processing:
        return
        
    is_processing = True
=======
    result = None
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
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
<<<<<<< HEAD
=======
        enable_detection = config_state.get('enable_detection', False)
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
        
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
<<<<<<< HEAD
            recon_img, stats = vcsc_pipeline.process_image(image, quantization, snr, threshold, save_dir=save_dir)
=======
            
            recon_img, result = vcsc_pipeline.process_image(
                image, 
                quantization, 
                snr, 
                threshold, 
                save_dir=save_dir,
                return_detections=enable_detection
            )
            
            stats = result['stats']
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
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
            
<<<<<<< HEAD
=======
            # Add detection info to logs
            if enable_detection and result.get('detections'):
                det_url = f"/static/downloads/{tx_id}/detections.png"
                logs.append(f"Object Detection: {len(result['detections'])} objects found -> <a href='{det_url}' target='_blank'>Download Visualization</a>")
                for det in result['detections'][:5]:
                    logs.append(f"  {det.class_name}: {det.confidence:.2f} @ {det.bbox}")
                if len(result['detections']) > 5:
                    logs.append(f"  ... and {len(result['detections']) - 5} more")
            
            # Add task evaluation to logs
            if result.get('task_eval'):
                te = result['task_eval']
                logs.append(f"Task Evaluation ({te['task_name']}): SSIM={te['ssim']:.3f}, CLIP={te['clip_similarity']:.3f}")
                if te.get('detection_precision'):
                    logs.append(f"  Detection P={te['detection_precision']:.3f}, R={te['detection_recall']:.3f}, mAP={te['detection_map']:.3f}")
        
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
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
<<<<<<< HEAD
        }, broadcast=True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
=======
        }
        
        # Add detection data if available (VCSC only)
        if method == 'vcsc' and result and result.get('detections'):
            response_data['detections'] = [d.to_dict() for d in result['detections']]
            response_data['latent_rois'] = [r.to_dict() for r in result['latent_rois']]
        
        # Add task evaluation if available (VCSC only)
        if method == 'vcsc' and result and result.get('task_eval'):
            response_data['task_eval'] = result['task_eval']
        
        # Emit result via socketio (thread-safe)
        socketio.emit('reconstructed_frame', response_data)
        print(f"[Server] Frame processed and emitted (tx_id: {tx_id})")
        
    except Exception as e:
        print(f"[Server] Error processing frame: {e}")
        import traceback
        traceback.print_exc()
>>>>>>> a8a67b3 (enabling object detection with proper files no sensitive dats leaked okay with proper readme files)
    finally:
        is_processing = False

@socketio.on('video_frame')
def handle_video_frame(data):
    global is_processing
    
    # If the server is currently processing a frame, drop this incoming frame
    if is_processing:
        print("[Server] Dropped frame - server busy, queue full")
        return
        
    is_processing = True
    # Use socketio background task for proper context
    socketio.start_background_task(process_frame_async, data)

if __name__ == '__main__':
    print("="*60)
    print("VCSC WIRELESS SURVEILLANCE SERVER")
    print("="*60)
    print("LAN access (HTTP):    http://0.0.0.0:5000")
    print("Use Ngrok or Localtunnel to expose this port for your smartphone camera.")
    print("="*60)
    
    # Run standard HTTP on port 5000 so localtunnel can proxy it perfectly
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
