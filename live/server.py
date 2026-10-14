import os
import sys
import shutil
import base64
import json
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
import time
import math
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_vcsc import VCSCPipeline
from metrics.evaluator import Evaluator
from baselines.jpeg import JPEGBaseline
from baselines.huffman import HuffmanBaseline
from semantic.task_aware import TaskType

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
    'bandwidth': 5.0,
    'task_type': 'object_identification',
    'enable_detection': False,
    'enable_retrieval': False,
    'detection_conf': 0.25,
    'retrieval_query': '',
    'retrieval_top_k': 10
}

# Concurrency lock to drop frames if the server is busy processing the previous one
is_processing = False

print("[Server] Initializing Neural Networks. This may take a moment...")
vcsc_pipeline = VCSCPipeline(
    use_turbo=True, 
    channel_type='awgn',
    enable_object_detection=False,
    enable_retrieval=False
)
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

@app.route('/api/retrieval/search', methods=['POST'])
def api_retrieval_search():
    """API endpoint for retrieval search."""
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 10)
        filters = data.get('filters', {})
        
        if not vcsc_pipeline.enable_retrieval or not vcsc_pipeline.search_engine:
            return jsonify({'error': 'Retrieval not enabled', 'results': []})
        
        results = vcsc_pipeline.search_retrieval(query_text, top_k=top_k, **filters)
        return jsonify({
            'results': [r.to_dict() for r in results],
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'results': []})

@app.route('/api/retrieval/stats', methods=['GET'])
def api_retrieval_stats():
    """API endpoint for retrieval stats."""
    try:
        stats = vcsc_pipeline.get_retrieval_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/retrieval/add', methods=['POST'])
def api_retrieval_add():
    """API endpoint to add current frame to retrieval."""
    try:
        data = request.get_json()
        record_id = data.get('record_id')
        object_class = data.get('object_class')
        confidence = data.get('confidence')
        
        # This would need the current reconstructed image
        # For now, return not implemented
        return jsonify({'error': 'Not implemented - use current frame from last transmission'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/detection/classes', methods=['GET'])
def api_detection_classes():
    """API endpoint to get available COCO classes."""
    if vcsc_pipeline.object_detector:
        classes = vcsc_pipeline.object_detector.COCO_CLASSES
        return jsonify({'classes': classes})
    return jsonify({'classes': []})

@socketio.on('connect')
def handle_connect():
    # Send the current configuration to the newly connected client
    emit('config_updated', config_state)

@socketio.on('update_config')
def handle_config_update(data):
    global config_state, vcsc_pipeline
    old_method = config_state['method']
    old_detection = config_state.get('enable_detection', False)
    old_retrieval = config_state.get('enable_retrieval', False)
    old_task = config_state.get('task_type', 'object_identification')
    
    config_state.update(data)
    
    # Recreate pipeline if object detection or retrieval toggled
    if (config_state.get('enable_detection', False) != old_detection or 
        config_state.get('enable_retrieval', False) != old_retrieval):
        print(f"[Server] Recreating pipeline: detection={config_state.get('enable_detection')}, retrieval={config_state.get('enable_retrieval')}")
        vcsc_pipeline = VCSCPipeline(
            use_turbo=str(config_state['turbo']).lower() == 'true',
            channel_type=config_state['channel'],
            enable_object_detection=config_state.get('enable_detection', False),
            enable_retrieval=config_state.get('enable_retrieval', False)
        )
    
    # Update task profile
    new_task = config_state.get('task_type', 'object_identification')
    if new_task != old_task:
        try:
            vcsc_pipeline.set_task_profile(TaskType(new_task))
        except Exception as e:
            print(f"[Server] Failed to set task profile: {e}")
    
    # Broadcast the new configuration to all connected clients
    socketio.emit('config_updated', config_state, broadcast=True)

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
    emit('logs_cleared', broadcast=True)

@socketio.on('video_frame')
def handle_video_frame(data):
    global is_processing
    
    # If the server is currently processing a frame, drop this incoming frame
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
        task_type = config_state.get('task_type', 'object_identification')
        enable_detection = config_state.get('enable_detection', False)
        enable_retrieval = config_state.get('enable_retrieval', False)
        
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
            # VCSC with full features
            vcsc_pipeline.channel.use_turbo = use_turbo
            vcsc_pipeline.channel.channel_type = channel_type
            
            # Update task profile if changed
            if vcsc_pipeline.current_task_profile is None or vcsc_pipeline.current_task_profile.task_type.value != task_type:
                try:
                    vcsc_pipeline.set_task_profile(TaskType(task_type))
                except Exception as e:
                    print(f"[Server] Task profile error: {e}")
            
            recon_img, result = vcsc_pipeline.process_image(
                image, 
                quantization, 
                snr, 
                threshold, 
                save_dir=save_dir,
                task_type=task_type,
                return_detections=enable_detection,
                return_embeddings=enable_retrieval
            )
            
            stats = result['stats']
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
                if te.get('retrieval_precision_at_k'):
                    for k, v in te['retrieval_precision_at_k'].items():
                        logs.append(f"  Retrieval P@{k}={v:.3f}")
        
        # Simulated Bandwidth processing
        snr_linear = 10 ** (snr / 10.0)
        capacity_bps = (bandwidth_mhz * 1e6) * math.log2(1 + snr_linear)
        transmission_time_ms = (stats['comp_bits'] / capacity_bps) * 1000
        stats['transmission_time_ms'] = transmission_time_ms
        
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
        
        # Prepare response data
        response_data = {
            'original': orig_b64,
            'reconstructed': recon_b64,
            'metrics': metrics,
            'stats': stats,
            'logs': logs
        }
        
        # Add detection data if available
        if result.get('detections'):
            response_data['detections'] = [d.to_dict() for d in result['detections']]
            response_data['latent_rois'] = [r.to_dict() for r in result['latent_rois']]
        
        # Add task evaluation if available
        if result.get('task_eval'):
            response_data['task_eval'] = result['task_eval']
        
        # Add retrieval results if available
        if result.get('retrieval_results'):
            response_data['retrieval_results'] = result['retrieval_results']
        
        # Send data back to the receiver specifically
        emit('reconstructed_frame', response_data, broadcast=True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
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