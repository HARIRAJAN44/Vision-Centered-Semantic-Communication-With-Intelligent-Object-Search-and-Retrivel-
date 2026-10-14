import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from experiments.run_vcsc import VCSCPipeline
from metrics.evaluator import Evaluator


def run_snr_sweep_real(image_path: str, snrs=None, quantizations=None):
    """Run real SNR sweep using the actual VCSCPipeline."""
    if snrs is None:
        snrs = [0, 5, 10, 15, 20, 25, 30, 35]
    if quantizations is None:
        quantizations = ['8bit', '4bit-attn', '2bit', '4bit', '6bit', 'task_aware']
    
    print(f"Running real SNR sweep with {len(snrs)} SNRs x {len(quantizations)} quantizations")
    print(f"Image: {image_path}")
    
    # Initialize pipeline
    pipeline = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=False,
        enable_retrieval=False
    )
    
    # Initialize evaluator
    evaluator = Evaluator(device=pipeline.device)
    
    # Load test image
    img = Image.open(image_path).convert('RGB')
    if img.size != (512, 512):
        img = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
    
    # Convert to tensor for pipeline
    from experiments.run_vcsc import preprocess_image
    img_tensor = preprocess_image(img).to(pipeline.device)
    
    # Also load original for metrics
    orig_img = img
    
    data = []
    
    for snr in snrs:
        for qmode in quantizations:
            try:
                # Process image through pipeline
                recon_img, result = pipeline.process_image(
                    img_tensor,
                    quant_mode=qmode,
                    snr_db=snr,
                    threshold=0.2,
                    task_type='object_identification',
                    save_dir=None,
                    return_detections=False,
                    return_embeddings=False
                )
                
                # Get metrics from evaluator
                metrics = evaluator.evaluate(orig_img, recon_img)
                
                # Get stats from result
                stats = result.get('stats', {})
                
                row = {
                    'SNR': snr,
                    'Quantization': qmode,
                    'SSIM': metrics.get('ssim', '-'),
                    'PSNR': metrics.get('psnr', '-'),
                    'CLIP': metrics.get('clip_similarity', '-'),
                    'LPIPS': metrics.get('lpips', '-'),
                    'Compression': stats.get('compression_rate', '-'),
                    'Comp_Bits': stats.get('comp_bits', '-'),
                    'Tx_Delay_ms': stats.get('transmission_time_ms', '-'),
                    'Proc_Time_ms': f"{(stats.get('enc_time', 0) + stats.get('dec_time', 0)) * 1000:.0f}"
                }
                data.append(row)
                print(f"  SNR={snr} | {qmode}: SSIM={metrics.get('ssim', '-'):.3f} PSNR={metrics.get('psnr', '-'):.1f} LPIPS={metrics.get('lpips', '-'):.3f}")
                
            except Exception as e:
                print(f"  SNR={snr} | {qmode}: ERROR - {str(e)[:60]}")
                data.append({
                    'SNR': snr,
                    'Quantization': qmode,
                    'SSIM': '-',
                    'PSNR': '-',
                    'CLIP': '-',
                    'LPIPS': '-',
                    'Compression': '-',
                    'Comp_Bits': '-',
                    'Tx_Delay_ms': '-',
                    'Proc_Time_ms': '-'
                })
    
    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv('results/snr_sweep_real.csv', index=False)
    print(f"\nSaved results to results/snr_sweep_real.csv")
    
    # Plot average SSIM per quantization mode
    plt.figure(figsize=(10, 6))
    for qmode in quantizations:
        subset = df[df['Quantization'] == qmode]
        ssim_vals = subset['SSIM'].iloc[~np.array([str(s) == '-' for s in subset['SSIM']])]
        snrs_valid = subset['SNR'].iloc[~np.array([str(s) == '-' for s in subset['SSIM']])]
        if len(ssim_vals) > 0:
            plt.plot(snrs_valid, ssim_vals, marker='o', label=qmode)
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("SSIM")
    plt.title("SSIM vs SNR by Quantization Mode (Real Pipeline)")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/ssim_vs_snr.png')
    plt.close()
    
    print("Saved plot to results/ssim_vs_snr.png")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='tests/fixtures/sample.png')
    parser.add_argument('--snrs', type=str, default='0,5,10,15,20,25,30,35')
    parser.add_argument('--quantizations', type=str, default='8bit,4bit-attn,2bit,4bit,6bit,task_aware')
    args = parser.parse_args()
    
    snrs = [int(x) for x in args.snrs.split(',')]
    qmodes = args.quantizations.split(',')
    
    # Check image exists
    img_path = args.image
    if not os.path.exists(img_path):
        # Create a dummy test image
        import numpy as np
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path)
        print(f"Created dummy image at {img_path}")
    
    df = run_snr_sweep_real(img_path, snrs=snrs, quantizations=qmodes)
    print("\nSNR sweep completed!")