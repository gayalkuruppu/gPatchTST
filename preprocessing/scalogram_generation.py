import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from pywt import frequency2scale


def compute_scales(wavelet, freqs, sampling_period):
    return [frequency2scale(wavelet, freq)/sampling_period for freq in freqs]
    

def save_wavelet(data, fname, wavelet="cmor1.5-1.0", time_window=10, sampling_rate=100, log_scale=True, image_size=256):
    time = np.linspace(0, time_window, num=data.shape[0])

    dpi =  image_size
    fig_size_inch = 1  # image_size px / image_size dpi = 1 inch

    fig, ax = plt.subplots(figsize=(fig_size_inch, fig_size_inch), dpi=dpi)
    
    if log_scale:
        widths = np.geomspace(3, 100, num=75) # hfreq = 100/3 , lfreq = 100/100
    else:
        widths = np.linspace(3, 100, num=75)
    cwtmatr, freqs = pywt.cwt(
        data, widths, wavelet, sampling_period=1./sampling_rate
    )
    cwtmatr = np.abs(cwtmatr[:-1, :-1])
    
    ax.pcolormesh(time, freqs, cwtmatr, shading='auto')
    if log_scale:
        ax.set_yscale('log')
    ax.axis('off')

    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return

def process_file(file_path, output_dir, wavelet="cmor1.5-1.0", sampling_rate=100, time_window=10, log_scale=True, image_size=256):
    np_rec = np.load(file_path)
    num_samples, num_channels = np_rec.shape
    segment_length = sampling_rate * time_window
    num_segments = num_samples // segment_length

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    logging.info(f"Segment length: {segment_length}, Number of segments: {num_segments}")
    with tqdm(total=num_segments, desc=f"Processing {base_name}", unit="segment") as pbar:
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = start_idx + segment_length
            for channel_idx in range(num_channels):
                segment_data = np_rec[start_idx:end_idx, channel_idx]
                fname = f"{base_name}_start_{segment_idx:03d}_ch_{channel_idx:02d}.png"
                save_path = os.path.join(output_dir, fname)
                save_wavelet(segment_data, save_path, wavelet=wavelet, time_window=time_window, 
                             sampling_rate=sampling_rate, log_scale=log_scale)
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Generate scalograms from .npy files.")
    parser.add_argument("--input_dir", type=str, help="Directory containing .npy files.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save scalograms. Defaults to <input_dir_basename>_scalogram."
    )
    parser.add_argument("--sampling_rate", type=int, default=100, help="Sampling rate of the data.")
    parser.add_argument("--time_window", type=int, default=10, help="Time window in seconds for each segment.")
    parser.add_argument(
        "--wavelet", type=str, default="cmor1.5-1.0",
        help="Wavelet to use for the scalogram. Default is 'cmor1.5-1.0'."
    )
    parser.add_argument("--log_scale", type=bool, default=True, help="Use logarithmic scale for the y-axis.")
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Size of the output image in pixels. Default is 256."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    input_dir_basename = os.path.basename(os.path.normpath(input_dir))
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_dir), f"{input_dir_basename}_scalogram")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"preprocessing/logs/scalogram_generation_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Arguments: sampling_rate={args.sampling_rate}, time_window={args.time_window}, "
                 f"wavelet={args.wavelet}, log_scale={args.log_scale}, image_size={args.image_size}")

    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    logging.info(f"Found {len(npy_files)} .npy files in the input directory.")
    with tqdm(total=len(npy_files), desc="Processing files", unit="file") as file_pbar:
        for file_name in npy_files:
            file_path = os.path.join(input_dir, file_name)
            logging.info(f"Processing file: {file_name}")
            process_file(file_path, output_dir, wavelet=args.wavelet, 
                         sampling_rate=args.sampling_rate, time_window=args.time_window,
                         log_scale=args.log_scale, image_size=args.image_size)
            file_pbar.update(1)

if __name__ == "__main__":
    main()
