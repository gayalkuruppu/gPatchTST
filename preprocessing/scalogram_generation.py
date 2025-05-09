import numpy as np
import pywt
import os
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from pywt import frequency2scale
import multiprocessing

def compute_scales(wavelet, sampling_period, l_freq, h_freq, image_height, log_scale):
    if log_scale:
        freqs = np.geomspace(l_freq, h_freq, num=image_height)
    else:
        freqs = np.linspace(l_freq, h_freq, num=image_height)
    return [frequency2scale(wavelet, freq)/sampling_period for freq in freqs]

def save_wavelet(data, save_path, wavelet, sampling_period, l_freq=0.5, h_freq=35, image_height=256, log_scale=True):
    widths = compute_scales(wavelet, sampling_period, l_freq, h_freq, image_height, log_scale)
    cwtmatr, freqs = pywt.cwt(
        data, widths, wavelet, sampling_period=sampling_period
    )
    cwtmatr = np.abs(cwtmatr) # dtype('float32')
    
    # Save the scalogram as a NumPy file
    np.save(save_path, cwtmatr)

def process_file(file_path, output_dir, wavelet="cmor1.5-1.0", sampling_rate=100, time_window=10, 
                 log_scale=True, image_height=256, l_freq=0.5, h_freq=35):
    try:
        np_rec = np.load(file_path)
        num_samples, num_channels = np_rec.shape
        segment_length = sampling_rate * time_window
        num_segments = num_samples // segment_length

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        logging.info(f"Segment length: {segment_length}, Number of segments: {num_segments}")

        # # if first segment and first channel is already processed, skip the rest
        # if os.path.exists(os.path.join(output_dir, f"{base_name}_start_000_ch_00.npy")):
        #     logging.info(f"File {file_path} already processed. Skipping.")
        #     return

        # with tqdm(total=num_segments, desc=f"Processing {base_name}", unit="segment") as pbar:
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = start_idx + segment_length
            for channel_idx in range(num_channels):
                segment_data = np_rec[start_idx:end_idx, channel_idx]
                fname = f"{base_name}_start_{segment_idx:03d}_ch_{channel_idx:02d}.npy"
                save_path = os.path.join(output_dir, fname)

                # new_base_name = base_name.split('_preprocessed')[0]
                # new_fname = fname = f"{new_base_name}_start_{segment_idx:03d}_ch_{channel_idx:02d}.npy"
                # new_save_path = os.path.join(output_dir, new_fname)
                # if os.path.exists(save_path):
                #     os.rename(save_path, new_save_path)
                #     continue
                # save_wavelet(segment_data, new_save_path, wavelet=wavelet, sampling_period=1. / sampling_rate, 
                #             l_freq=l_freq, h_freq=h_freq, image_height=image_height, log_scale=log_scale)
                if os.path.exists(save_path):
                    continue
                else:
                    # # change output directory 
                    save_path = os.path.join(output_dir+'_new', fname)
                    save_wavelet(segment_data, save_path, wavelet=wavelet, sampling_period=1. / sampling_rate, 
                                l_freq=l_freq, h_freq=h_freq, image_height=image_height, log_scale=log_scale)
                # pbar.update(1)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

def process_file_wrapper(args_tuple):
    """Wrapper function to call process_file with the necessary arguments."""
    file_name, input_dir, output_dir, args = args_tuple
    return process_file(
        os.path.join(input_dir, file_name), output_dir, wavelet=args.wavelet,
        sampling_rate=args.sampling_rate, time_window=args.time_window,
        log_scale=args.log_scale, image_height=args.image_height,
        l_freq=args.l_freq, h_freq=args.h_freq
    )

def main():
    parser = argparse.ArgumentParser(description="Generate scalograms from .npy files.")
    parser.add_argument("--input_dir", type=str, help="Directory containing .npy files.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save scalograms. Defaults to <input_dir_basename>_scalograms."
    )
    parser.add_argument("--sampling_rate", type=int, default=100, help="Sampling rate of the data.")
    parser.add_argument("--time_window", type=int, default=10, help="Time window in seconds for each segment.")
    parser.add_argument(
        "--wavelet", type=str, default="cmor1.5-1.0",
        help="Wavelet to use for the scalogram. Default is 'cmor1.5-1.0'."
    )
    parser.add_argument("--log_scale", type=bool, default=True, help="Use logarithmic scale for the y-axis.")
    parser.add_argument(
        "--image_height", type=int, default=250,
        help="Size of the output image in pixels (height of the scalogram). Default is 250."
    )
    parser.add_argument("--l_freq", type=float, default=0.5, help="Lower frequency bound for wavelet transform.")
    parser.add_argument("--h_freq", type=float, default=35, help="Higher frequency bound for wavelet transform.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes to use for parallel processing.")
    args = parser.parse_args()

    input_dir = args.input_dir
    input_dir_basename = os.path.basename(os.path.normpath(input_dir))
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_dir), f"{input_dir_basename}_scalograms")
    os.makedirs(output_dir, exist_ok=True)

    new_output_basename = os.path.basename(output_dir) + '_new'
    new_output_dir = os.path.join(os.path.dirname(output_dir), new_output_basename)
    os.makedirs(new_output_dir, exist_ok=True)

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
                 f"wavelet={args.wavelet}, log_scale={args.log_scale}, image_height={args.image_height}, "
                 f"l_freq={args.l_freq}, h_freq={args.h_freq}, num_workers={args.num_workers}")

    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    logging.info(f"Found {len(npy_files)} .npy files in the input directory.")
    if args.num_workers==-1:
        args.num_workers = multiprocessing.cpu_count()

    logging.info(f"Using {args.num_workers} worker processes.")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(npy_files), desc="Processing files", unit="file") as file_pbar:
            for _ in pool.imap_unordered(
                process_file_wrapper,
                [(file_name, input_dir, output_dir, args) for file_name in npy_files]
            ):
                file_pbar.update(1)
    
    # Uncomment the following lines to use the original for loop instead of multiprocessing
    # for file_name in tqdm(npy_files, desc="Processing files", unit="file"):
    #     process_file_wrapper((file_name, input_dir, output_dir, args))

    logging.info("All files processed.")

if __name__ == "__main__":
    main()
