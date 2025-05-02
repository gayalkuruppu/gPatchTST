#!/usr/bin/env python3
# filepath: /home/gayal/ssl-analyses-repos/NeuroGPT/preprocess_tuh.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import mne
from mne.channels import make_standard_montage
from scipy import signal
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import time
from datetime import datetime

from eeg_utils import map_tuh_to_standard_channels, get_standard_channel_lists

# https://braindecode.org/0.7/auto_examples/plot_tuh_eeg_corpus.html


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('eeg_preprocessing')

warnings.filterwarnings("ignore", category=RuntimeWarning)

def read_edf_file(file_path):
    """Read an EDF file and return a raw MNE object."""
    try:
        logger.debug(f"Reading EDF file: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        logger.debug(f"Successfully read EDF file: {file_path}")
        return raw
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None
    
def crop_data(raw, file_path, crop_start=300, crop_end=300):
    """
    Crop the raw data by removing x seconds from the beginning and y seconds from the end.
    Default is 5 minutes (300 seconds) from both ends.
    """
    try:
        logger.debug(f"Cropping data for {file_path}: start={crop_start}s, end={crop_end}s")
        total_time = raw.times[-1] - raw.times[0]  # Total recording length in seconds
        if total_time < crop_start + crop_end:
            logger.warning(f"Data too short to crop: {file_path}")
            return None  # Return None if we can't crop properly
        
        # Crop the data
        raw = raw.crop(tmin=crop_start, tmax=total_time - crop_end)
        logger.debug(f"Data cropped successfully: new length = {raw.times[-1] - raw.times[0]} seconds")
        return raw
    except Exception as e:
        logger.error(f"Error cropping data for {file_path}: {e}")
        return None

def select_channels(raw, channels_to_use_ref=None, channels_to_use_le=None):
    """
    Select desired channels from the raw EEG data.
    Handles both -REF and -LE naming conventions.
    """
    # Get standard channel lists
    channel_lists = get_standard_channel_lists()
    
    # Set default channel lists if not provided
    if channels_to_use_ref is None:
        channels_to_use_ref = channel_lists['tuh_reduced']
    if channels_to_use_le is None:
        # Create LE version from REF version
        channels_to_use_le = [ch.replace('-REF', '-LE') for ch in channels_to_use_ref]
    
    # Use the utility function to detect reference system
    from eeg_utils import detect_reference_system
    ref_system = detect_reference_system(raw)
    
    # Select the appropriate channel list based on the reference system
    if ref_system == 'ar':
        channels_to_use = channels_to_use_ref
        logger.debug("Using REF reference system")
    else:
        channels_to_use = channels_to_use_le
        logger.debug("Using LE reference system")
    
    # Get all available channel names and standardize them
    available_channels = [ch.upper() for ch in raw.ch_names]
    
    # Select only channels that we want to use and are available
    channels_present = [ch for ch in channels_to_use if ch.upper() in available_channels]
    
    logger.debug(f"Selected channels: {channels_present} ({len(channels_present)}/{len(channels_to_use)})")
    
    # If we don't have enough channels, return None
    if len(channels_present) < len(channels_to_use) * 0.8:  # 80% threshold
        logger.warning(f"Not enough channels, skipping... (Found {len(channels_present)}/{len(channels_to_use)})")
        logger.debug(f"Missing channels: {set(channels_to_use) - set(channels_present)}")
        return None
    
    # Pick only the channels we want to use
    try:
        logger.debug(f"Picking channels: {[ch.upper() for ch in channels_present]}")
        raw.pick([ch.upper() for ch in channels_present])
        logger.debug("Successfully picked channels")
        return raw
    except Exception as e:
        logger.error(f"Error picking channels: {e}")
        return None

def apply_montage(raw):
    """Apply the standard 10-20 montage to the raw data."""
    logger.debug("Setting up 10-20 montage")
    montage = make_standard_montage('standard_1020')

    # Map TUH channel names to standard 10-20 names
    logger.debug(f"Original channels: {raw.ch_names}")
    raw = map_tuh_to_standard_channels(raw)
    logger.debug(f"Renamed channels: {raw.ch_names}")
        
    # Now apply the montage
    raw.set_montage(montage, match_case=False)
    return raw

def handle_bad_channels(raw):
    """Detect and interpolate bad channels."""
    logger.debug("Detecting bad channels")
    flat_threshold = 1e-7
    bad_channels = []
    data = raw.get_data()
    
    for i, ch_name in enumerate(raw.ch_names):
        if np.std(data[i]) < flat_threshold or np.all(data[i] == 0):
            bad_channels.append(ch_name)
    
    if bad_channels:
        logger.info(f"Found {len(bad_channels)} bad channels: {bad_channels}")
        raw.info['bads'] = bad_channels
        
        # Interpolate bad channels
        try:
            logger.debug(f"Interpolating bad channels: {bad_channels}")
            raw.interpolate_bads(reset_bads=True)
            logger.debug("Successfully interpolated bad channels")
        except Exception as e:
            logger.warning(f"Error interpolating bad channels: {e}")
            logger.debug("Setting bad channel data to zeros instead")
            # If interpolation fails, we'll just set the bad channels to zeros
            for ch in bad_channels:
                idx = raw.ch_names.index(ch)
                data[idx] = np.zeros_like(data[idx])
            raw._data = data
    
    return raw

def apply_filters(raw, notch_freq=60, bandpass_freqs=(0.5, 100)):
    """Apply standard filters to the EEG data."""
    # Re-reference to average
    logger.debug("Re-referencing to average")
    raw.set_eeg_reference('average', projection=False)
    
    # Apply notch filter to remove power line noise (60 Hz)
    if bandpass_freqs[1] < notch_freq:
        logger.warning(f"Notch frequency {notch_freq} Hz is higher than the upper bandpass frequency {bandpass_freqs[1]} Hz. Skipping notch filter.")
        return raw
    else:
        logger.info(f"Applying notch filter at {notch_freq} Hz")
        raw.notch_filter(freqs=[notch_freq], picks='eeg')
    
    # Apply bandpass filter (0.5 - 100 Hz)
    logger.debug(f"Applying bandpass filter from {bandpass_freqs[0]} Hz to {bandpass_freqs[1]} Hz")
    raw.filter(l_freq=bandpass_freqs[0], h_freq=bandpass_freqs[1], picks='eeg', njobs=4, verbose=False)
    
    return raw

def resample_data(raw, target_freq=250):
    """Resample the data to the target frequency."""
    logger.debug(f"Resampling to {target_freq} Hz (from {raw.info['sfreq']} Hz)")
    raw.resample(target_freq)
    return raw

def normalize_data(data):
    """Apply signal normalization techniques."""
    # # DC offset correction (remove mean from each channel)
    # logger.debug("Applying DC offset correction")
    # data = data - np.mean(data, axis=1, keepdims=True)
    
    # # Remove linear trend from each channel
    # logger.debug("Removing linear trends")
    # for i in range(data.shape[0]):
    #     data[i] = signal.detrend(data[i])
    
    # Z-transform along time dimension
    logger.debug("Applying Z-transform")
    for channel in range(data.shape[0]):
        std = np.std(data[channel])
        if std > 0:  # Avoid division by zero
            data[channel] = (data[channel] - np.mean(data[channel])) / std
            
    return data

def standardize_channels(data, target_channels=22):
    """Ensure the data has exactly the target number of channels."""
    original_channel_count = data.shape[0]
    
    if data.shape[0] < target_channels:
        # If we have fewer channels, we'll pad with zeros
        logger.debug(f"Padding channels: {data.shape[0]} -> {target_channels}")
        padded_data = np.zeros((target_channels, data.shape[1]))
        padded_data[:data.shape[0]] = data
        data = padded_data
    elif data.shape[0] > target_channels:
        # If we have more channels, we'll take only the first N
        logger.debug(f"Truncating channels: {data.shape[0]} -> {target_channels}")
        data = data[:target_channels]
        
    return data, original_channel_count

def save_processed_data(data, file_path, output_dir, output_format):
    """Save the processed data as a PyTorch tensor."""
    base_name = os.path.basename(file_path)
    output_name = base_name.replace('.edf', f'_preprocessed.{output_format}')
    output_path = os.path.join(output_dir, output_name)
    
    # Transpose data to [time, channels] format
    data = data.T  # Transpose the data

    # Save in the requested format
    logger.debug(f"Saving preprocessed data to {output_path} in {output_format} format")
    if output_format.lower() == 'pt':
        torch.save(torch.FloatTensor(data), output_path)
    elif output_format.lower() == 'npy':
        # np.save(output_path, data)
        # save float32 instead of float64
        np.save(output_path, data.astype(np.float32))
    else:
        logger.warning(f"Unsupported format '{output_format}', defaulting to PyTorch format")
        torch.save(torch.FloatTensor(data), output_path.replace(f'.{output_format}', '.pt'))
    
    return output_path

def preprocess_eeg(file_path, args, channels_to_use_ref=None, channels_to_use_le=None):
    """
    Main function to preprocess a single EEG file and save as PT file.
    
    Args:
        file_path: Path to the EDF file
        output_dir: Data args
        channels_to_use_ref: List of REF channels to keep (if None, will use default channels)
        channels_to_use_le: List of LE channels to keep (if None, will derive from REF list)
    """
    logger.debug(f"Starting preprocessing of: {file_path}")
    
    # Get channel lists
    channel_lists = get_standard_channel_lists()
    
    # Define standard channel names if not provided
    if channels_to_use_ref is None:
        channels_to_use_ref = channel_lists['tuh_reduced']
    
    if channels_to_use_le is None:
        channels_to_use_le = [ch.replace('-REF', '-LE') for ch in channels_to_use_ref]
    
    # Step 1: Read the EDF file
    raw = read_edf_file(file_path)
    if raw is None:
        return None
    
    # Step 1.1: Crop x seconds from the begining and y seconds from the end
    raw = crop_data(raw, file_path, crop_start=args.crop_start, crop_end=args.crop_end)
    if raw is None:
        logger.warning(f"Skipping {file_path} due to cropping error")
        return None
    
    # Step 1.2: if the cropped data is shorter than 10 minutes or longer than 1 hours, skip it
    if not (args.filter_length_min <= (raw.times[-1] - raw.times[0]) <= args.filter_length_max):
        logger.warning(f"Skipping {file_path} due to insufficient time length after cropping or exceeding maximum length")
        return None

    # Step 2: Select the desired channels
    raw = select_channels(raw, channels_to_use_ref, channels_to_use_le)
    if raw is None:
        return None
    
    # Step 3: Apply montage
    raw = apply_montage(raw)
    
    # Step 4: Handle bad channels
    raw = handle_bad_channels(raw)
    
    # Step 5: Apply filters (re-referencing, notch, bandpass)
    raw = apply_filters(raw, notch_freq=args.notch_freq, bandpass_freqs=(args.bandpass_freqs[0], args.bandpass_freqs[1]))
    
    # Step 6: Resample data
    raw = resample_data(raw, target_freq=args.resample_freq)
    
    # Get the data array
    data = raw.get_data()
    logger.debug(f"Data shape after preprocessing: {data.shape}")
    
    # Step 7: Normalize data
    data = normalize_data(data) # [num_channels, num_time_points]
    
    # Step 8: Standardize channel count
    data, original_channel_count = standardize_channels(data, target_channels=19)
    
    # Step 9: Save processed data
    output_path = save_processed_data(data, file_path, args.output_dir, args.output_format)
    
    logger.info(f"Successfully preprocessed {file_path} -> {output_path} "
                f"(Original channels: {original_channel_count}, Time points: {data.shape[1]})")
    
    # Return output path and time length
    return output_path, data.shape[1]

def find_edf_files(data_dir):
    """Find all EDF files in the dataset."""
    logger.info(f"Searching for EDF files in {data_dir}...")
    edf_files = glob.glob(os.path.join(data_dir, '**', '*.edf'), recursive=True)
    logger.info(f"Found {len(edf_files)} EDF files")
    return edf_files

def create_csv_file(processed_files, output_csv):
    """Create a CSV file with the filenames and time lengths."""
    logger.info(f"Creating CSV file at {output_csv}")
    df = pd.DataFrame(processed_files, columns=['filepath', 'time_len'])
    df['filename'] = df['filepath'].apply(os.path.basename)
    df = df[['filename', 'time_len']]
    
    # Log some statistics
    logger.info(f"Time length statistics:")
    logger.info(f"- Min: {df['time_len'].min()}")
    logger.info(f"- Max: {df['time_len'].max()}")
    logger.info(f"- Mean: {df['time_len'].mean():.2f}")
    logger.info(f"- Median: {df['time_len'].median()}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Created CSV file with {len(df)} files at {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description='Preprocess TUH EEG dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with the TUH EEG dataset')
    parser.add_argument('--output-dir', type=str, default='./preprocessed_eeg_data',
                        help='Directory to save preprocessed data')
    parser.add_argument('--output-format', type=str, default='npy',
                        help='Output format for preprocessed data (pt or npy)')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to save the CSV file (default: output_dir/file_lengths_map.csv)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    # preprocess data args:
    parser.add_argument('--crop-start', type=int, default=300,
                        help='Seconds to crop from the beginning of the recording (default: 300s)')
    parser.add_argument('--crop-end', type=int, default=300,
                        help='Seconds to crop from the end of the recording (default: 300s)')
    parser.add_argument('--filter-length-min', type=int, default=600,
                        help='Minimum length of recording to process (default: 600s)')
    parser.add_argument('--filter-length-max', type=int, default=3600,
                        help='Maximum length of recording to process (default: 3600s)')
    parser.add_argument('--notch-freq', type=int, default=60,
                        help='Frequency for notch filter (default: 60 Hz)')
    parser.add_argument('--bandpass-freqs', type=int, nargs=2, default=(0.5, 35),
                        help='Frequency range for bandpass filter (default: 0.5-100 Hz)')
    parser.add_argument('--resample-freq', type=int, default=100,
                        help='Target frequency for resampling (default: 250 Hz)')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_path = f"./logs/preprocess_{timestamp}.log"
    parser.add_argument('--log-file', type=str, default=default_log_path,
                        help='Path to save log file (default: logs/preprocess_timestamp.log)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose (debug) logging')
    
    args = parser.parse_args()
    
    # Always ensure logs directory exists
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Configure file logging if specified
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")
    
    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Log start time and arguments
    start_time = time.time()
    logger.info(f"Starting preprocessing with args: {vars(args)}")

    # Log preprocessing parameters
    logger.info(f"Preprocessing parameters:")
    logger.info(f"- Crop start: {args.crop_start} seconds")
    logger.info(f"- Crop end: {args.crop_end} seconds")
    logger.info(f"- Filter length min: {args.filter_length_min} seconds")
    logger.info(f"- Filter length max: {args.filter_length_max} seconds")
    logger.info(f"- Notch frequency: {args.notch_freq} Hz")
    logger.info(f"- Bandpass frequencies: {args.bandpass_freqs[0]}-{args.bandpass_freqs[1]} Hz")
    logger.info(f"- Resample frequency: {args.resample_freq} Hz")
    logger.info(f"- Output format: {args.output_format}")
    logger.info(f"- Output directory: {args.output_dir}")
    logger.info(f"- CSV file path: {args.csv_path}")
    logger.info(f"- Max files to process: {args.max_files if args.max_files is not None else 'all'}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    if args.csv_path is None:
        args.csv_path = os.path.join(args.output_dir, 'file_lengths_map.csv')
    
    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir and csv_dir != args.output_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # Find all EDF files
    edf_files = find_edf_files(args.data_dir)
    
    if args.max_files:
        logger.info(f"Limiting processing to {args.max_files} files")
        edf_files = edf_files[:args.max_files]
    
    logger.info(f"Starting preprocessing of {len(edf_files)} files...")
    
    # Process each file
    processed_files = []
    error_count = 0
    
    for file_path in tqdm(edf_files, desc="Preprocessing files"):
        try:
            result = preprocess_eeg(file_path, args)
            if result is not None:
                output_path, time_len = result
                processed_files.append([output_path, time_len])
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")
            error_count += 1
    
    # Create CSV file
    if processed_files:
        df = create_csv_file(processed_files, args.csv_path)
    else:
        logger.error("No files were successfully processed!")
    
    # Log summary
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Preprocessing complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"- Total files found: {len(edf_files)}")
    logger.info(f"- Successfully processed: {len(processed_files)}")
    logger.info(f"- Failed: {error_count}")
    if processed_files:
        logger.info(f"- Success rate: {len(processed_files)/len(edf_files)*100:.2f}%")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"CSV file saved to: {args.csv_path}")

if __name__ == "__main__":
    main()
