import numpy as np
import pandas as pd
import mne
import os
import logging
import sys
sys.path.append('/home/gayal/ssl-project/gpatchTST')
from tqdm import tqdm
import pickle
from datetime import datetime

from data import get_tuh_dataloaders_old_splits

DELTA_POWER = (1, 4)
THETA_POWER = (4, 8)
ALPHA_POWER = (8, 13)
BETA_POWER = (13, 25)
GAMMA_POWER = (25, 35)
SAMPLING_FREQ = 100

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"preprocessing/logs/power_band_calc_from_dl_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

def calc_rel_power_bands(data, fmin=0.5, fmax=35.0, epoch_duration=10.0, sfreq = 100):
    global ALPHA_POWER, DELTA_POWER, THETA_POWER, BETA_POWER, GAMMA_POWER
    data = data.T
    ch_names = [f'EEG {i+1}' for i in range(data.shape[0])]
    ch_types = ['eeg'] * len(ch_names)
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)

    psds, freqs = epochs.compute_psd(fmin=0.5, fmax=35.0, method='multitaper').get_data(return_freqs=True)

    psds = psds / np.sum(psds, axis=-1, keepdims=True)  # (n_epochs, n_channels, n_freqs)

    alpha_idx = np.logical_and(freqs >= ALPHA_POWER[0], freqs <= ALPHA_POWER[1])
    alpha_power = psds[:, :, alpha_idx].sum(axis=-1)  # (n_epochs, n_channels)

    delta_idx = np.logical_and(freqs >= DELTA_POWER[0], freqs <= DELTA_POWER[1])
    delta_power = psds[:, :, delta_idx].sum(axis=-1)  # (n_epochs, n_channels)

    theta_idx = np.logical_and(freqs >= THETA_POWER[0], freqs <= THETA_POWER[1])
    theta_power = psds[:, :, theta_idx].sum(axis=-1)  # (n_epochs, n_channels)

    beta_idx = np.logical_and(freqs >= BETA_POWER[0], freqs <= BETA_POWER[1])
    beta_power = psds[:, :, beta_idx].sum(axis=-1)  # (n_epochs, n_channels)

    gamma_idx = np.logical_and(freqs >= GAMMA_POWER[0], freqs <= GAMMA_POWER[1])
    gamma_power = psds[:, :, gamma_idx].sum(axis=-1)  # (n_epochs, n_channels)

    return alpha_power, delta_power, theta_power, beta_power, gamma_power

if __name__=="__main__":
    root_path='/mnt/ssd_4tb_0/data/tuhab_preprocessed'
    epoch_duration = 10.0
    patch_length = 100

    data_filenames = os.listdir(root_path)
    data_filenames = [f for f in data_filenames if f.endswith('.npy')]
    logging.info(f'Found {len(data_filenames)} files in {root_path}')

    powers_dict = {}

    _, _, test_loader = get_tuh_dataloaders_old_splits(
        root_path,
        data_path='',
        csv_path='/mnt/ssd_4tb_0/data/tuhab_preprocessed/file_lengths_map.csv',
        batch_size=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=True,
        drop_last=False,
        size=[SAMPLING_FREQ*int(epoch_duration), 0, patch_length] # (seq_len, target_dim, patch_length)
    )

    pbar = tqdm(total=len(test_loader), desc='Processing files', unit='file')
    logging.info(f'Processing {len(test_loader)} files')
    for batch in test_loader:
        pbar.update(1)
        try:
            data = batch['past_values'].numpy().squeeze(0)  # (n_channels, n_samples)
            ap, dp, tp, bp, gp = calc_rel_power_bands(data, epoch_duration=epoch_duration)  # (n_epochs, n_channels)
        except Exception as e:
            logging.error(f'Error processing batch: {e}')
            continue

        filename_prefix = batch['filename'][0].split('_preprocessed.npy')[0]
        powers_dict[filename_prefix] = [ap, dp, tp, bp, gp]

    pbar.close()

    # Save the alpha power dictionary to a pickle file
    with open(f'preprocessing/outputs/powers_dict_patch_len_{patch_length}_seq_len_{int(epoch_duration)}_secs.pkl', 'wb') as f:
        pickle.dump(powers_dict, f)

    logging.info(f'Saved all power dictionaries to pickle files')
