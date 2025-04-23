import numpy as np
import pandas as pd
import mne
import os
import logging
import sys
from tqdm import tqdm
import pickle

ALPHA_POWER = (8, 13)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing/logs/cal_alpha_powers.log')
    ]
)

def calc_alpha_power(data, fmin=0.5, fmax=35.0, epoch_duration=10.0, sfreq = 100):
    global ALPHA_POWER
    data = data.T
    ch_names = [f'EEG {i+1}' for i in range(data.shape[0])]
    ch_types = ['eeg'] * len(ch_names)
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)

    psds, freqs = epochs.compute_psd(fmin=0.5, fmax=35.0, method='multitaper').get_data(return_freqs=True)

    psds = psds / np.sum(psds, axis=-1, keepdims=True)  # (n_epochs, n_channels, n_freqs)

    alpha_idx = np.logical_and(freqs >= ALPHA_POWER[0], freqs <= ALPHA_POWER[1])
    alpha_power = psds[:, :, alpha_idx].mean(axis=-1)  # (n_epochs, n_channels)

    return alpha_power

if __name__=="__main__":
    root_path='/mnt/ssd_4tb_0/data/tuhab_preprocessed'
    csv_path='/mnt/ssd_4tb_0/data/tuhab_preprocessed/file_lengths_map.csv'

    data_filenames = os.listdir(root_path)
    data_filenames = [f for f in data_filenames if f.endswith('.npy')]
    logging.info(f'Found {len(data_filenames)} files in {root_path}')

    alpha_power_dict = {}

    pbar = tqdm(total=len(data_filenames), desc='Processing files', unit='file')
    for filename in data_filenames:
        pbar.update(1)
        logging.info(f'Processing {filename}')
        data = np.load(os.path.join(root_path, filename))
        try:
            alpha_power = calc_alpha_power(data) # (n_epochs, n_channels)
        except Exception as e:
            logging.error(f'Error processing {filename}: {e}')
            continue

        filename_prefix = filename.split('_preprocessed.npy')[0]
        alpha_power_dict[filename_prefix] = alpha_power

    pbar.close()

    # Save the alpha power dictionary to a pickle file
    with open('preprocessing/outputs/alpha_power_tuab_dict.pkl', 'wb') as f:
        pickle.dump(alpha_power_dict, f)
    logging.info(f'Saved alpha power dictionary to alpha_power_tuab_dict.pkl')
