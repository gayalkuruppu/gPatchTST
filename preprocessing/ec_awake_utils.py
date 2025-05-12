import numpy as np
import mne
from mne_features.univariate import compute_samp_entropy as entropy
import yasa


CH_NAMES_10_20 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'   
]
CH_NAMES_10_20 = [x.lower() for x in CH_NAMES_10_20]

# https://gitlab-03.engr.illinois.edu/varatha2/eeg_ingestion/-/blob/master/feature_extraction/eeg_stats.py?ref_type=heads#L120
def count_eye_blinks(wake_epochs, threshold='auto'):
    '''
    Compute the number of eye blinks in each epoch.
    Recommended:
        1. Pass only wake epochs
        2. Do not perform artifact removal prior to this.
        Remove bad epochs if possible (flat-line channel, etc.)
    Args:
        wake_epochs: MNE epoch structure containing wake data only.
        threshold: threshold to detect eye blinks (pass 3e-4 for epochs that may not contain blinks, otherwise 'auto')

    Returns:
        blink_counts: # blinks in each epoch
    '''

    combined_data = mne.concatenate_epochs([wake_epochs[i] for i in range(len(wake_epochs))], add_offset=False)
    split_data = np.vsplit(combined_data.get_data(), len(combined_data))
    full_data = np.squeeze(np.dstack(split_data), axis=0)
    full_raw = mne.io.RawArray(full_data, wake_epochs.info)

    eye_channels = [ch for ch in full_raw.ch_names if 'fp' in ch.lower()]
    used_eye_channel = 'fpz' if 'fpz' in eye_channels else eye_channels[0]

    eog_signal = np.reshape(full_raw.copy().pick(picks=used_eye_channel).get_data(), [-1])
    if threshold == 'auto':
        thresh = min(3e-4, np.abs(np.max(eog_signal) - np.min(eog_signal)) / 8)
    else:
        thresh = threshold

    filter_length = mne.filter.next_fast_len(int(round(4 * full_raw.info['sfreq'])))

    blink_counts = []
    for i in range(len(wake_epochs)):

        epoch = wake_epochs[i]

        epoch_data = np.squeeze(epoch.get_data(), axis=0)
        epoch_raw = mne.io.RawArray(epoch_data, wake_epochs.info)

        eog_events = mne.preprocessing.find_eog_events(epoch_raw, ch_name=used_eye_channel, h_freq=5.,
                                                       thresh=thresh, filter_length=filter_length)
        blink_counts.append(len(eog_events))

    return np.array(blink_counts), thresh


yasa_behavioral_state_mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'N/A': 5}
def _run_yasa_sleep_staging(eeg_data, eeg_channel='c4', is_epochs=True):
    '''
    setup for YASA call
    '''
    if is_epochs:
        combined_data = mne.concatenate_epochs([eeg_data[i] for i in range(len(eeg_data))], add_offset=False)
        split_data = np.vsplit(combined_data.get_data(), len(combined_data))
        full_data = np.squeeze(np.dstack(split_data), axis=0)
        full_raw = mne.io.RawArray(full_data, eeg_data.info)
    else:
        full_raw = eeg_data

    eye_channels = [ch for ch in eeg_data.ch_names if 'fp' in ch.lower()]
    emg_name = None

    '''
    run YASA
    '''
    sls = yasa.SleepStaging(full_raw, eeg_name=eeg_channel,
                            eog_name=eye_channels[0],
                            emg_name=emg_name)
    # Return the predicted sleep stage for each 30-sec epoch of data.
    sleep_stages = sls.predict()

    # one 30s prediction from YASA = 3 10s predictions for input
    sleep_stages_10s = []
    for pred in sleep_stages:
        temp = [pred] * 3
        sleep_stages_10s += temp

    '''
    CAUTION: last one or two epochs in eeg_data will not be labeled by YASA 
    and therefore will be ignored in find_eyes_closed_epochs_alpha_pow()
    '''
    # print("DEBUG:", len(sleep_stages_10s), len(eeg_data))
    # print("DEBUG:", sleep_stages_10s)
    # assert len(sleep_stages_10s) == len(eeg_data)
    if len(sleep_stages_10s) != len(eeg_data):
        sleep_stages_10s += ['N/A'] * (len(eeg_data) - len(sleep_stages_10s))
    
    return sleep_stages_10s

    
# https://gitlab-03.engr.illinois.edu/varatha2/eeg_ingestion/-/blob/master/labeling/sleep_staging.py?ref_type=heads#L211
def find_eyes_closed_epochs_alpha_pow(epochs, eeg_channel_names=CH_NAMES_10_20,
                                      remove_on_entropy=True, max_epochs=6, min_epochs=3):
        '''
        Identify eyes closed epochs in a series of epochs.
        Args:
            epochs:
            eeg_channel_names:
            max_epochs: max requested number of epochs
            min_epochs: min requested number of epochs
            remove_on_entropy: whether to remove epochs that have low entropy
        Returns:
            ec_epochs
        '''


        '''
        01: run auto sleep staging
        '''
        sleep_stages = _run_yasa_sleep_staging(epochs)
        sleep_stages_numeric = np.vectorize(yasa_behavioral_state_mapping.get)(sleep_stages)
        all_wake_epochs = epochs[sleep_stages_numeric == 0]

        if (len(all_wake_epochs) > max_epochs):
            '''
            02: run blink detection
            '''
            blink_counts, thresh = count_eye_blinks(all_wake_epochs)
            ec_epochs = all_wake_epochs[blink_counts == np.min(blink_counts)]

            if (len(ec_epochs) > 0):

                '''
                03: run entropy filter
                '''
                if remove_on_entropy:
                    ## remove epochs that have low entropy channels
                    val_ec = []
                    for i in range(len(ec_epochs)):
                        epoch_data = np.squeeze(ec_epochs[i].get_data(picks=eeg_channel_names), axis=0)
                        ent_vals = entropy(epoch_data)
                        val_ec.append(1 if np.sum(ent_vals < 0.4) == 0 else 0)  # 0.4 is empirically chosen threshold
        
                    ec_epochs = ec_epochs[np.array(val_ec) == 1]

                    if len(ec_epochs) < min_epochs:
                        print('Not enough EC epochs found..')
                        return []

                '''
                04: run occipital alpha sort only if enough EC epochs remain
                '''
                psds, freqs = ec_epochs.compute_psd('welch', fmin=8., fmax=13.).get_data(return_freqs=True)
                tot_alpha_power = np.sum(psds, axis=-1, keepdims=False)
                alpha_occ = np.reshape(tot_alpha_power[:, ec_epochs.info['ch_names'].index('o1')] \
                                       + tot_alpha_power[:, ec_epochs.info['ch_names'].index('o2')], [-1])
        
                index_array = np.argsort(alpha_occ)
                ec_epochs = ec_epochs[index_array]
        
                return ec_epochs[-max_epochs:]
                
            else:
                print('Not enough EC epochs found..')
                return []
                
        else:
            print('Not enough EC epochs found..')
            return []
            