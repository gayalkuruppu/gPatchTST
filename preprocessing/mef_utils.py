import os
from datetime import datetime, timezone

import mne
from pymef.mef_session import MefSession

import pandas as pd
import numpy as np

# MEF channel names (applies to Mayo EEGs)
_eeg_channels = ['fp1', 'f3', 'f7', 'f9', 'c3', 't7', 't9', 'tp11', 'p3', 'p7', 'p9', 'o1',
                 'fp2', 'f4', 'f8', 'f10', 'c4', 't8', 't10', 'tp12', 'p4', 'p8', 'p10', 'o2',
                 'fpz', 'fz', 'cz', 'pz', 'oz']
_misc_channels = ['ecg', 'hr', 'o2sat', 'eye-1', 'eye-2', 'emg-1', 'emg-2', 'resp-1', 'resp-2',
                  'chin-1', 'chin-2', 'chin-3', 'loc', 'roc']   # the channels in line 2 are specific to PSGs.
_used_channels = _eeg_channels + _misc_channels

# when old 10-20 channel names are used
_old_name_map = {'t3': 't7', 't4': 't8', 't5': 'p7', 't6': 'p8'}

# when the converter chooses the wrong montage for headbox type 17 (46 channels)
_wrong_montage_map = {'resp1': 't9', 'resp2': 't10', 'le': 'nz', 're': 'iz', 'chin1': 'f9', 'chin2': 'f10',
                      'x11': 'gnd', 'x12': 'ref', 'x13': 'eye-1', 'x14': 'eye-2', 'x15': 'emg-1', 'x16': 'emg-2',
                      'x17': 'resp-1', 'x18': 'resp-2'}

_ch_types = ['eeg'] * len(_eeg_channels) + ['misc'] * len(_misc_channels)
_SAMP_FREQ = 256.0
_UNITS = 1e-6


def mef_to_mne(mef_path, password='', sample_rate=_SAMP_FREQ, return_chan_info=False):
    '''
    Convert mef object to MNE raw object.
    :param mef_path: Path to a mef file.
    :param password: MEF password
    :param sample_rate: required sample rate (EEG will be resampled to this rate)
    :return raw: MNE RAW object.
    '''

    if not mef_path.endswith('/'):
        mef_path = mef_path + '/'

    if os.path.isfile(os.path.join(mef_path, 'conversion_failed.txt')):
        print("Corrupt data")
        return None

    if not any(fname.endswith('.rdat') for fname in os.listdir(mef_path)):
        print("Corrupt data")
        return None

    try:
        ms = MefSession(mef_path, password, check_all_passwords=False)
    except:
        print("An exception occurred")
        return None

    chan_info = ms.read_ts_channel_basic_info()

    if return_chan_info:
        return chan_info

    all_ch_data = []
    chan_names = []

    min_len = np.min(np.array([chan['nsamp'] for chan in chan_info]))
    min_ch_name = chan_info[np.argmin(np.array([chan['nsamp'] for chan in chan_info]))]['name']
    toc = ms.get_channel_toc(min_ch_name)

    for chan in chan_info:
        chan_name = chan['name']
        chan_fs = chan['fsamp']

        # convert channel name to lower case to be consistent.
        new_chan_name = chan_name.lower()

        if new_chan_name in _old_name_map:
            new_chan_name = _old_name_map[new_chan_name]

        if (new_chan_name in _wrong_montage_map) and (len(chan_info) == 46):
            new_chan_name = _wrong_montage_map[new_chan_name]

        if new_chan_name.lower() in _used_channels:
            ch_data = ms.read_ts_channels_sample(chan_name, [[0, min_len]]) # use the original chan_name to read data

            if new_chan_name in _eeg_channels:
                mne_info = mne.create_info([new_chan_name], ch_types=['eeg'], sfreq=chan_fs)
            else:
                mne_info = mne.create_info([new_chan_name], ch_types=['misc'], sfreq=chan_fs)

            raw = mne.io.RawArray(np.reshape(ch_data, [1, ch_data.shape[0]]) * _UNITS, mne_info)

            if (chan_fs != sample_rate):
                raw = raw.resample(sfreq=sample_rate)

            all_ch_data.append(raw.get_data())
            chan_names.append(new_chan_name)

    all_ch_data = np.squeeze(all_ch_data)

    available_channels = []
    available_ch_types = []
    ch_order = []
    for ch, ch_type in zip(_used_channels, _ch_types):
        if ch in chan_names:
            available_channels.append(ch)
            available_ch_types.append(ch_type)
            ch_order.append(chan_names.index(ch))
    reduced_data = all_ch_data[ch_order, :]

    # create MNE object
    mne_info = mne.create_info(available_channels, ch_types=available_ch_types, sfreq=sample_rate)
    dt = datetime.fromtimestamp(chan_info[0]['start_time'][0] / 1000000, tz=timezone.utc)
    raw = mne.io.RawArray(reduced_data, mne_info)
    raw = raw.set_meas_date(dt)

    try:
        # extract annotations and offset them using discont information
        events = pd.read_csv(mef_path + 'events.csv', engine='python', names=['Timestamp', 'Event'])
    except:
        print("An error occurred while reading events.csv in %s." % mef_path)
        return raw

    onsets = []
    for ev_ts in events['Timestamp'].values:
        if toc[3, toc[3, :] < ev_ts].size == 0:
            nearest_ts = chan_info[0]['start_time'][0]
            nearest_samp = 0
        else:
            nearest_ts = toc[3, toc[3, :] < ev_ts].max()
            nearest_samp = toc[2, toc[3, :] < ev_ts].max()
        ons = (nearest_samp / sample_rate) + (ev_ts - nearest_ts) / 1000000
        onsets.append(ons)

    durations = np.repeat(10., len(events))
    event_names = [str(event).replace('(Created by Persyst)', '').strip().lower() for event in events['Event'].values]
    annotations = mne.Annotations(onsets, durations, event_names)
    raw.set_annotations(annotations)

    return raw