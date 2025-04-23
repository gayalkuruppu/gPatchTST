from collections import OrderedDict
import numpy as np

from scipy.signal import welch
from scipy.integrate import simps

from mne.time_frequency import psd_array_multitaper

_BRAIN_RHYTHMS = OrderedDict({
	"delta" : [2.0, 4.0], 	# "delta" : [2.0, 4.0],
	"theta": [4.0, 8.0],
	"lower_alpha": [8.0, 10.0],
	"higher_alpha": [10.0, 13.0],
	"lower_beta": [13.0, 16.0],
	"higher_beta": [16.0, 25.0],
	"gamma": [25.0, 40.0]
}) # 10-30 if youre active/ awake , below 10 drowsy/asleep (rule of thumb)

_BETA_RANGE = [13.0, 25.0] # needed for db ratio

_BRAIN_RHYTHMS_WITH_ONE_BETA = OrderedDict({
	"delta" : [2.0, 4.0], 	# "delta" : [2.0, 4.0],
	"theta": [4.0, 8.0],
	"lower_alpha": [8.0, 10.0],
	"higher_alpha": [10.0, 13.0],
	"beta": [13.0, 25.0],
	"gamma": [25.0, 40.0]
})

# # https://raphaelvallat.com/bandpower.html
# def get_behavioral_state(window_data, use_ch, relative, ch_names, sfreq):
# 	use_ch_idx = ch_names.index(use_ch)
# 	delta_power = get_band_power(
# 		window_data[use_ch_idx, ...], sf=sfreq, band=_BRAIN_RHYTHMS['delta'],
# 		method='welch', relative=relative, db=False)
# 	beta_power = get_band_power(
# 		window_data[use_ch_idx, ...], sf=sfreq, band=_BETA_RANGE,
# 		method='welch', relative=relative, db=False)
# 	ratio = delta_power / beta_power
# 	assert ratio > 0.0
# 	return ratio

# https://raphaelvallat.com/bandpower.html
def get_behavioral_state(window_data, relative: bool, sfreq: float):
	"""Compute the average power of the signal x in a specific frequency band.
    Parameters
    ----------
    window_data : 1d-array
        Input signal in the time-domain.
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.
	sfreq : float
        Sampling frequency of the data.

	Return
	------
	ratio : float
	  Delta power and Beta power ratio.
	  """
	delta_power = get_band_power(
		window_data, sf=sfreq, band=_BRAIN_RHYTHMS['delta'],
		method='welch', relative=relative, db=False)
	beta_power = get_band_power(
		window_data, sf=sfreq, band=_BETA_RANGE,
		method='welch', relative=relative, db=False)
	ratio = delta_power / beta_power
	assert ratio > 0.0
	
	return ratio

# https://raphaelvallat.com/bandpower.html
def get_band_power(data, sf, band, method='welch', window_sec=None, relative=False, db=False):
	"""Compute the average power of the signal x in a specific frequency band.

	Requires MNE-Python >= 0.14.

	Parameters
	----------
	data : 1d-array
	  Input signal in the time-domain.
	sf : float
	  Sampling frequency of the data.
	band : list
	  Lower and upper frequencies of the band of interest.
	method : string
	  Periodogram method: 'welch' or 'multitaper'
	window_sec : float
	  Length of each window in seconds. Useful only if method == 'welch'.
	  If None, window_sec = (1 / min(band)) * 2.
	relative : boolean
	  If True, return the relative power (= divided by the total power of the signal).
	  If False (default), return the absolute power.

	Return
	------
	bp : float
	  Absolute or relative band power.
	"""

	band = np.asarray(band)
	low, high = band

	# Compute the modified periodogram (Welch)
	if method == 'welch':
		if window_sec is not None:
			nperseg = window_sec * sf
		else:
			nperseg = (2 / low) * sf
			# nperseg = 1024
		# print(f'nperseg: {nperseg}')
		freqs, psd = welch(data, sf, nperseg=nperseg)

	# CAUTION: may encounter RuntimeWarning: Iterative multi-taper PSD computation did not converge, takes longer to compute than welch
	elif method == 'multitaper':
		psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
										  normalization='full', verbose=0)

	# convert to dB scale
	if db:
		psd = 10 * np.log10(psd)

	# Frequency resolution
	freq_res = freqs[1] - freqs[0]

	# Find index of band in frequency vector
	idx_band = np.logical_and(freqs >= low, freqs <= high)

	# Integral approximation of the spectrum using parabola (Simpson's rule)
	bp = simps(psd[idx_band], dx=freq_res)

	if relative:
		bp /= simps(psd, dx=freq_res)

	return bp


def get_power_bands(window_data, sfreq, relative=False, db=False):
	"""Compute the average power of the signal x in a specific frequency band.
	Parameters
	----------
	window_data : 1d-array
		Input signal in the time-domain.
	sfreq : float
		Sampling frequency of the data.
	relative : boolean
		If True, return the relative power (= divided by the total power of the signal).
		If False (default), return the absolute power.
	db : boolean
		If True, return the power in decibels.
		If False (default), return the power in raw units (e.g., uV^2).

	Return
	------
	power_bands : dict
		Contains the power in the specified frequency bands.
	"""
	power_bands = []
	for _, freq in _BRAIN_RHYTHMS_WITH_ONE_BETA.items():
		power_band = get_band_power(
			window_data, sf=sfreq, band=freq, method='welch', relative=relative, db=db)
		power_bands.append(power_band)

	return power_bands
