def spatial_mapping(channel_name: str) -> int:
    mapping = {
            'EEG FP1-REF': 0,
            'EEG FP2-REF': 0,
            'EEG F7-REF': 0,
            'EEG F3-REF': 0,
            'EEG FZ-REF': 0,
            'EEG F4-REF': 0,
            'EEG F8-REF': 0,
            'EEG T3-REF': 1,
            'EEG C3-REF': 2,
            'EEG CZ-REF': 2,
            'EEG C4-REF': 2,
            'EEG T4-REF': 1,
            'EEG T5-REF': 1,
            'EEG P3-REF': 3,
            'EEG PZ-REF': 3,
            'EEG P4-REF': 3,
            'EEG T6-REF': 1,
            'EEG O1-REF': 4,
            'EEG O2-REF': 4,
        }

    return mapping[channel_name]


def left_right_mapping(channel_name: str) -> int:
    mapping = {
            'EEG FP1-REF': 1,
            'EEG FP2-REF': 2,
            'EEG F7-REF': 1,
            'EEG F3-REF': 1,
            'EEG FZ-REF': 0,
            'EEG F4-REF': 2,
            'EEG F8-REF': 2,
            'EEG T3-REF': 1,
            'EEG C3-REF': 1,
            'EEG CZ-REF': 0,
            'EEG C4-REF': 2,
            'EEG T4-REF': 2,
            'EEG T5-REF': 1,
            'EEG P3-REF': 1,
            'EEG PZ-REF': 0,
            'EEG P4-REF': 2,
            'EEG T6-REF': 2,
            'EEG O1-REF': 1,
            'EEG O2-REF': 2,
        }

    return mapping[channel_name]

channel_list = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
        'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
        'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF'
    ]

simplified_channel_list = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T3', 'C3', 'CZ', 'C4', 'T4',
        'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2'
    ]
