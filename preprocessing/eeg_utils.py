# def get_standard_channel_lists():
#     """Return standard EEG channel lists for different naming conventions."""
    
#     # Standard 10-20 channel names (e.g., used by MNE montage)
#     standard_1020 = [
#         'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
#         'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
#         'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2'
#     ]
    
#     # TUH EEG channel names with REF
#     tuh_ref_channels = [
#         'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
#         'EEG T1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG T2-REF',
#         'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG OZ-REF', 'EEG O2-REF'
#     ]
    
#     # TUH reduced channel set (19 channels)
#     tuh_reduced_channels = [
#         'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
#         'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
#         'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF'
#     ]
    
#     return {
#         'standard_1020': standard_1020,
#         'tuh_ref': tuh_ref_channels,
#         'tuh_reduced': tuh_reduced_channels
#     }

# def map_tuh_to_standard_channels(raw):
#     """
#     Map TUH EEG channel names to standard 10-20 system names.
    
#     Args:
#         raw: MNE Raw object with TUH channel names
        
#     Returns:
#         raw: MNE Raw object with renamed channels
#     """
#     # Create mapping from TUH to standard format
#     mapping = {}
#     for ch in raw.ch_names:
#         if 'EEG ' in ch and '-REF' in ch:
#             # Extract the electrode name (e.g., FP1 from EEG FP1-REF)
#             electrode = ch.replace('EEG ', '').replace('-REF', '')
            
#             # Handle specific case differences
#             if electrode.upper() == 'FP1':
#                 mapping[ch] = 'Fp1'
#             elif electrode.upper() == 'FP2':
#                 mapping[ch] = 'Fp2'
#             elif electrode.upper() == 'FZ':
#                 mapping[ch] = 'Fz'
#             elif electrode.upper() == 'CZ':
#                 mapping[ch] = 'Cz'
#             elif electrode.upper() == 'PZ':
#                 mapping[ch] = 'Pz'
#             elif electrode.upper() == 'OZ':
#                 mapping[ch] = 'Oz'
#             else:
#                 # For other channels, just use the electrode part
#                 mapping[ch] = electrode
    
#     # Rename the channels
#     raw.rename_channels(mapping)
    
#     return raw


def get_standard_channel_lists():
    """Return standardized EEG channel lists for different naming conventions."""
    
    # Standard 10-20 channel names (short form)
    short_ch_names = [
        'A1', 'A2',
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    # TUH EEG channel names with REF
    ar_ch_names = [
        'EEG A1-REF', 'EEG A2-REF',
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ]
    
    # TUH EEG channel names with LE
    le_ch_names = [
        'EEG A1-LE', 'EEG A2-LE',
        'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
        'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
        'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
        'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
    ]
    
    # TUH reduced channel set (19 channels)
    tuh_reduced_channels = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
        'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
        'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF'
    ]
    
    # Create mappings
    ar_ch_mapping = {ch_name: short_name for ch_name, short_name in zip(ar_ch_names, short_ch_names)}
    le_ch_mapping = {ch_name: short_name for ch_name, short_name in zip(le_ch_names, short_ch_names)}
    
    return {
        'short': short_ch_names,
        'ar_names': ar_ch_names,
        'le_names': le_ch_names,
        'ar_mapping': ar_ch_mapping,
        'le_mapping': le_ch_mapping,
        'tuh_reduced': tuh_reduced_channels
    }

def detect_reference_system(raw):
    """
    Detect which reference system (REF or LE) is used in the raw EEG data.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        str: 'ar' for REF system, 'le' for LE system
    """
    # Default to AR/REF system
    ref_system = 'ar'
    
    # Check for any channel with reference indicator
    for ch in raw.ch_names:
        if 'EEG' in ch:
            if '-REF' in ch:
                ref_system = 'ar'
                break
            elif '-LE' in ch:
                ref_system = 'le'
                break
    
    return ref_system

def map_tuh_to_standard_channels(raw):
    """
    Map TUH EEG channel names to standard 10-20 system names.
    Handles both -REF and -LE naming conventions.
    
    Args:
        raw: MNE Raw object with TUH channel names
        
    Returns:
        raw: MNE Raw object with renamed channels
    """
    # Get channel mappings
    channel_lists = get_standard_channel_lists()
    ar_mapping = channel_lists['ar_mapping']
    le_mapping = channel_lists['le_mapping']
    
    # Use the utility function to detect reference system
    ref_system = detect_reference_system(raw)
    
    # Select the appropriate mapping
    mapping = ar_mapping if ref_system == 'ar' else le_mapping
    # print(f"Using {ref_system.upper()} reference system")
    
    # Create channel mapping for this recording
    final_mapping = {}
    for ch in raw.ch_names:
        # If channel is in our standard mapping
        if ch in mapping:
            final_mapping[ch] = mapping[ch]
        # For non-standard channel names
        elif 'EEG ' in ch:
            # Extract electrode name
            if '-REF' in ch:
                electrode = ch.replace('EEG ', '').replace('-REF', '')
            elif '-LE' in ch:
                electrode = ch.replace('EEG ', '').replace('-LE', '')
            else:
                electrode = ch.replace('EEG ', '')
            
            # Fix common case issues
            if electrode.upper() == 'FP1':
                final_mapping[ch] = 'Fp1'
            elif electrode.upper() == 'FP2':
                final_mapping[ch] = 'Fp2'
            elif electrode.upper() == 'FZ':
                final_mapping[ch] = 'Fz'
            elif electrode.upper() == 'CZ':
                final_mapping[ch] = 'Cz'
            elif electrode.upper() == 'PZ':
                final_mapping[ch] = 'Pz'
            elif electrode.upper() == 'OZ':
                final_mapping[ch] = 'Oz'
            else:
                final_mapping[ch] = electrode
    
    # # Print the mapping
    # print(f"Channel mapping: {final_mapping}")
    
    # Rename the channels
    raw.rename_channels(final_mapping)
    
    return raw
