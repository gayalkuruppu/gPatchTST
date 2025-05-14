import os
import numpy as np
import pandas as pd
from mapping_utils import spatial_mapping, left_right_mapping, channel_list, simplified_channel_list

patch_lengths = ["patch_len_1", "patch_len_10", "patch_len_100"]
powers_dict_paths = {
    # "patch_len_1": 'preprocessing/outputs/powers_dict_patch_len_1_seq_len_2_secs.pkl',
    "patch_len_10": 'preprocessing/outputs/powers_dict_patch_len_10_seq_len_10_secs.pkl',
    "patch_len_100": 'preprocessing/outputs/powers_dict_patch_len_100_seq_len_10_secs.pkl'
}
checkpoints = [f"checkpoint_{str(i).zfill(3)}" for i in range(10, 101, 10)]
# alpha_powers_path = 'preprocessing/outputs/alpha_power_dict_10_secs.pkl'
recording_metadata_path = 'create_data_sheets_for_skl_probes/tuhab_patient_metadata.csv'

recording_metadata_df = pd.read_csv(recording_metadata_path)

for patch_len in patch_lengths[-1:3]:
    seq_len = None
    powers_dict_path = powers_dict_paths[patch_len]
    seq_len = int(powers_dict_path.split('_')[-2])

    # Load the power bands data
    power_bands_dict = np.load(powers_dict_path, allow_pickle=True)

    for checkpoint in checkpoints[0:1]:
        test_embeddings_dir = f"/mnt/ssd_4tb_0/data/test_embeddings/{patch_len}/{checkpoint}"
        test_embeddings_files = [
            os.path.join(test_embeddings_dir, f) for f in os.listdir(test_embeddings_dir)
            if f.endswith(".npy")
        ]

        test_embeddings_files.sort() # 19, 128

        test_embeddings = []
        sample_names = []
        recording_ids = []
        subject_ids = []
        alpha_power = []
        delta_power = []
        theta_power = []
        beta_power = []
        gamma_power = []
        age = []
        gender = []
        abnormal = []  # 0: normal, 1: abnormal
        channel = []
        brain_region = []  # lobe, frontal: 0, temporal: 1, central: 2, parietal: 3, occipital: 4
        brain_hemisphere = []  # midline: 0, left: 1, right: 2


        for f in test_embeddings_files[0:2]: # f-> aaaaamhb_s001_t000_id_084.npy
            data = np.load(f) # f shape: (19, 128)
            sample_name = os.path.basename(f).split(".")[0]
            power_bands = np.array(power_bands_dict[sample_name]).squeeze(1) # (5, 19)
            for ch in range(data.shape[0]):
                # add data to the test_embeddings list
                test_embeddings.append(data[ch])
                
                # Extract the subject ID from the filename
                subject_id = os.path.basename(f).split("_")[0]
                recording_id = os.path.basename(f).split("_id_")[0]
                
                sample_names.append(sample_name)
                recording_ids.append(recording_id)
                subject_ids.append(subject_id)

                ap, dp, tp, bp, gp = power_bands[:, ch] # (5, 19) -> (5,)
                # Get power bands for the recording
                alpha_power.append(ap)
                delta_power.append(dp)
                theta_power.append(tp)
                beta_power.append(bp)
                gamma_power.append(gp)
                
                # Get metadata for the recording
                metadata_row = recording_metadata_df[recording_metadata_df['recording_name'] == recording_id]
                if metadata_row.empty:
                    print(f"Warning: No metadata found for recording {recording_id}.")
                    continue
                age.append(metadata_row['age'].values[0])
                gender.append(metadata_row['gender'].values[0])
                abnormal.append(0 if metadata_row['label'].values[0] == 'normal' else 1)
                
                # Map the channels to brain regions and hemispheres
                simplified_ch_name = simplified_channel_list[ch]
                channel.append(simplified_ch_name)
                channel_name = channel_list[ch]
                brain_region.append(spatial_mapping(channel_name))
                brain_hemisphere.append(left_right_mapping(channel_name))
                
        # Create a DataFrame with all the collected data
        data_dict = {
            "recordings": sample_names,
            # "recording_id": recording_ids,
            "subject_id": subject_ids,
            "alpha_power": alpha_power,
            "delta_power": delta_power,
            "theta_power": theta_power,
            "beta_power": beta_power,
            "gamma_power": gamma_power,
            "age": age,
            "gender": gender,
            "abnormal": abnormal,
            "channel": channel,
            "brain_region": brain_region,
            "brain_hemisphere": brain_hemisphere,
        }
        df = pd.DataFrame(data_dict)
        
        test_embeddings = np.array(test_embeddings)

        # Save the DataFrame to a CSV file
        output_dir = f"create_data_sheets_for_skl_probes/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"data_sheet_{patch_len}_seq_len_{seq_len}_{checkpoint}.csv")
        df.to_csv(output_file, index=False)
        
        # Save the test embeddings to a .npy file
        test_embeddings_file = os.path.join(output_dir, f"data_{patch_len}_seq_len_{seq_len}_{checkpoint}.npy")
        np.save(test_embeddings_file, test_embeddings)
        
        print(f"Data sheet saved to {output_file}")
