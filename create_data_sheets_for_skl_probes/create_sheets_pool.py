import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from mapping_utils import spatial_mapping, left_right_mapping, channel_list, simplified_channel_list

patch_lengths = ["patch_len_1", "patch_len_10", "patch_len_100"]
powers_dict_paths = {
    "patch_len_1": 'preprocessing/outputs/powers_dict_patch_len_1_seq_len_2_secs.pkl',
    "patch_len_10": 'preprocessing/outputs/powers_dict_patch_len_10_seq_len_10_secs.pkl',
    "patch_len_100": 'preprocessing/outputs/powers_dict_patch_len_100_seq_len_10_secs.pkl'
}
checkpoints = [f"checkpoint_{str(i).zfill(3)}" for i in range(10, 101, 10)]
# alpha_powers_path = 'preprocessing/outputs/alpha_power_dict_10_secs.pkl'
recording_metadata_path = 'create_data_sheets_for_skl_probes/tuhab_patient_metadata.csv'

recording_metadata_df = pd.read_csv(recording_metadata_path)

# Precompute channel mappings
channel_mappings = {
    ch: (simplified_channel_list[ch], spatial_mapping(channel_list[ch]), left_right_mapping(channel_list[ch]))
    for ch in range(len(channel_list))
}

# Convert metadata DataFrame to a dictionary for faster lookups
metadata_dict = recording_metadata_df.set_index('recording_name').to_dict(orient='index')

def generate_data_sheet_for_checkpoint(powers_dict_paths, channel_mappings, metadata_dict, patch_len, checkpoint):
    powers_dict_path = powers_dict_paths[patch_len]
    seq_len = int(powers_dict_path.split('_')[-2])

    power_bands_dict = np.load(powers_dict_path, allow_pickle=True)
        
    test_embeddings_dir = f"/mnt/ssd_4tb_0/data/test_embeddings/{patch_len}/{checkpoint}"
    test_embeddings_files = [
            os.path.join(test_embeddings_dir, f) for f in os.listdir(test_embeddings_dir)
            if f.endswith(".npy")
        ]

    test_embeddings_files.sort()  # 19, 128

    test_embeddings = []
    data_dict = {
            "recordings": [],
            "subject_id": [],
            "alpha_power": [],
            "delta_power": [],
            "theta_power": [],
            "beta_power": [],
            "gamma_power": [],
            "age": [],
            "gender": [],
            "abnormal": [],
            "channel": [],
            "brain_region": [],
            "brain_hemisphere": [],
        }

    for f in test_embeddings_files:  # f-> aaaaamhb_s001_t000_id_084.npy
        data = np.load(f)  # f shape: (19, 128)
        sample_name = os.path.basename(f).split(".")[0]
        power_bands = np.array(power_bands_dict[sample_name]).squeeze(1)  # (5, 19)

            # Extract the subject ID and recording ID
        subject_id = os.path.basename(f).split("_")[0]
        recording_id = os.path.basename(f).split("_id_")[0]

            # Get metadata for the recording
        metadata_row = metadata_dict.get(recording_id, None)
        if metadata_row is None:
            print(f"Warning: No metadata found for recording {recording_id}.")
            continue

        for ch in range(data.shape[0]):
                # Add data to the test_embeddings list
            test_embeddings.append(data[ch])

                # Append data to the dictionary
            data_dict["recordings"].append(sample_name)
            data_dict["subject_id"].append(subject_id)
            ap, dp, tp, bp, gp = power_bands[:, ch]  # (5, 19) -> (5,)
            data_dict["alpha_power"].append(ap)
            data_dict["delta_power"].append(dp)
            data_dict["theta_power"].append(tp)
            data_dict["beta_power"].append(bp)
            data_dict["gamma_power"].append(gp)
            data_dict["age"].append(metadata_row['age'])
            data_dict["gender"].append(metadata_row['gender'])
            data_dict["abnormal"].append(0 if metadata_row['label'] == 'normal' else 1)

                # Use precomputed mappings
            simplified_ch_name, brain_region, brain_hemisphere = channel_mappings[ch]
            data_dict["channel"].append(simplified_ch_name)
            data_dict["brain_region"].append(brain_region)
            data_dict["brain_hemisphere"].append(brain_hemisphere)

        # Create a DataFrame with all the collected data
    df = pd.DataFrame(data_dict)

    test_embeddings = np.array(test_embeddings)

        # Save the DataFrame to a CSV file
    output_dir = f"/mnt/ssd_4tb_0/data/LP_sheets_patchtst"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"data_sheet_{patch_len}_seq_len_{seq_len}_{checkpoint}.csv")
    df.to_csv(output_file, index=False)

        # Save the test embeddings to a .npy file
    test_embeddings_file = os.path.join(output_dir, f"data_{patch_len}_seq_len_{seq_len}_{checkpoint}.npy")
    np.save(test_embeddings_file, test_embeddings)

    print(f"Data sheet saved to {output_file}")


def main():
    tasks = [
        (powers_dict_paths, channel_mappings, metadata_dict, patch_len, checkpoint)
        for patch_len in patch_lengths
        for checkpoint in checkpoints
    ]

    with Pool() as pool:
        pool.starmap(generate_data_sheet_for_checkpoint, tasks)

if __name__ == "__main__":
    main()