import os
import shutil
import numpy as np

# Define patch lengths and checkpoints
patch_lengths = ["patch_len_1", "patch_len_10", "patch_len_100"]
checkpoints = [f"checkpoint_{str(i).zfill(3)}" for i in range(10, 101, 10)]

# Filter out the test split embeddings
patients_split_path = "/mnt/ssd_4tb_0/data/tuhab_preprocessed/tuh_patient_splits.pkl"
patients_split = np.load(patients_split_path, allow_pickle=True)
patients_split = patients_split['test']

for patch_len in patch_lengths:
    for checkpoint in checkpoints:
        embeddings_dir = f"/mnt/ssd_4tb_0/data/embeddings/{patch_len}/{checkpoint}"
        test_embeddings_dir = f"/mnt/ssd_4tb_0/data/test_embeddings/{patch_len}/{checkpoint}"
        
        # Ensure the embeddings directory exists
        if not os.path.exists(embeddings_dir):
            print(f"Skipping {embeddings_dir}, directory does not exist.")
            continue
        
        # Get test embeddings files
        test_embeddings_files = [
            os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir)
            if f.endswith(".npy") and f.split("_")[0] in patients_split
        ]
        
        # Copy the test embeddings to the new directory
        os.makedirs(test_embeddings_dir, exist_ok=True)
        for f in test_embeddings_files:
            shutil.copy(f, test_embeddings_dir)

        print(f"Copied {len(test_embeddings_files)} test embeddings from {embeddings_dir} to {test_embeddings_dir}.")

