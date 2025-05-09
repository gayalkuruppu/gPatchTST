from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.io import read_image
import torch
import os
import numpy as np
import pandas as pd
import random
import pickle

# [
#         'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
#         'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
#         'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF'
#     ]

class TUH_Dataset(Dataset):
    def __init__(self, root_path, data_path, csv_path, size=None,
                 split='train', train_split=0.7, test_split=0.2, seed=42):
        # Initialize parameters
        self.root_path = root_path
        self.data_path = data_path
        self.csv_path = csv_path
        self.seed = seed
        
        # Split settings
        assert split in ['train', 'test', 'val']
        self.split = split
        self.train_split = train_split
        self.test_split = test_split
        
        # Sequence parameters
        if size is None:
            raise ValueError("Must specify size parameter")
        else:
            self.seq_len = size[0]
            self.label_len = size[1] # Not used
            self.pred_len = size[2]
        
        # Path for saving/loading patient splits with seed
        self.splits_path = os.path.join(self.root_path, f"tuh_patient_splits_seed_{self.seed}.pkl")
        
        # Load file lengths from CSV
        self.file_lengths = self._load_file_lengths()
        
        # Load and process the data
        self.__read_data__()
        
        # Binary search optimization - precompute file boundaries
        self.file_boundaries = [(0, cum_len) if i == 0 else 
                               (self.cumulative_lengths[i-1], cum_len) 
                               for i, cum_len in enumerate(self.cumulative_lengths)]
        
    def _load_file_lengths(self):
        df = pd.read_csv(self.csv_path)
        file_lengths = {row['filename']: row['time_len'] for _, row in df.iterrows()}
        return file_lengths

    def __read_data__(self):
        # Get list of all tensor files
        tensor_dir = os.path.join(self.root_path, self.data_path)
        recordings = sorted([f for f in os.listdir(tensor_dir) if f.endswith('.npy')])
        
        # Group files by patient ID
        patient_files_dic = {} # dic to hold patient files with patient id as key
        for file in recordings:
            # Extract patient ID from filename (e.g., 'aaaaauon_s001_t001_preprocessed.pt')
            patient_id = file.split('_')[0]
            if patient_id not in patient_files_dic:
                patient_files_dic[patient_id] = []
            patient_files_dic[patient_id].append(file)
        print(f"Found {len(patient_files_dic)} patients with {len(recordings)} files")
        
        # Get patient splits
        try:
            # Try to load existing splits
            with open(self.splits_path, 'rb') as f:
                splits = pickle.load(f)
                train_patients = splits['train']
                val_patients = splits['val']
                test_patients = splits['test']
                print(f"Loaded existing patient splits from {self.splits_path}")
        except (FileNotFoundError, EOFError):
            # Create new splits if file doesn't exist
            print(f"Creating new patient splits...")
            
            # Set random seed for reproducibility
            random.seed(self.seed)
            
            # Shuffle patient IDs
            patient_ids = list(patient_files_dic.keys())
            random.shuffle(patient_ids)
            
            # Split patients
            num_patients = len(patient_ids)
            train_end = int(num_patients * self.train_split)
            val_end = train_end + int(num_patients * (1 - self.train_split - self.test_split))
            
            train_patients = patient_ids[:train_end]
            val_patients = patient_ids[train_end:val_end]
            test_patients = patient_ids[val_end:]
            
            # Save splits for future use
            splits = {'train': train_patients, 'val': val_patients, 'test': test_patients}
            with open(self.splits_path, 'wb') as f:
                pickle.dump(splits, f)
            print(f"Saved patient splits to {self.splits_path}")
        
        # Select appropriate files based on split
        if self.split == 'train':
            selected_patients = train_patients
        elif self.split == 'val':
            selected_patients = val_patients
        else:  # test
            selected_patients = test_patients
            
        # Get files for selected patients
        self.selected_files = []
        # print(f"Selected patients: {selected_patients}")
        for patient in selected_patients:
            self.selected_files.extend(patient_files_dic.get(patient, []))
        # print(f"Selected files: {self.selected_files}")
            
        print(f"Split: {self.split}, Patients: {len(selected_patients)}, Files: {len(self.selected_files)}")
        
        # Calculate cumulative sequence lengths for indexing
        self.cumulative_lengths = []
        total_sequences = 0
        for file in self.selected_files:
            file_length = self.file_lengths[file]
            num_sequences = file_length // (self.seq_len + self.pred_len) # max(0, file_length - self.seq_len - self.pred_len + 1)
            self.cumulative_lengths.append(total_sequences + num_sequences)
            total_sequences += num_sequences
        # print("self.cumulative_lengths:", self.cumulative_lengths)
        print(f"Total Sequences: {total_sequences}")
    
    def _load_tensor(self, file_idx):
        """Load tensor without caching"""
        file_name = self.selected_files[file_idx]
        
        # Load the file directly
        file_path = os.path.join(self.root_path, self.data_path, file_name)
        np_array = np.load(file_path, mmap_mode='r')
        
        # Ensure np_arr is 2D [time, channels]
        if np_array.ndim == 1:
            np_array = np_array[:, np.newaxis]
            
        return np_array

    def __getitem__(self, index):
        # Binary search to find file index more efficiently
        left, right = 0, len(self.cumulative_lengths) - 1
        while left <= right:
            mid = (left + right) // 2
            if index < self.cumulative_lengths[mid]:
                if mid == 0 or index >= self.cumulative_lengths[mid-1]:
                    file_index = mid
                    break
                right = mid - 1
            else:
                left = mid + 1
        
        # Calculate local index in file
        if file_index > 0:
            local_index = index - self.cumulative_lengths[file_index - 1]
        else:
            local_index = index
        
        # Load tensor directly without caching
        tensor = self._load_tensor(file_index)
        # print(f"Loading tensor from {self.selected_files[file_index]} with shape {tensor.shape}", index, local_index)
        
        # Extract the sequence
        seq_x = tensor[local_index*self.seq_len:(local_index+1)*self.seq_len]
        seq_y = tensor[(local_index+1)*self.seq_len:(local_index+1)*self.seq_len + self.pred_len]
        
        # return seq_x, seq_y
        return {"past_values": seq_x.copy(), "future_values": seq_y.copy()}
        
    def __len__(self):
        return self.cumulative_lengths[-1]


class TUAB_Dataset(TUH_Dataset):
    def __init__(self, root_path, data_path, csv_path, metadata_csv_path=None,
                 size=None, split='train', train_split=0.7, test_split=0.2, seed=42):
        super().__init__(root_path, data_path, csv_path, size=size,
                         split=split, train_split=train_split, test_split=test_split, seed=seed)
        
        # Load metadata
        self.metadata = self._read_metadata(metadata_csv_path)

    def _read_metadata(self, metadata_csv_path):
        """Read metadata from CSV file and return as DataFrame"""
        if metadata_csv_path:
            metadata = pd.read_csv(metadata_csv_path)
        else:
            metadata = pd.read_csv(os.path.join(self.root_path, self.data_path, 'metadata.csv'))

        metadata.set_index('filename', inplace=True)
        # Ensure the index is unique
        assert metadata.index.is_unique, "Index in metadata CSV must be unique"
        return metadata

    def __getitem__(self, index):
        # Binary search to find file index more efficiently
        left, right = 0, len(self.cumulative_lengths) - 1
        while left <= right:
            mid = (left + right) // 2
            if index < self.cumulative_lengths[mid]:
                if mid == 0 or index >= self.cumulative_lengths[mid-1]:
                    file_index = mid
                    break
                right = mid - 1
            else:
                left = mid + 1
        
        # Calculate local index in file
        if file_index > 0:
            local_index = index - self.cumulative_lengths[file_index - 1]
        else:
            local_index = index
        
        # Load tensor directly without caching
        tensor = self._load_tensor(file_index)

        # Extract the sequence
        seq_x = tensor[local_index*self.seq_len:(local_index+1)*self.seq_len]
        seq_y = tensor[(local_index+1)*self.seq_len:(local_index+1)*self.seq_len + self.pred_len]
        
        filename = self.selected_files[file_index].split('_preprocessed.npy')[0]
        metadata = self.metadata.loc[filename]


        data = {"past_values": seq_x.copy(), 
                "future_values": seq_y.copy(),
                "age": metadata["age"],
                "gender": metadata["gender"],
                "label": torch.tensor(0 if metadata["label"] == "normal" else 1, dtype=torch.long),
                "filename": filename,
                "patient_name": metadata["patient_name"]
                }
        
        return data
    
class TUAB_Dataset_ALPHA(TUH_Dataset):
    def __init__(self, root_path, data_path, csv_path, alpha_dict_path=None,
                 size=None, split='train', train_split=0.7, test_split=0.2, seed=42):
        super().__init__(root_path, data_path, csv_path, size=size, 
                         split=split, train_split=train_split, test_split=test_split, seed=seed)
        
        self.alpha_dict = self._read_alpha_dict(alpha_dict_path)

    def _read_alpha_dict(self, alpha_dict_path):
        """Read alpha power dictionary from pickle file"""
        if alpha_dict_path:
            with open(alpha_dict_path, 'rb') as f:
                alpha_dict = pickle.load(f)
        else:
            raise ValueError("Alpha dictionary path must be provided")
        
        return alpha_dict
    
    def __getitem__(self, index):
        # Binary search to find file index more efficiently
        left, right = 0, len(self.cumulative_lengths) - 1
        while left <= right:
            mid = (left + right) // 2
            if index < self.cumulative_lengths[mid]:
                if mid == 0 or index >= self.cumulative_lengths[mid-1]:
                    file_index = mid
                    break
                right = mid - 1
            else:
                left = mid + 1
        
        # Calculate local index in file
        if file_index > 0:
            local_index = index - self.cumulative_lengths[file_index - 1]
        else:
            local_index = index
        
        # Load tensor directly without caching
        tensor = self._load_tensor(file_index)

        # Extract the sequence
        seq_x = tensor[local_index*self.seq_len:(local_index+1)*self.seq_len]
        seq_y = tensor[(local_index+1)*self.seq_len:(local_index+1)*self.seq_len + self.pred_len]
        
        filename = self.selected_files[file_index].split('_preprocessed.npy')[0]
        alpha_powers_arr = self.alpha_dict[filename] # (n_epochs, n_channels)

        # get the relative epoch
        epoch_alpha_power = np.array(alpha_powers_arr)[local_index] # (n_channels,)

        alpha_power = torch.tensor(epoch_alpha_power, dtype=torch.float32)
        
        data = {"past_values": seq_x.copy(), 
                "future_values": seq_y.copy(),
                "label": alpha_power,
                }

        return data

# class MPI_LEMON_ALPHA()
import os
import random
import pickle
from collections import defaultdict

def create_splits(data_dir, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if type(data_dir)==list:
        splits_path = os.path.join(data_dir[0], f"patient_splits_seed_{seed}.pkl")
    else:
        splits_path = os.path.join(data_dir, f"patient_splits_seed_{seed}.pkl")
    
    if not os.path.exists(splits_path):
        print(f"Creating new patient splits...")
        random.seed(seed)

        # Group files by patient
        file_groups = defaultdict(list)
        if type(data_dir)==str:
            data_dir = [data_dir]
        for dir in data_dir:
            for fname in os.listdir(dir):
                if fname.endswith('.npy'):
                    patient_key = fname.split('_s')[0]  # 'aaaaapto'
                    file_groups[patient_key].append(fname)

        # Sort patient keys for reproducibility
        all_patients = sorted(file_groups.keys())
        random.shuffle(all_patients)

        n_total = len(all_patients)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        train_patients = all_patients[:n_train]
        val_patients = all_patients[n_train:n_train+n_val]
        test_patients = all_patients[n_train+n_val:]

        splits = {'train': train_patients, 'val': val_patients, 'test': test_patients}
        
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"Saved patient splits to {splits_path}")
    else:
        print(f"Patient splits already exist at {splits_path}")

def load_split(data_dir, seed=42, split="train"):
    if type(data_dir)==list:
        splits_path = os.path.join(data_dir[0], f"patient_splits_seed_{seed}.pkl")
    else:
        splits_path = os.path.join(data_dir, f"patient_splits_seed_{seed}.pkl")
    
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Split file {splits_path} not found. Please create splits first.")

    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    if split not in splits:
        raise ValueError(f"Split '{split}' not found in saved splits.")
    
    selected_patients = splits[split]
    return selected_patients


class TUH_Scalograms_Dataset(Dataset):
    def __init__(self, data_dir, split="train", seed=42):
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.single_data_dir = type(data_dir) == str

        # Make sure splits exist
        create_splits(self.data_dir, seed=self.seed)

        # Load only the requested split
        selected_patients = load_split(self.data_dir, seed=self.seed, split=self.split)

        # Find all files for selected patients
        all_files = []
        if type(self.data_dir)==list:
            for dir in self.data_dir:
                all_files += os.listdir(dir)
        else:
            all_files = os.listdir(self.data_dir)

        self.file_list = [
            fname for fname in all_files
            if fname.endswith('.npy') and any(fname.startswith(patient) for patient in selected_patients)
        ]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if self.single_data_dir:
            file_path = os.path.join(self.data_dir, self.file_list[idx])
        else:
            file_path = os.path.join(self.data_dir[0], self.file_list[idx])
            if not os.path.exists(file_path):
                file_path = os.path.join(self.data_dir[1], self.file_list[idx])

        data = np.load(file_path)

        return {"filename": self.file_list[idx], "data": data}


import torch
from torchvision import datasets, transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Converts HWC [0–255] to CHW [0–1]
# ])

import os
from PIL import Image
import torchvision.transforms as transforms
import random
from sklearn.model_selection import train_test_split

class CelebAFolderDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_celeba_dataloaders(root_path, batch_size=64, num_workers=4, seed=42, 
                           prefetch_factor=1, pin_memory=False, drop_last=False):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(178),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    all_images = sorted([
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    random.seed(seed)
    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed)

    train_ds = CelebAFolderDataset(train_imgs, transform)
    val_ds = CelebAFolderDataset(val_imgs, transform)
    test_ds = CelebAFolderDataset(test_imgs, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, prefetch_factor=prefetch_factor,
                              pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, prefetch_factor=prefetch_factor, 
                            pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)

    return train_loader, val_loader, test_loader

# # Get a batch and check shape
# images, _ = next(iter(celeba_loader))
# print(images.shape)  # [64, 3, 218, 178]


def get_tuh_dataloaders(root_path, data_path, csv_path, batch_size=64, num_workers=4, 
                   prefetch_factor=1, pin_memory=False, drop_last=False, size=None, seed=42):
    # Create datasets
    train_dataset = TUH_Dataset(root_path, data_path, csv_path, size=size, split='train', seed=seed)
    val_dataset = TUH_Dataset(root_path, data_path, csv_path, size=size, split='val', seed=seed)
    test_dataset = TUH_Dataset(root_path, data_path, csv_path, size=size, split='test', seed=seed)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)
    
    return train_loader, val_loader, test_loader


def get_tuab_dataloaders(root_path, data_path, csv_path, metadata_csv_path=None,
                        batch_size=64, num_workers=4, prefetch_factor=1, 
                        pin_memory=False, drop_last=False, size=None, seed=42):
    # Create datasets
    train_dataset = TUAB_Dataset(root_path, data_path, csv_path, metadata_csv_path=metadata_csv_path,
                                 size=size, split='train', seed=seed)
    val_dataset = TUAB_Dataset(root_path, data_path, csv_path, metadata_csv_path=metadata_csv_path,
                               size=size, split='val', seed=seed)
    test_dataset = TUAB_Dataset(root_path, data_path, csv_path, metadata_csv_path=metadata_csv_path,
                                size=size, split='test', seed=seed)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor,
                              pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)
    
    return train_loader, val_loader, test_loader


def get_tuab_alpha_dataloaders(root_path, data_path, csv_path, alpha_dict_path=None,
                            batch_size=64, num_workers=4, prefetch_factor=1, 
                            pin_memory=False, drop_last=False, size=None, seed=42):
    # Create datasets
    train_dataset = TUAB_Dataset_ALPHA(root_path, data_path, csv_path, alpha_dict_path=alpha_dict_path,
                                       size=size, split='train', seed=seed)
    val_dataset = TUAB_Dataset_ALPHA(root_path, data_path, csv_path, alpha_dict_path=alpha_dict_path,
                                     size=size, split='val', seed=seed)
    test_dataset = TUAB_Dataset_ALPHA(root_path, data_path, csv_path, alpha_dict_path=alpha_dict_path,
                                      size=size, split='test', seed=seed)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor,
                              pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)
    
    return train_loader, val_loader, test_loader


def get_tuab_scalogram_dls(data_dir, batch_size=64, num_workers=4, prefetch_factor=1, 
                            pin_memory=False, drop_last=False, seed=42):
    # Create datasets
    train_dataset = TUH_Scalograms_Dataset(data_dir, split='train', seed=seed)
    val_dataset = TUH_Scalograms_Dataset(data_dir, split='val', seed=seed)
    test_dataset = TUH_Scalograms_Dataset(data_dir, split='test', seed=seed)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor,
                              pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                             pin_memory=pin_memory, drop_last=drop_last)
    
    return train_loader, val_loader, test_loader
