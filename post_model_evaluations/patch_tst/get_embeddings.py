import sys
sys.path.append('/home/gayal/ssl-project/gpatchTST')
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from data import get_tuh_dataloaders, get_tuh_dataloaders_old_splits
from configs import Config
from get_models import get_patchTST_model
from models.patchtst.layers.revin import RevIN
import os
from utils.mask_utils import create_patches
import random
import numpy as np
import skdim
import pickle
import logging

from scipy.spatial import KDTree
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
                            logging.StreamHandler(sys.stdout),
                            logging.FileHandler('post_model_evaluations/logs/get_embeddings.log')
                    ]
                )

BASE_PATH = '/home/gayal/ssl-project/gpatchTST'

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")

def get_embeddings_from_checkpoint(checkpoint, new_dataset_path, device):
    logging.info(f"Loading checkpoint from: {checkpoint}")
    config_file_path = [i for i in os.listdir(os.path.dirname(checkpoint)) if i.endswith('.yaml')][0]
    config_file_path = os.path.join(os.path.dirname(checkpoint), config_file_path)
    config = Config(config_file=config_file_path).get()

    data_config = config['data']
    data_config['root_path'] = new_dataset_path
    data_config['csv_path'] = os.path.join(new_dataset_path, 'file_lengths_map.csv')
    model_config = config['model']
    revin = model_config['revin']
    patch_len = model_config['patch_length']
    stride = model_config['stride']

    train_loader, val_loader, test_loader = get_tuh_dataloaders(
            data_config['root_path'],
            data_config['data_path'],
            data_config['csv_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            prefetch_factor=data_config['prefetch_factor'],
            pin_memory=data_config['pin_memory'],
            drop_last=False,
            size=[model_config['seq_len'], 
                  model_config['target_dim'],
                  model_config['patch_length']],
        )

        # Load pretrained model
    model = get_patchTST_model(num_variates=data_config['n_vars'],
                                forecast_length=model_config['target_dim'],
                                patch_len=model_config['patch_length'],
                                stride=model_config['stride'],
                                num_patch=(model_config['seq_len'] - model_config['patch_length']) // model_config['stride'] + 1,
                                n_layers=model_config['num_layers'],
                                d_model=model_config['d_model'],
                                n_heads=model_config['num_heads'],
                                shared_embedding=model_config['shared_embedding'],
                                d_ff=model_config['d_ff'],
                                norm=model_config['norm'],
                                attn_dropout=model_config['attn_dropout'],
                                dropout=model_config['dropout'],
                                activation=model_config['activation'],
                                res_attention=model_config['res_attention'],
                                pe=model_config['pe'],
                                learn_pe=model_config['learn_pe'],
                                head_dropout=model_config['head_dropout'],
                                head_type=model_config['head_type'],
                                use_cls_token=model_config['use_cls_token'],
                            ).to(device)
    
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Extracting embeddings"):
            data = data['past_values'].to(device)
            
            if revin:
                data = revin(data, mode='norm')
                target = revin(target, mode='norm')

            input_patches, _ = create_patches(data, patch_len, stride)

            output = model.backbone(input_patches) # [bs x nvars x d_model x (num_patch+1 or num_patch)]
            output = output[:, :, :, 0] # [bs x nvars x d_model]

            embeddings.append(output.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0) # [num_samples x nvars x d_model]
    return embeddings

def save_embeddings_for_data(checkpoint, new_dataset_path, save_dir, device):
    logging.info(f"Processing checkpoint: {checkpoint}")
    config_file_path = [i for i in os.listdir(os.path.dirname(checkpoint)) if i.endswith('.yaml')][0]
    config_file_path = os.path.join(os.path.dirname(checkpoint), config_file_path)
    config = Config(config_file=config_file_path).get()

    data_config = config['data']
    data_config['root_path'] = new_dataset_path
    data_config['csv_path'] = os.path.join(new_dataset_path, 'file_lengths_map.csv')
    model_config = config['model']
    revin = model_config['revin']
    patch_len = model_config['patch_length']
    stride = model_config['stride']

    train_loader, val_loader, test_loader = get_tuh_dataloaders_old_splits(
            data_config['root_path'],
            data_config['data_path'],
            data_config['csv_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            prefetch_factor=data_config['prefetch_factor'],
            pin_memory=data_config['pin_memory'],
            drop_last=False,
            size=[model_config['seq_len'], 
                  model_config['target_dim'],
                  model_config['patch_length']],
        )

        # Load pretrained model
    model = get_patchTST_model(num_variates=data_config['n_vars'],
                                forecast_length=model_config['target_dim'],
                                patch_len=model_config['patch_length'],
                                stride=model_config['stride'],
                                num_patch=(model_config['seq_len'] - model_config['patch_length']) // model_config['stride'] + 1,
                                n_layers=model_config['num_layers'],
                                d_model=model_config['d_model'],
                                n_heads=model_config['num_heads'],
                                shared_embedding=model_config['shared_embedding'],
                                d_ff=model_config['d_ff'],
                                norm=model_config['norm'],
                                attn_dropout=model_config['attn_dropout'],
                                dropout=model_config['dropout'],
                                activation=model_config['activation'],
                                res_attention=model_config['res_attention'],
                                pe=model_config['pe'],
                                learn_pe=model_config['learn_pe'],
                                head_dropout=model_config['head_dropout'],
                                head_type=model_config['head_type'],
                                use_cls_token=model_config['use_cls_token'],
                            ).to(device)
    
    model.eval()
    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            logging.info(f"Processing loader: {loader}")
            for batch in tqdm(loader, desc=f"Extracting embeddings from {loader}"):
                data = batch['past_values'].to(device)
                filename = batch['filename']
                
                if revin:
                    data = revin(data, mode='norm')

                input_patches, _ = create_patches(data, patch_len, stride)

                output = model.backbone(input_patches) # [bs x nvars x d_model x (num_patch+1 or num_patch)]
                output = output[:, :, :, 0] # [bs x nvars x d_model]

                # save embeddings
                for sample in range(output.shape[0]): # samples in batch
                    sample_embeddings = output[sample].cpu().numpy() # [nvars x d_model]
                    sample_filename = filename[sample]
                    save_path = os.path.join(save_dir, f"{sample_filename}.npy")
                    np.save(save_path, sample_embeddings)
                    
def save_embeddings(embeddings, save_path):
    """
    Save embeddings as a .npy file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)
    logging.info(f"Embeddings saved to: {save_path}")

def compute_twoNN_ratios(data):
    """
    Computes the distance ratio μ = r2 / r1 for each point in the dataset.
    """
    tree = KDTree(data)  # Use KDTree for efficient nearest neighbor search
    distances, _ = tree.query(data, k=3)  # k=3 because first neighbor is itself
    r1, r2 = distances[:, 1], distances[:, 2]  # First and second nearest neighbor distances
    mu = r2 / r1  # Compute distance ratio
    return mu

def estimate_ID(mu_values):
    """
    Estimates the intrinsic dimension by fitting log(1 - F(μ)) vs log(μ).
    """
    mu_sorted = np.sort(mu_values)
    F_empirical = np.arange(len(mu_sorted)) / len(mu_sorted)  # ECDF
    log_mu = np.log(mu_sorted)
    log_one_minus_F = np.log(1 - F_empirical)

    # Linear regression to find the slope (which estimates ID)
    slope, _, _, _, _ = linregress(log_mu, log_one_minus_F)
    estimated_d = -slope
    return estimated_d, log_mu, log_one_minus_F

def get_id_from_embeddings(embeddings, id_type='danco'):
    logging.info(f"Estimating intrinsic dimension using method: {id_type}")
    embeddings = embeddings.reshape(-1, embeddings.shape[-1]) # [num_samples*nvars x d_model]
    print(f"Number of samples: {embeddings.shape[0]}")
    print(f"Number of features: {embeddings.shape[1]}")
    if id_type == 'danco':
        id = skdim.id.DANCo().fit(embeddings).dimension_
    elif id_type == 'twoNN':
        id = skdim.id.TwoNN().fit(embeddings).dimension_
    elif id_type == 'my_TwoNN':
        mu_values = compute_twoNN_ratios(embeddings)
        id, log_mu, log_one_minus_F = estimate_ID(mu_values)
    else:
        raise ValueError(f"Unknown ID type: {id_type}")

    logging.info(f"Estimated intrinsic dimension: {id}")
    return id


def get_id_from_checkpoint(checkpoint, new_data_path, device, id_type='danco'):
    logging.info(f"Getting ID from checkpoint: {checkpoint}")
    embeddings = get_embeddings_from_checkpoint(checkpoint, new_data_path, device)
    danco_id = get_id_from_embeddings(embeddings, id_type=id_type)

    logging.info(f"Estimated ID: {danco_id}")
    return danco_id


def get_checkpoint_dirs(base_path):
    logging.info(f"Scanning for checkpoints in: {base_path}")
    checkpoint_dirs = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith('checkpoint') and file.endswith('.pth'):
                checkpoint_num = int(file.split('.pth')[0].split('_')[-1])
                checkpoint_dirs[checkpoint_num] = os.path.join(root, file)
    logging.info(f"Found {len(checkpoint_dirs)} checkpoints.")
    return checkpoint_dirs


def main():
    logging.info("Starting main process.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrained_paths = {
        # 100: 'saved_models/pretrain/tuhab_pretrain_tuab_with_cls_token/TUH-101/2025-04-17_21-01-03',
        10: 'saved_models/pretrain/tuhab_pretrain_patch_len_10/TUH-111/2025-04-18_12-55-17',
        1: 'saved_models/pretrain/tuhab_pretrain_patch_len_1_seq_200/TUH-122/2025-04-23_01-11-04',
    }

    for patch_len,  pretrained_path in pretrained_paths.items():
        logging.info(f"Processing pretrained path for patch length {patch_len}: {pretrained_path}")
        pretrained_base_path = os.path.join(BASE_PATH, pretrained_path)
        checkpoints = get_checkpoint_dirs(pretrained_base_path)
        print(f"Found {len(checkpoints)} checkpoints")

        new_data_path = '/mnt/ssd_4tb_0/data/tuhab_preprocessed'
        embeddings_save_path = os.path.join(os.path.dirname(new_data_path), 'embeddings')
        os.makedirs(embeddings_save_path, exist_ok=True)

        patch_len_save_path = os.path.join(embeddings_save_path, f'patch_len_{patch_len}')
        os.makedirs(patch_len_save_path, exist_ok=True)

        for checkpoint_num, checkpoint_path in checkpoints.items():
            logging.info(f"Processing checkpoint {checkpoint_num}: {checkpoint_path}")
            checkpoint_save_path = os.path.join(patch_len_save_path, f'checkpoint_{checkpoint_num:03d}')
            os.makedirs(checkpoint_save_path, exist_ok=True)
            save_embeddings_for_data(checkpoint_path, new_data_path, checkpoint_save_path, device)
    logging.info("Main process completed.")

if __name__ == "__main__":
    main()
