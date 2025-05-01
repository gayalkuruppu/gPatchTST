import os
import ast
import shutil
import argparse
import numpy as np
import logging
from datetime import datetime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.patchtst.layers.revin import RevIN
from einops import rearrange

from get_models import get_pretrain_model
from dataloaders.dataloaders import get_dataloaders
from configs import Config
from utils.utils import init_neptune
from utils.pretrain_utils import load_checkpoint
from utils.mask_utils import fixed_mask_inputs
# Initialize model


def plot_sample_reconstruction(model, sample, device, model_config, save_path, epoch):
    model.eval()

    data = sample['data'].to(device, non_blocking=True)[0] # first sample from the batch
    data = data.unsqueeze(0).unsqueeze(0) # [1, 1, 25, 100]

    fs_patch, ts_patch = model_config['patch_size']
    input_patches, freq_patches, time_patches = create_patches_2D(data, model_config)
    mask = create_mask_2D(input_patches, model_config)

    masked_patches = apply_mask(input_patches, mask, masked_value=0) # [bs x n_chn x freq_patches x time_patches x fs_patch x ts_patch]
    masked_patches = rearrange(masked_patches, 'b c fp tp fs ts -> b c (fp fs) (tp ts)')
    print(masked_patches.shape)
    with torch.no_grad():
        predicted_patches = model(masked_patches)
        print(f"predicted_patches shape: {predicted_patches.shape}")

    reconstruction = rearrange(predicted_patches, 'b (fp tp) (fs ts) -> b fp tp fs ts', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch).unsqueeze(1)
    masked_reconstruction = apply_inv_mask(reconstruction, mask, masked_value=0)
    masked_reconstruction = rearrange(masked_reconstruction, 'b c fp tp fs ts -> b c (fp fs) (tp ts)', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch)
    os.makedirs(save_path, exist_ok=True)

    # plot the original and reconstructed patches side by side
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].imshow(data[0, 0].cpu().numpy(), cmap='gray')
    axs[0].set_title('Original Input')
    axs[1].imshow(masked_patches[0,0].cpu().numpy(), cmap='gray')
    axs[1].set_title('Masked Input')
    axs[2].imshow(masked_reconstruction[0,0].cpu().numpy(), cmap='gray')
    axs[2].set_title('Reconstructed Output')
    # axs[3].imshow(mask[0, 0].cpu().numpy(), cmap='gray')
    # axs[3].set_title('Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_reconstruction.png'))
    plt.close(fig)

    return

def train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, train_loader, train_sample, model, revin, optimizer, scheduler, criterion, epoch):
    model.train()
    train_loss = 0
    plot_sample_reconstruction(model, train_sample, device, model_config, 
                            os.path.join(timestamped_file_name, 'train'), epoch)

    # Training step
    train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
    for batch in train_step_pbar:
        # Forward pass
        optimizer.zero_grad()
        inputs = batch['data'].to(device, non_blocking=True).unsqueeze(1)

        # masked_input, mask = fixed_mask_inputs(inputs, 
        #                                        time_mask_consecutive_min=train_config['time_mask_consecutive_min'],
        #                                        time_mask_consecutive_max=train_config['time_mask_consecutive_max'],
        #                                        freq_mask_consecutive_min=train_config['freq_mask_consecutive_min'],
        #                                        freq_mask_consecutive_max=train_config['freq_mask_consecutive_max'],
        #                                        time_mask_p=train_config['time_mask_p'],
        #                                        freq_mask_p=train_config['freq_mask_p'])

        input_patches, freq_patches, time_patches = create_patches_2D(inputs, model_config)
        fs_patch, ts_patch = model_config['patch_size']
        mask = create_mask_2D(input_patches, model_config)

        masked_patches = apply_mask(input_patches, mask, masked_value=0) # [bs x n_chn x freq_patches x time_patches x fs_patch x ts_patch]
        masked_patches = rearrange(masked_patches, 'b c fp tp fs ts -> b c (fp fs) (tp ts)')
        target_patches = apply_inv_mask(input_patches, mask, masked_value=0)

        if revin:
            inputs = revin(masked_patches, mode='norm')
        outputs = model(masked_patches)

        outputs = rearrange(outputs, 'b (fp tp) (fs ts) -> b fp tp fs ts', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch).unsqueeze(1)
        predicted_masked_regions = apply_inv_mask(outputs, mask, masked_value=0)
        loss = criterion(predicted_masked_regions, target_patches)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})

    avg_train_loss = train_loss / len(train_loader)

    # Log to Neptune
    if neptune_enabled:
        run["train/epoch_loss"].log(avg_train_loss, step=epoch)


    plot_sample_reconstruction(model, train_sample, device, model_config,
                               os.path.join(timestamped_file_name, 'train'), epoch)

    return avg_train_loss


def val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, val_loader, val_sample, model, revin, criterion, epoch_pbar, epoch, avg_train_loss):
    model.eval()
    val_loss = 0

    val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
    for batch in val_step_pbar:
        with torch.no_grad():
            inputs = batch['data'].to(device, non_blocking=True).unsqueeze(1)
            # masked_input, mask = fixed_mask_inputs(inputs,
            #                                        time_mask_consecutive_min=train_config['time_mask_consecutive_min'],
            #                                        time_mask_consecutive_max=train_config['time_mask_consecutive_max'],
            #                                        freq_mask_consecutive_min=train_config['freq_mask_consecutive_min'],
            #                                        freq_mask_consecutive_max=train_config['freq_mask_consecutive_max'],
            #                                        time_mask_p=train_config['time_mask_p'],
            #                                        freq_mask_p=train_config['freq_mask_p'])

            input_patches, freq_patches, time_patches = create_patches_2D(inputs, model_config)
            fs_patch, ts_patch = model_config['patch_size']
            mask = create_mask_2D(input_patches, model_config)

            masked_patches = apply_mask(input_patches, mask, masked_value=0) # [bs x n_chn x freq_patches x time_patches x fs_patch x ts_patch]
            masked_patches = rearrange(masked_patches, 'b c fp tp fs ts -> b c (fp fs) (tp ts)')
            target_patches = apply_inv_mask(input_patches, mask, masked_value=0)

            if revin:
                inputs = revin(inputs, mode='norm')

            outputs = model(masked_patches)

            outputs = rearrange(outputs, 'b (fp tp) (fs ts) -> b fp tp fs ts', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch).unsqueeze(1)
            predicted_masked_regions = apply_inv_mask(outputs, mask, masked_value=0)
            loss = criterion(predicted_masked_regions, target_patches)

            val_loss += loss.item()
            val_step_pbar.set_postfix({"Val Loss": val_loss / (val_step_pbar.n + 1)})

    avg_val_loss = val_loss / len(val_loader)

    # Log to Neptune
    if neptune_enabled:
        run["val/epoch_loss"].log(avg_val_loss, step=epoch)

    plot_sample_reconstruction(model, val_sample, device, model_config, 
                                os.path.join(timestamped_file_name, 'val'), epoch)

    # Update progress bar
    epoch_pbar.set_postfix({
        "Train Loss": avg_train_loss,
        "Val Loss": avg_val_loss
    })

    return avg_val_loss


def save_models(config, timestamped_file_name, model, revin, optimizer, scheduler, best_val_loss, num_checkpoints, epoch, avg_train_loss, avg_val_loss):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'revin_state_dict': revin.state_dict() if revin else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': config,
    }

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(checkpoint, os.path.join(timestamped_file_name, 'best_model.pth'))
        print(f"Best model saved at epoch {epoch + 1} with val loss: {avg_val_loss}")

    # Save periodic checkpoints
    if num_checkpoints != 0 and (epoch + 1) % num_checkpoints == 0:
        torch.save(checkpoint, os.path.join(timestamped_file_name, f'checkpoint_epoch_{epoch+1}.pth'))
        print(f"Checkpoint saved at epoch {epoch + 1}")

def create_patches_2D(inputs, model_config):
    # Create patches from the input data
    bs, n_chn, fs, ts = inputs.shape
    # patch_size = ast.literal_eval(model_config['patch_size']) # (fs_patch, ts_patch)
    patch_size = model_config['patch_size'] # (fs_patch, ts_patch)
    fs_patch, ts_patch = patch_size

    freq_patches = fs // fs_patch
    time_patches = ts // ts_patch
    num_patches = freq_patches * time_patches
    patch_dim = n_chn * fs_patch * ts_patch

    patches = inputs.unfold(2, fs_patch, fs_patch).unfold(3, ts_patch, ts_patch) # [bs, n_chn, num_fs_patches, num_ts_patches, fs_patch, ts_patch]
    # patches = patches.contiguous().view(bs, n_chn, num_patches, fs_patch, ts_patch) # [bs, n_chn, num_patches, fs_patch, ts_patch]

    return patches, freq_patches, time_patches


def create_mask_2D(patches, model_config):
    # Create a mask for the patches
    bs, n_chn, freq_patches, time_patches, fs_patch, ts_patch = patches.shape
    time_mask_ratio = model_config['time_masking_ratio']
    freq_mask_ratio = model_config['freq_masking_ratio']

    time_masked_patches = int(time_patches * time_mask_ratio)
    freq_masked_patches = int(freq_patches * freq_mask_ratio)

    mask = torch.zeros((bs, n_chn, freq_patches, time_patches), dtype=torch.bool).to(patches.device, non_blocking=True)

    if model_config['mask_type'] == 'random':
        time_mask_indices = torch.randperm(time_patches)[:time_masked_patches]
        freq_mask_indices = torch.randperm(freq_patches)[:freq_masked_patches]
        mask[:, :, freq_mask_indices, :] = True
        mask[:, :, :, time_mask_indices] = True

    return mask

def apply_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[mask] = masked_value

    return masked_patches

def apply_inv_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[~mask] = masked_value

    return masked_patches

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretraining script for PatchTST.")
    # parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--config', type=str, default='configs/pretrain/scalogram_based/test_pretrain_masked_vit_10_normal.yaml', help="Path to the configuration file.")
    parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint file to resume training.")
    args = parser.parse_args()

    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_file_path = args.config
    print(f"Using config file: {config_file_path}")
    # Load configuration
    config = Config(config_file=config_file_path).get()

    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    neptune_config = config['neptune']
    neptune_enabled = neptune_config['enabled']

    # Initialize Neptune
    run = None
    if neptune_enabled:
        run = init_neptune(config)

    # Save the model
    if not os.path.exists(model_config['save_path']):
        os.makedirs(model_config['save_path'])

    # create a folder with the current date and time
    experiment_name = neptune_config['experiment_name']
    if neptune_enabled:
        experiment_id = run["sys/id"].fetch()
        timestamped_file_name = os.path.join(model_config['save_path'], experiment_name, str(experiment_id), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        timestamped_file_name = os.path.join(model_config['save_path'], experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    if not os.path.exists(timestamped_file_name):
        os.makedirs(timestamped_file_name)
    
    # Save a copy of the configuration file for reproducibility
    import shutil
    config_backup_path = os.path.join(timestamped_file_name, os.path.basename(config_file_path))
    shutil.copy(config_file_path, config_backup_path)
    print(f"Configuration saved to {config_backup_path}")

    # Create data loaders
    dataset_name = "tuab_scalogram"
    train_loader, val_loader, _ = get_dataloaders(dataset_name, data_config, model_config)

    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))

    # Create model
    model_name = "ViT_MAE"
    model = get_pretrain_model(model_name, model_config, data_config).to(device)
    
    # Initialize Revin
    if model_config['revin']:
        revin = RevIN(data_config['n_vars'], 
                        float(model_config['revin_eps']),
                        bool(model_config['revin_affine'])).to(device)
    else:
        revin = None
    
    # Set up optimizer and scheduler
    optimizer = Adam(model.parameters(), 
                     lr=float(train_config['learning_rate']), 
                     weight_decay=float(train_config['weight_decay']))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_config['step_size'], 
                                                gamma=train_config['gamma'])

    criterion = torch.nn.MSELoss(reduction='mean') 

    num_epochs = train_config['num_epochs']
    val_interval_epochs = train_config['val_interval_epochs']
    best_val_loss = float('inf')
    best_epoch = 0
    checkpoint_interval = model_config['checkpoint_interval']
    num_checkpoints = int(checkpoint_interval * num_epochs)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint:
        best_model_path = os.path.join(os.path.dirname(args.checkpoint), 'best_model.pth')
        start_epoch, best_val_loss = load_checkpoint(
            args.checkpoint, model, revin, optimizer, scheduler, device, best_model_path
        )
        scheduler = reinitialize_scheduler(scheduler, optimizer, train_config, start_epoch, len(train_loader))

    # Training loop
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        avg_train_loss = train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                     train_loader, train_sample, model, revin, 
                                     optimizer, scheduler, criterion, epoch)
        
        if (epoch+1) % val_interval_epochs == 0:
            avg_val_losses = val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                    val_loader, val_sample, model, 
                                    revin, criterion, epoch_pbar, epoch, avg_train_loss)
    
            save_models(config, timestamped_file_name, model, revin, optimizer, scheduler, best_val_loss, num_checkpoints, epoch, avg_train_loss, avg_val_losses)


if __name__ == "__main__":
    main()
