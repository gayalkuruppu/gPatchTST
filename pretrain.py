from data import get_tuh_dataloaders
from dataloaders.dataloaders import get_dataloaders
from configs import Config
from get_models import get_patchTST_model, get_pretrain_model
from models.patchtst.layers.revin import RevIN
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch
from tqdm import tqdm
from datetime import datetime
import neptune
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

DEBUG = False

def init_neptune(config):
    """
    Initialize a new Neptune run.
    """
    print("Creating a new Neptune run.")
    return neptune.init_run(
        project=config['neptune']['project'], 
        name=config['neptune']['experiment_name'],
        capture_stdout=False,  # Avoid duplicate logging of stdout
        capture_stderr=False   # Avoid duplicate logging of stderr
    )

def create_patches(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    # breakpoint()
    num_patches = (seq_len - patch_len) // stride + 1
    tgt_len = patch_len + stride*(num_patches - 1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patches


def create_mask(patches, mask_ratio, independent_channel_masking=False, 
                mask_type='random', forecasting_num_patches=None, fixed_position=None):
    bs, num_patches, n_vars, patch_len = patches.shape
    num_masked_patches = int(num_patches * mask_ratio)

    mask = torch.zeros((bs, num_patches, n_vars), dtype=torch.bool).to(patches.device, non_blocking=True) #CHANGES

    if mask_type == 'random':
        if independent_channel_masking:
            # Generate a mask for each channel
            for i in range(n_vars):
                mask_indices = torch.randperm(num_patches)[:num_masked_patches]
                mask[:, mask_indices, i] = True #CHANGES
        else:
            # Generate a mask for each patch
            mask_indices = torch.randperm(num_patches)[:num_masked_patches]
            mask[:, mask_indices, :] = True #CHANGES
    elif mask_type == 'forecasting': # or continous towards the end
        if forecasting_num_patches is None:
            raise ValueError("forecasting_num_patches must be provided for forecasting masking.")
        else:
            mask[:, -forecasting_num_patches:, :] = True #CHANGES
    elif mask_type == 'backcasting': # or continous towards the beginning
        if forecasting_num_patches is None:
            raise ValueError("forecasting_num_patches must be provided for backcasting masking.")
        else:
            mask[:, :forecasting_num_patches, :] = True
    elif mask_type == 'fixed_position':
        if fixed_position is None:
            raise ValueError("fixed_position must be provided for fixed position masking.")
        else:
            mask[:, fixed_position, :] = True #CHANGES
    else:
        raise ValueError("Invalid mask type. Choose 'random', 'forecasting', or 'fixed_position'.")

    return mask

def apply_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[mask] = masked_value

    return masked_patches # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

def apply_inv_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[~mask] = masked_value

    return masked_patches # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

def debug_plot(masked_patches, target_patches, predicted_masked_regions, mask, split='train'):
    mask = mask[0, :, 0].detach().cpu().numpy() # [num_patch]
    plt.figure(figsize=(15, 10))
    plt.plot(masked_patches[0, :, 0, :].detach().cpu().numpy().flatten(), label='Masked Input', color='orange')
    plt.plot(target_patches[0, :, 0, :].detach().cpu().numpy().flatten(), label='Target', color='green')
    plt.plot(predicted_masked_regions[0, :, 0, :].detach().cpu().numpy().flatten(), label='Predicted', color='red')
    plt.legend()
    plt.title('Masked Patch, Masked Output and Mask')
    plt.savefig(f'debug_run_{split}.png')
    plt.close()

def plot_sample_reconstruction(model, revin, sample, mask_ratio, masked_value, mask_type, stride,
                               independent_channel_masking, patch_len, device, epoch, split='train', num_channels=3, 
                               forecasting_num_patches=None, fixed_position=None):
    """
    Plot a constant sample's reconstruction at the end of each epoch
    Shows multiple channels stacked vertically
    
    Args:
        model: The model to use for reconstruction
        revin: RevIN normalization module (or None)
        sample: The sample to reconstruct
        mask_ratio: Ratio of patches to mask
        masked_value: Value to use for masked patches
        mask_type: Type of masking ('random', 'forecasting', 'fixed_position')
        stride: Stride for patching
        independent_channel_masking: Whether to mask channels independently
        patch_len: Patch length
        device: Device to use
        epoch: Current epoch number
        split: Dataset split ('train' or 'val')
        num_channels: Number of channels to plot
        forecasting_num_patches: Number of patches to mask for forecasting
        fixed_position: Fixed position for masking
    """
    model.eval()
    
    # Prepare sample
    data = sample['past_values'].to(device, non_blocking=True)
    
    if revin:
        data = revin(data, mode='norm')
    
    input_patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(input_patches, mask_ratio, independent_channel_masking, mask_type=mask_type,
                       forecasting_num_patches=forecasting_num_patches, fixed_position=fixed_position)
    masked_patches = apply_mask(input_patches, mask, masked_value)
    target_patches = apply_inv_mask(input_patches, mask, masked_value)
    
    # Get model prediction
    with torch.no_grad():
        predicted_sequence = model(masked_patches)
    
    # Get total available channels and limit to actual number available
    n_vars = input_patches.shape[2]
    num_channels = min(num_channels, n_vars)
    
    # Create figure with subplots stacked vertically
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 5*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # Make it iterable for the loop
    
    total_length = input_patches.shape[1] * patch_len
    
    # Plot each channel
    for ch_idx in range(num_channels):
        ax = axes[ch_idx]
        
        # Get data for current channel
        orig_signal = input_patches[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        masked_signal = masked_patches[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        recon_signal = predicted_sequence[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        mask_regions = mask[0, :, ch_idx].detach().cpu().numpy() #CHANGES
        # breakpoint()
        # Plot signals
        ax.plot(orig_signal, label='Original', color='blue', alpha=0.7)
        ax.plot(recon_signal, label='Reconstruction', color='red')
        
        # Draw patch boundary grid
        for i in range(num_patches + 1):
            boundary_pos = i * patch_len
            if boundary_pos <= total_length:
                ax.axvline(x=boundary_pos, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight masked regions
        for i in range(len(mask_regions)):
            if (mask_regions[i]):
                start_idx = i * patch_len
                end_idx = start_idx + patch_len
                ax.axvspan(start_idx, end_idx, color='yellow', alpha=0.2)
        
        # Add grid
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # Add channel label
        ax.set_title(f'Channel {ch_idx+1} of {n_vars}', fontsize=14)
        
        # # Only add legend to the first subplot to avoid redundancy
        # if ch_idx == 0:
        ax.legend()
    
    # Add overall title
    fig.suptitle(f'Epoch {epoch}: {os.path.basename(split).capitalize()} Sample Reconstruction - {split} split, Mask Type: {mask_type}', fontsize=16)
    
    # Add x-ticks at patch boundaries
    tick_positions = [i * patch_len for i in range(num_patches + 1)]
    tick_labels = [str(i * stride) for i in range(num_patches + 1)]

    # Interleave tick labels (e.g., show every other label)
    interleave_factor = max(1, len(tick_positions) // 20)  # Adjust based on the number of patches
    tick_labels = [label if i % interleave_factor == 0 else '' for i, label in enumerate(tick_labels)]

    # Dynamically increase plot size horizontally based on the number of patches
    fig_width = max(15, num_patches // 10)  # Adjust width dynamically
    fig.set_size_inches(fig_width, fig.get_size_inches()[1])  # Keep the height unchanged

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis='x', which='both', labelsize=10, labelrotation=90, labelbottom=True)
    
    axes[-1].set_xlabel('Timestamps')

    # Create directory if it doesn't exist
    os.makedirs(split, exist_ok=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust top to make room for the suptitle
    plt.savefig(os.path.join(split, f'reconstruction_epoch_{epoch+1}_mask_type_{mask_type}.png'))
    plt.close()
    
    return

def train_step(model, revin, batch, optimizer, scheduler, criterion, model_config, device):
    # only norm, no denorm
    mask_ratio = model_config['mask_ratio']
    masked_value = model_config['masked_value']
    stride = model_config['stride']
    independent_channel_masking = model_config['independent_channel_masking']
    patch_len = model_config['patch_length']
    
    model.train()

    # Unpack batch
    data = batch['past_values'].to(device, non_blocking=True) # [batch_size, seq_len, n_vars]

    if revin:
        data = revin(data, mode='norm')

    input_patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(input_patches, mask_ratio, independent_channel_masking, mask_type='random') # True: unmasked, False: masked
    masked_patches = apply_mask(input_patches, mask, masked_value)
    target_patches = apply_inv_mask(input_patches, mask, masked_value)
    # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

    # Forward pass
    optimizer.zero_grad()
    predicted_sequence = model(masked_patches)
    
    predicted_masked_regions = apply_inv_mask(predicted_sequence, mask, masked_value)

    if DEBUG:
        # print(mask[0, :, 0])
        # debug_plot(masked_patches, output, mask, split='train')
        debug_plot(masked_patches, target_patches, predicted_masked_regions, mask, split='train')

    # Compute loss
    loss = criterion(predicted_masked_regions, target_patches)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return loss.item()

def val_step(model, revin, batch, criterion, model_config, device, val_mask_type='random', fixed_position=None):
    # only norm, no denorm
    mask_ratio = model_config['mask_ratio']
    masked_value = model_config['masked_value']
    stride = model_config['stride']
    independent_channel_masking = model_config['independent_channel_masking']
    patch_len = model_config['patch_length']
    forecasting_num_patches = model_config['forecasting_num_patches']

    model.eval()

    # Unpack batch
    data = batch['past_values'].to(device, non_blocking=True) # [batch_size, seq_len, n_vars]

    if revin:
        data = revin(data, mode='norm')

    input_patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(input_patches, mask_ratio, independent_channel_masking, mask_type=val_mask_type, 
                       fixed_position=fixed_position, forecasting_num_patches=forecasting_num_patches)
    masked_patches = apply_mask(input_patches, mask, masked_value)
    target_patches = apply_inv_mask(input_patches, mask, masked_value)

    # Forward pass
    with torch.no_grad():
        predicted_sequence = model(masked_patches)

    predicted_masked_regions = apply_inv_mask(predicted_sequence, mask, masked_value)

    if DEBUG:
        # debug_plot(masked_patches, output, mask, split='val')
        debug_plot(masked_patches, target_patches, predicted_masked_regions, mask, split='val')
    
    # Compute loss
    loss = criterion(predicted_masked_regions, target_patches)
    
    return loss.item()

def load_checkpoint(checkpoint_path, model, revin, optimizer, scheduler, device, best_model_path=None):
    """
    Load model, optimizer, scheduler, and other states from a checkpoint.
    Optionally load the best_val_loss from the best_model.pth checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if revin and checkpoint['revin_state_dict']:
        revin.load_state_dict(checkpoint['revin_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    # Load best_val_loss from best_model.pth if provided
    if best_model_path and os.path.exists(best_model_path):
        best_model_checkpoint = torch.load(best_model_path, map_location=device)
        best_val_loss = best_model_checkpoint['val_loss']['random']
        print(f"Loaded best_val_loss from {best_model_path}: {best_val_loss}")

    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with best val loss: {best_val_loss}")
    return start_epoch, best_val_loss

def reinitialize_scheduler(scheduler, optimizer, train_configs, start_epoch, steps_per_epoch):
    """
    Reinitialize the OneCycleLR scheduler with the correct total steps when resuming training.
    """
    total_steps = train_configs['num_epochs'] * steps_per_epoch
    completed_steps = start_epoch * steps_per_epoch
    remaining_steps = total_steps - completed_steps

    if remaining_steps <= 0:
        raise ValueError("No remaining steps for the scheduler. Check your num_epochs and start_epoch values.")

    return OneCycleLR(
        optimizer,
        max_lr=float(train_configs['learning_rate']),
        total_steps=remaining_steps,
        pct_start=0.3,  # Default value
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
    )

def create_checkpoint(model, revin, optimizer, scheduler, avg_train_loss, avg_val_losses, config, epoch):
    """
    Create a checkpoint dictionary with all necessary states.
    """
    return {
        'epoch': epoch + 1,  # Assuming model has an `epoch` attribute
        'model_state_dict': model.state_dict(),
        'revin_state_dict': revin.state_dict() if revin else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_losses,
        'config': config,
    }

def save_checkpoint(checkpoint, filepath, description="Checkpoint"):
    """
    Save a checkpoint to the specified filepath.
    """
    torch.save(checkpoint, filepath)
    print(f"{description} saved at {filepath}")

def save_models(config, timestamped_file_name, model, revin, optimizer, scheduler, best_val_loss, num_checkpoints, epoch, avg_train_loss, avg_val_losses):
    """
    Save the best model and periodic checkpoints.
    """
    checkpoint = create_checkpoint(model, revin, optimizer, scheduler, avg_train_loss, avg_val_losses, config, epoch)

    # Save the best model
    if avg_val_losses['random'] < best_val_loss:
        best_val_loss = avg_val_losses['random']
        save_checkpoint(checkpoint, os.path.join(timestamped_file_name, 'best_model.pth'), "Best model")

    # Save periodic checkpoints
    if num_checkpoints != 0 and (epoch + 1) % num_checkpoints == 0:
        save_checkpoint(checkpoint, os.path.join(timestamped_file_name, f'checkpoint_epoch_{epoch+1}.pth'), "Periodic checkpoint")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretraining script for PatchTST.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
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
    dataset_name = data_config['dataset_name']
    train_loader, val_loader, _ = get_dataloaders(dataset_name, data_config, model_config)

    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))

    num_patches = (max(model_config['seq_len'], model_config['patch_length']) - model_config['patch_length']) // model_config['stride'] + 1
    train_num_mask_patches = int(model_config['mask_ratio']* num_patches)
    print(f"num_patches: {num_patches}, train_num_mask_patches: {train_num_mask_patches}")
    # Create model
    model_name = model_config['model_name']
    model = get_pretrain_model(model_name, model_config, data_config).to(device)
    
    # Initialize Revin
    if model_config['revin']:
        revin = RevIN(data_config['n_vars'], 
                        float(model_config['revin_eps']),
                        bool(model_config['revin_affine'])).to(device)
    else:
        revin = None
    print(f"model_config['revin_affine'] : {model_config['revin_affine'] is bool}")
    
    train_configs = config['train']
    # Set up optimizer and scheduler
    optimizer = Adam(model.parameters(), 
                     lr=float(train_configs['learning_rate']), 
                     weight_decay=float(train_configs['weight_decay']))
    
    scheduler = OneCycleLR(optimizer,
                            max_lr=float(train_configs['learning_rate']),
                            epochs=train_configs['num_epochs'],
                            steps_per_epoch=len(train_loader),
                            pct_start=0.3, # pytroch and patchtst git repo default
                            anneal_strategy='cos',
                            cycle_momentum=True,
                            base_momentum=0.85,
                            max_momentum=0.95,
                            div_factor=25,)

    criterion = torch.nn.MSELoss(reduction='mean') 

    num_epochs = train_configs['num_epochs']
    val_interval_epochs = train_configs['val_interval_epochs']
    best_val_loss = float('inf')
    best_epoch = 0
    checkpoint_interval = model_config['checkpoint_interval']
    num_checkpoints = int(checkpoint_interval * num_epochs)
    val_mask_types = model_config['val_mask_types']
    print(f"num_checkpoints: {num_checkpoints}, val_mask_types: {val_mask_types}")

    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint:
        best_model_path = os.path.join(os.path.dirname(args.checkpoint), 'best_model.pth')
        start_epoch, best_val_loss = load_checkpoint(
            args.checkpoint, model, revin, optimizer, scheduler, device, best_model_path
        )
        scheduler = reinitialize_scheduler(scheduler, optimizer, train_configs, start_epoch, len(train_loader))

    # Training loop
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        avg_train_loss = train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                     train_loader, train_sample, train_num_mask_patches, model, revin, 
                                     optimizer, scheduler, criterion, epoch)
        
        if (epoch+1) % val_interval_epochs == 0:
            avg_val_losses = val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                    val_loader, val_sample, num_patches, train_num_mask_patches, model, 
                                    revin, criterion, val_mask_types, epoch_pbar, epoch, avg_train_loss)
    
            save_models(config, timestamped_file_name, model, revin, optimizer, scheduler, best_val_loss, num_checkpoints, epoch, avg_train_loss, avg_val_losses)


def val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, val_loader,
               val_sample, num_patches, train_num_mask_patches, model, revin, criterion,
                 val_mask_types, epoch_pbar, epoch, avg_train_loss):
    
    val_mask_losses = {mask_type: 0 for mask_type in val_mask_types}

    for mask_type in val_mask_types:
        val_step_pbar = tqdm(val_loader, desc=f"Validation ({mask_type})", total=len(val_loader), leave=False)
        for batch in val_step_pbar:
            if mask_type == 'fixed_position':
                for fixed_position in range(num_patches):
                    val_step_loss = val_step(
                        model, revin, batch, criterion, model_config, device,
                        val_mask_type=mask_type, fixed_position=fixed_position
                    )
                    val_mask_losses[mask_type] += val_step_loss / num_patches
            else:
                val_step_loss = val_step(
                    model, revin, batch, criterion, model_config, device,
                    val_mask_type=mask_type
                )
                divisor = train_num_mask_patches if mask_type == 'random' else model_config['forecasting_num_patches']
                val_mask_losses[mask_type] += val_step_loss / divisor

            # Update progress bar
            val_step_pbar.set_postfix({"Val Loss": val_mask_losses[mask_type] / (val_step_pbar.n + 1)})

    avg_val_losses = {mask_type: loss / len(val_loader) for mask_type, loss in val_mask_losses.items()}

    # Log to Neptune
    if neptune_enabled:
        for mask_type, loss in avg_val_losses.items():
            run[f"val/{mask_type}_epoch_loss"].log(loss, step=epoch)

    # Update progress bar
    epoch_pbar.set_postfix({
        "Train Loss": avg_train_loss,
        "Val Loss": avg_val_losses
    })

    # Plot sample reconstructions
    for mask_type in ['random', 'forecasting', 'backcasting']:
        plot_sample_reconstruction(
            model, revin, val_sample,
            model_config['mask_ratio'], model_config['masked_value'], mask_type,
            model_config['stride'], model_config['independent_channel_masking'],
            model_config['patch_length'], device, epoch,
            os.path.join(timestamped_file_name, 'val'),
            num_channels=6,
            forecasting_num_patches=model_config.get('forecasting_num_patches')
        )
        
    return avg_val_losses

def train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, train_loader, train_sample, train_num_mask_patches, model, revin, optimizer, scheduler, criterion, epoch):
    model.train()
    train_loss = 0

        # Training step
    train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
        
    for batch in train_step_pbar:
        train_step_loss = train_step(model, revin, batch, optimizer, scheduler, criterion, model_config, device)
        train_loss += train_step_loss/train_num_mask_patches

            # Update progress bar
        train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})

        # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)

        # Log to Neptune
    if neptune_enabled:
        run["train/epoch_loss"].log(avg_train_loss, step=epoch)

    plot_sample_reconstruction(
            model, revin, train_sample, 
            model_config['mask_ratio'], model_config['masked_value'], 'random',
            model_config['stride'], model_config['independent_channel_masking'],
            model_config['patch_length'], device, epoch, 
            os.path.join(timestamped_file_name, 'train'),
            num_channels=6
        )
    
    return avg_train_loss


if __name__ == "__main__":
    main()
