from data import get_tuh_dataloaders
from configs import Config
from get_models import get_patchTST_model
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

DEBUG = True

def init_neptune(config):

    return neptune.init_run(
        project=config['neptune']['project'], 
        name=config['neptune']['experiment_name']
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

    mask = torch.ones((bs, num_patches, n_vars), dtype=torch.bool).to(patches.device)

    if mask_type == 'random':
        if independent_channel_masking:
            # Generate a mask for each channel
            for i in range(n_vars):
                mask_indices = torch.randperm(num_patches)[:num_masked_patches]
                mask[:, mask_indices, i] = False
        else:
            # Generate a mask for each patch
            mask_indices = torch.randperm(num_patches)[:num_masked_patches]
            mask[:, mask_indices, :] = False
    elif mask_type == 'forecasting': # or continous towards the end
        if forecasting_num_patches is None:
            raise ValueError("forecasting_num_patches must be provided for forecasting masking.")
        else:
            mask[:, -forecasting_num_patches:, :] = False
    elif mask_type == 'fixed_position':
        if fixed_position is None:
            raise ValueError("fixed_position must be provided for fixed position masking.")
        else:
            mask[:, fixed_position, :] = False
    else:
        raise ValueError("Invalid mask type. Choose 'random', 'forecasting', or 'fixed_position'.")

    return mask

def apply_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[mask] = masked_value

    return masked_patches # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

def debug_plot(masked_patches, output, mask, split='train'):
    # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]
    #plot input and output and mask
    mask = mask[0, :, 0].detach().cpu().numpy() # [num_patch]
    plt.figure(figsize=(15, 10))
    plt.plot(masked_patches[0, :, 0, :].detach().cpu().numpy().flatten(), label='Masked Input', color='orange')
    plt.plot(output[0, :, 0, :].detach().cpu().numpy().flatten(), label='Output', color='blue', alpha=0.5)
    # plt.plot([np.ones(10)*i for i in mask[0, :, 0].detach().cpu().numpy()], label='Mask', alpha=0.5)
    # mask_value = int(mask[0, 0, 0].detach().cpu().numpy())  # 0 or 1
    # plt.axhline(y=mask_value, color='r', linestyle='--', 
    #             label=f'Mask ({mask_value})', alpha=0.5)
    plt.legend()
    plt.title('Masked Patch, Masked Output and Mask')
    plt.savefig(f'debug_{split}.png')
    plt.close()

def plot_sample_reconstruction(model, revin, sample, mask_ratio, masked_value, mask_type, stride, 
                               independent_channel_masking, patch_len, device, epoch, split='train', num_channels=3):
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
    """
    model.eval()
    
    # Prepare sample
    data = sample['past_values'].to(device)
    
    # if revin:
    #     data = revin(data, mode='norm')
    
    patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(patches, mask_ratio, independent_channel_masking, mask_type=mask_type)
    masked_patches = apply_mask(patches, mask, masked_value)
    
    # Get model prediction
    with torch.no_grad():
        output = model(masked_patches)
    
    # Get total available channels and limit to actual number available
    n_vars = patches.shape[2]
    num_channels = min(num_channels, n_vars)
    
    # Create figure with subplots stacked vertically
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 5*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # Make it iterable for the loop
    
    total_length = patches.shape[1] * patch_len
    
    # Plot each channel
    for ch_idx in range(num_channels):
        ax = axes[ch_idx]
        
        # Get data for current channel
        orig_signal = patches[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        masked_signal = masked_patches[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        recon_signal = output[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        mask_regions = ~mask[0, :, ch_idx].detach().cpu().numpy()
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
            if mask_regions[i]:
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

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis='x', which='both', labelsize=10, labelbottom=True)
    
    axes[-1].set_xlabel('Timestamps')

    # Create directory if it doesn't exist
    os.makedirs(split, exist_ok=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust top to make room for the suptitle
    plt.savefig(os.path.join(split, f'reconstruction_epoch_{epoch}_mask_type_{mask_type}.png'))
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
    data = batch['past_values'].to(device) # [batch_size, seq_len, n_vars]

    if revin:
        data = revin(data, mode='norm')

    input_patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(input_patches, mask_ratio, independent_channel_masking, mask_type='random') # True: unmasked, False: masked
    masked_patches = apply_mask(input_patches, mask, masked_value)
    # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

    # Forward pass
    optimizer.zero_grad()
    output = model(masked_patches)
    
    masked_output_patches = apply_mask(output, mask, masked_value)

    if DEBUG:
        # debug_plot(masked_patches, output, mask, split='train')
        debug_plot(masked_output_patches, masked_patches, mask, split='val')

    # Compute loss
    loss = criterion(masked_output_patches, masked_patches)

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
    data = batch['past_values'].to(device) # [batch_size, seq_len, n_vars]

    if revin:
        data = revin(data, mode='norm')

    input_patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(input_patches, mask_ratio, independent_channel_masking, mask_type=val_mask_type, 
                       fixed_position=fixed_position, forecasting_num_patches=forecasting_num_patches) # True: unmasked, False: masked
    masked_patches = apply_mask(input_patches, mask, masked_value)

    # Forward pass
    with torch.no_grad():
        output = model(masked_patches)
    
    masked_output_patches = apply_mask(output, mask, masked_value)

    if DEBUG:
        # debug_plot(masked_patches, output, mask, split='val')
        debug_plot(masked_output_patches, masked_patches, mask, split='val')
    
    # Compute loss
    loss = criterion(masked_output_patches, masked_patches)
    
    return loss.item()

def main():
    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_file_path = 'configs/test_pretrain_repo_tuh.yaml'
    # Load configuration
    config = Config(config_file=config_file_path).get()

    data_config = config['data']
    model_config = config['model']
    neptune_config = config['neptune']
    neptune_enabled = neptune_config['enabled']

    # Initialize Neptune
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
    train_loader, val_loader, _ = get_tuh_dataloaders(
        data_config['root_path'],
        data_config['data_path'],
        data_config['csv_path'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        size=[model_config['seq_len'], 
              model_config['target_dim'],
              model_config['patch_length']]
    )

    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))

    num_patches = (max(model_config['seq_len'], model_config['patch_length']) - model_config['patch_length']) // model_config['stride'] + 1
    train_num_mask_patches = int(model_config['mask_ratio']* num_patches)
    print(f"num_patches: {num_patches}, train_num_mask_patches: {train_num_mask_patches}")
    # Create model
    model = get_patchTST_model(num_variates=data_config['n_vars'],
                                forecast_length=data_config['pred_len'],
                                patch_len=model_config['patch_length'],
                                stride=model_config['stride'],
                                num_patch=num_patches,
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
                                head_type=model_config['head_type']
                            ).to(device)
    
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
    val_mask_types = ['random', 'forecasting', 'fixed_position']

    # Training loop
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        # val_mask_losses = initialize this for all the mask types
        val_mask_losses = {mask_type: 0 for mask_type in val_mask_types}

        # Training step
        train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
        
        for batch in train_step_pbar:
            train_step_loss = train_step(model, revin, batch, optimizer, scheduler, criterion, model_config, device)
            train_loss += train_step_loss/train_num_mask_patches

            # Update progress bar
            train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})


        # TODO: add comparable val losses for all the mask types
        # TODO: plot sample reconstruction for all the mask types
        # TODO: overallapping is not working
        # Validation step for random masking

        # if epoch % val_interval_epochs != 0:

        val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
        for batch in val_step_pbar:
            val_step_loss = val_step(model, revin, batch, criterion, model_config, device, val_mask_type='random')
            val_mask_losses['random'] += val_step_loss/train_num_mask_patches
            # Update progress bar
            val_step_pbar.set_postfix({"Val Loss": val_mask_losses['random'] / (val_step_pbar.n + 1)})

        # Validation step for forecasting masking
        val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
        forecasting_num_patches = model_config['forecasting_num_patches']
        for batch in val_step_pbar:
            val_step_loss = val_step(model, revin, batch, criterion, model_config, device, val_mask_type='forecasting')
            val_mask_losses['forecasting'] += val_step_loss/forecasting_num_patches
            # Update progress bar
            val_step_pbar.set_postfix({"Val Loss": val_mask_losses['forecasting'] / (val_step_pbar.n + 1)})

        # Validation step for fixed position masking
        val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
        for batch in val_step_pbar:
            for fixed_position in range(num_patches):
                val_step_loss = val_step(model, revin, batch, criterion, model_config, device, val_mask_type='fixed_position', fixed_position=fixed_position)
                val_mask_losses['fixed_position'] += val_step_loss/num_patches
                # Update progress bar
                val_step_pbar.set_postfix({"Val Loss": val_mask_losses['fixed_position'] / (val_step_pbar.n + 1)})

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_losses = {mask_type: loss / len(val_loader) for mask_type, loss in val_mask_losses.items()}

        # Log to Neptune
        if neptune_enabled:
            run["train/epoch_loss"].log(avg_train_loss)
            for mask_type, loss in avg_val_losses.items():
                run[f"val/{mask_type}_epoch_loss"].log(loss)
    
        # Update progress bar
        epoch_pbar.set_postfix({"Train Loss": avg_train_loss,
                               "Val Loss": {mask_type: loss for mask_type, loss in avg_val_losses.items()}})

        plot_sample_reconstruction(
            model, revin, train_sample, 
            model_config['mask_ratio'], model_config['masked_value'], 'random',
            model_config['stride'], model_config['independent_channel_masking'],
            model_config['patch_length'], device, epoch, 
            os.path.join(timestamped_file_name, 'train'),
            num_channels=6
        )
        
        # plot_sample_reconstruction(
        #     model, revin, val_sample, 
        #     model_config['mask_ratio'], model_config['masked_value'], 
        #     model_config['stride'], model_config['independent_channel_masking'],
        #     model_config['patch_length'], device, epoch, 
        #     os.path.join(timestamped_file_name, 'val'),
        #     num_channels=6
        # )

        # TODO visualize all the mask types
        plot_sample_reconstruction(
            model, revin, val_sample, 
            model_config['mask_ratio'], model_config['masked_value'], 'random',
            model_config['stride'], model_config['independent_channel_masking'],
            model_config['patch_length'], device, epoch, 
            os.path.join(timestamped_file_name, 'val'),
            num_channels=6
        )
    
        if avg_val_losses['random'] < best_val_loss:
            best_val_loss = avg_val_losses['random']
            best_epoch = epoch

            # Save the model
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'revin_state_dict': revin.state_dict() if revin else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_losses,
                'config': config,
            }
            print(f"Best model saved at epoch {epoch+1} with val loss: {avg_val_losses['random']}")
            torch.save(checkpoint, os.path.join(timestamped_file_name, 'best_model.pth'))

        # Save the model every checkpoint_interval epochs
        if num_checkpoints != 0 and (epoch + 1) % num_checkpoints == 0:
            # save model, optimizer, scheduler, epoch
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'revin_state_dict': revin.state_dict() if revin else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_losses,
                'config': config,
            }

            print(f"Checkpoint saved at epoch {epoch+1}")

            torch.save(checkpoint, os.path.join(timestamped_file_name, f'checkpoint_epoch_{epoch+1}.pth'))


if __name__ == "__main__":
    main()