import os
import ast
import shutil
import argparse
import numpy as np
import logging
from datetime import datetime

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

from get_models import get_pretrain_model
from dataloaders.dataloaders import get_dataloaders
from configs import Config
from utils.utils import init_neptune
from utils.pretrain_utils import load_checkpoint
from models.patchtst.layers.revin import RevIN
# Initialize model

DEBUG = True

def plot_sample_reconstruction(model, sample, device, model_config, save_path, epoch):
    model.eval()

    data = sample['data'].to(device, non_blocking=True)[0] # first sample from the batch
    data = data.unsqueeze(0).unsqueeze(0) # [1, 1, 25, 100]

    fs_patch, ts_patch = model_config['patch_size']
    input_patches, freq_patches, time_patches = create_patches_2D(data, model_config)
    mask = create_mask_2D(input_patches, model_config)

    masked_patches = process_masked_patches_2D(input_patches, mask, masked_value=0) # [bs x n_chn x freq_patches x time_patches x fs_patch x ts_patch]
    print(masked_patches.shape)
    with torch.no_grad():
        predicted_patches = model(masked_patches)
        print(f"predicted_patches shape: {predicted_patches.shape}")

    # reconstruction = rearrange(predicted_patches, 'b (fp tp) (fs ts) -> b fp tp fs ts', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch).unsqueeze(1)
    # masked_reconstruction = apply_inv_mask(reconstruction, mask, masked_value=0)
    predicted_masked_regions = process_output_patches_2D(predicted_patches, mask, freq_patches, time_patches, fs_patch, ts_patch, masked_value=0)
    masked_reconstruction = rearrange(predicted_masked_regions, 'b c fp tp fs ts -> b c (fp fs) (tp ts)', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch)
    os.makedirs(save_path, exist_ok=True)

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
    

def debug_plot(masked_patches, target_patches, predicted_masked_regions, outputs, mask, split='train'):
    # Plot
    # p1 = fs, p2 = ts, h =fp, w = tp
    fp = mask.shape[2]
    tp = mask.shape[3]
    fs = target_patches.shape[4]
    ts = target_patches.shape[5]
    # masked_patches = rearrange(masked_patches, 'b c (fp fs) (tp ts) -> b c fp tp fs ts', fp=mask.shape[2], tp=mask.shape[3], fs=target_patches.shape[4], ts=target_patches.shape[5])
    target_patches = rearrange(target_patches, 'b c fp tp fs ts -> b c (fp fs) (tp ts)', fp=mask.shape[2], tp=mask.shape[3], fs=target_patches.shape[4], ts=target_patches.shape[5])
    predicted_masked_regions = rearrange(predicted_masked_regions, 'b c fp tp fs ts -> b c (fp fs) (tp ts)', fp=mask.shape[2], tp=mask.shape[3], fs=predicted_masked_regions.shape[4], ts=predicted_masked_regions.shape[5])
    outputs_test = outputs.permute(0, 2, 1)
    # outputs = rearrange(outputs, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)')
    outputs = rearrange(outputs, 'b (fp tp) (fs ts c) -> b c (fp fs) (tp ts)', fp=fp, tp=tp, fs=fs, ts=ts)
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    axs[0].imshow(masked_patches[0, 0].cpu().numpy(), cmap='viridis')
    axs[0].set_title(f'{split} - Masked Input')
    axs[1].imshow(target_patches[0, 0].cpu().numpy(), cmap='viridis')
    axs[1].set_title(f'{split} - Target Patches')
    axs[2].imshow(predicted_masked_regions[0, 0].cpu().detach().numpy(), cmap='viridis')
    axs[2].set_title(f'{split} - Predicted Masked Regions')
    axs[3].imshow(outputs[0, 0].cpu().detach().numpy(), cmap='viridis')
    axs[3].set_title(f'{split} - Model Output')
    axs[4].imshow(outputs_test[0].cpu().detach().numpy(), cmap='viridis')
    axs[4].set_title(f'{split} - outputs_test')
    plt.tight_layout()
    plt.savefig(f'{split}_debug_plot.png')
    plt.close(fig)


def patch_rearange(patches):
    # patches -> [500, 70] # [num_patches, patch_size]
    patches = rearrange(patches, '(h_num w_num) (ph pw) -> (h_num ph) (w_num pw)', ph=7, pw=10, h_num=10, w_num=50)
    return patches

# def plot_mae(masked_patches, pred_pixel_values, patches, masked_indices, save_path, epoch):
#     # Plot MAE
#     os.makedirs(save_path, exist_ok=True)

#     input_image = patches[0].cpu().numpy() # [num_patches x ]

#     masked_indices = masked_indices.cpu().numpy()
#     all_indices = np.arange(input_image.shape[0])
#     unmasked_indices = all_indices[~np.isin(all_indices, masked_indices)]

#     masked_input = np.zeros_like(input_image)
#     masked_input[unmasked_indices] = input_image[unmasked_indices]

#     reconstruction = np.zeros_like(input_image)
#     reconstruction[masked_indices] = pred_pixel_values[0].cpu().detach().numpy()

#     reconstruction_and_visible = reconstruction.copy()
#     reconstruction_and_visible[unmasked_indices] = input_image[unmasked_indices]

#     input_image = patch_rearange(input_image)
#     masked_input = patch_rearange(masked_input)
#     reconstruction = patch_rearange(reconstruction)
#     reconstruction_and_visible = patch_rearange(reconstruction_and_visible)

#     fig, axs = plt.subplots(4, 1, figsize=(10, 10))
#     axs[0].imshow(input_image, cmap='viridis')
#     axs[0].set_title('Original Patches')
#     axs[1].imshow(masked_input, cmap='viridis')
#     axs[1].set_title('Masked Input')
#     axs[2].imshow(reconstruction, cmap='viridis')
#     axs[2].set_title('Reconstructed Output')
#     axs[3].imshow(reconstruction_and_visible, cmap='viridis')
#     axs[3].set_title('Reconstruction + Visible')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'epoch_{epoch}_mae.png'))
#     plt.close(fig)

def plot_mae(inputs, pred, target, mask, save_path, epoch, model_config):
    # Plot MAE
    image_size = model_config['image_size']
    patch_size = model_config['patch_size']
    image_height, image_width = image_size if type(image_size)==tuple else ast.literal_eval(model_config['image_size']) # (h, w)
    patch_height, patch_width = patch_size if type(patch_size)==tuple else ast.literal_eval(model_config['patch_size']) # (h, w)
    h_num = image_height // patch_height
    w_num = image_width // patch_width
    num_channels = model_config['channels']
    os.makedirs(save_path, exist_ok=True)

    masked_input = torch.zeros_like(inputs[0])
    masked_input = rearrange(masked_input, 'c (h_num ph) (w_num pw) -> (h_num w_num) (ph pw) c', h_num=h_num, w_num=w_num, ph=patch_height, pw=patch_width, c=num_channels)
    masked_input[mask[0]==0] = target[0].unsqueeze(-1)[mask[0]==0]
    masked_input = masked_input.clone().cpu().numpy()
    masked_input = rearrange(masked_input, '(h_num w_num) (ph pw) c -> (h_num ph) (w_num pw) c', h_num=h_num, w_num=w_num, ph=patch_height, pw=patch_width, c=num_channels)

    input_image = inputs[0].clone().cpu().numpy().transpose(1, 2, 0) # [H W C]
    prediction = pred[0].clone().cpu().detach().numpy()
    prediction = rearrange(prediction, '(h_num w_num) (ph pw c) -> (h_num ph) (w_num pw) c', h_num=h_num, w_num=w_num, ph=patch_height, pw=patch_width, c=num_channels)
    # print(f"prediction : {prediction}")
    # normalized_prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))

    target = target[0].clone().cpu().detach().numpy()
    target = rearrange(target, '(h_num w_num) (ph pw c) -> (h_num ph) (w_num pw) c', h_num=h_num, w_num=w_num, ph=patch_height, pw=patch_width, c=num_channels)
    # target = (target - np.min(target)) / (np.max(target) - np.min(target))
    # breakpoint()
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].imshow(input_image)
    axs[0].set_title('Original Patches')
    axs[1].imshow(masked_input)
    axs[1].set_title('Masked Input')
    axs[2].imshow(prediction)
    axs[2].set_title('Predicted Pixel Values')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_mae.png'))
    plt.close(fig)

def train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, train_loader, train_sample, model, revin, optimizer, scheduler, criterion, epoch):
    model.train()
    train_loss = 0
    # plot_sample_reconstruction(model, train_sample, device, model_config, 
    #                         os.path.join(timestamped_file_name, 'train'), epoch)

    # Training step
    train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
    for batch in train_step_pbar:
        optimizer.zero_grad()
        inputs = batch['data'].to(device, non_blocking=True).unsqueeze(1) # [bs x 1 x freq x time]

        loss, pred, target, mask = model(inputs)

        train_loss += loss.item()
        train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})

        # Backward pass
        loss.backward()
        optimizer.step()
    
    if scheduler:
        scheduler.step()

    avg_train_loss = train_loss / len(train_loader)

    # plot_mae(masked_patches, pred_pixel_values, patches, masked_indices, os.path.join(timestamped_file_name, 'train'), epoch)
    plot_mae(inputs, pred, target, mask, os.path.join(timestamped_file_name, 'train'), epoch, model_config)

    # Log to Neptune
    if neptune_enabled:
        run["train/epoch_loss"].log(avg_train_loss, step=epoch)
        run["train/lr"].log(optimizer.param_groups[0]['lr'], step=epoch)

    # plot_sample_reconstruction(model, train_sample, device, model_config,
    #                            os.path.join(timestamped_file_name, 'train'), epoch)

    return avg_train_loss


def val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, val_loader, val_sample, model, revin, criterion, epoch_pbar, epoch, avg_train_loss):
    model.eval()
    val_loss = 0

    val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
    for batch in val_step_pbar:
        with torch.no_grad():
            inputs = batch['data'].to(device, non_blocking=True).unsqueeze(1)
            loss, pred, target, mask = model(inputs)

            val_loss += loss.item()
            val_step_pbar.set_postfix({"Val Loss": val_loss / (val_step_pbar.n + 1)})

    avg_val_loss = val_loss / len(val_loader)

    # plot_mae(masked_patches, pred_pixel_values, patches, masked_indices, os.path.join(timestamped_file_name, 'val'), epoch)
    plot_mae(inputs, pred, target, mask, os.path.join(timestamped_file_name, 'val'), epoch, model_config)

    # Log to Neptune
    if neptune_enabled:
        run["val/epoch_loss"].log(avg_val_loss, step=epoch)

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

def process_masked_patches_2D(input_patches, mask, masked_value=0):
    """
    Applies a mask to 2D patches and rearranges them for model input.
    """
    masked_patches = apply_mask(input_patches, mask, masked_value=masked_value)
    masked_patches = rearrange(masked_patches, 'b c fp tp fs ts -> b c (fp fs) (tp ts)')
    return masked_patches

def process_output_patches_2D(outputs, mask, freq_patches, time_patches, fs_patch, ts_patch, masked_value=0):
    """
    Rearranges the model outputs and applies the inverse mask to reconstruct the masked regions.
    """
    outputs = rearrange(outputs, 'b (fp tp) (fs ts) -> b fp tp fs ts', fp=freq_patches, tp=time_patches, fs=fs_patch, ts=ts_patch).unsqueeze(1)
    predicted_masked_regions = apply_inv_mask(outputs, mask, masked_value=masked_value)
    return predicted_masked_regions

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretraining script for PatchTST.")
    # parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--config', type=str, default='configs/pretrain/scalogram_based/pretrain_MaskedAutoencoderViT_TUAB.yaml', help="Path to the configuration file.")
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
    dataset_name = data_config['dataset_name']
    train_loader, val_loader, _ = get_dataloaders(dataset_name, data_config, model_config)

    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))

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
    
    # Set up optimizer and scheduler
    optimizer = Adam(model.parameters(), 
                     lr=float(train_config['learning_rate']), 
                     weight_decay=float(train_config['weight_decay']))
    
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=train_config['num_epochs'], 
                                  eta_min=float(train_config['min_lr']),
                                  last_epoch=-1)

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
        
        if epoch % val_interval_epochs == 0:
            avg_val_losses = val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                    val_loader, val_sample, model, 
                                    revin, criterion, epoch_pbar, epoch, avg_train_loss)
    
            save_models(config, timestamped_file_name, model, revin, optimizer, scheduler, best_val_loss, num_checkpoints, epoch, avg_train_loss, avg_val_losses)


if __name__ == "__main__":
    main()
