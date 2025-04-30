import os
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


from get_models import get_pretrain_model
from dataloaders.dataloaders import get_dataloaders
from configs import Config
from utils.utils import init_neptune
from utils.pretrain_utils import load_checkpoint
# Initialize model


# # Dummy dataset
# images = torch.randn(100, 3, 256, 256)  # 100 dummy images
# dataset = TensorDataset(images)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Optimizer
# optimizer = Adam(mae.parameters(), lr=1e-4)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     epoch_loss = 0
#     for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         optimizer.zero_grad()
#         loss = mae(batch[0])
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

# # Save the trained model
# torch.save(mae.encoder.state_dict(), './trained-vit.pt')


def train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, train_loader, train_sample, model, revin, optimizer, scheduler, criterion, epoch):
    model.train()
    train_loss = 0

    # Training step
    train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
    for batch in train_step_pbar:
        # Forward pass
        optimizer.zero_grad()
        inputs = batch['data'].to(device, non_blocking=True).unsqueeze(1)
        if revin:
            inputs = revin(inputs, mode='norm')
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
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

    return avg_train_loss


def val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, val_loader, model, revin, criterion, epoch_pbar, epoch, avg_train_loss):
    model.eval()
    val_loss = 0

    val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
    for batch in val_step_pbar:
        with torch.no_grad():
            inputs = batch['data'].to(device, non_blocking=True).unsqueeze(1)
            if revin:
                inputs = revin(inputs, mode='norm')
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()
            val_step_pbar.set_postfix({"Val Loss": val_loss / (val_step_pbar.n + 1)})

    avg_val_loss = val_loss / len(val_loader)

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
    
    train_configs = config['train']
    # Set up optimizer and scheduler
    optimizer = Adam(model.parameters(), 
                     lr=float(train_configs['learning_rate']), 
                     weight_decay=float(train_configs['weight_decay']))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_configs['step_size'], 
                                                gamma=train_configs['gamma'])

    criterion = torch.nn.MSELoss(reduction='mean') 

    num_epochs = train_configs['num_epochs']
    val_interval_epochs = train_configs['val_interval_epochs']
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
        scheduler = reinitialize_scheduler(scheduler, optimizer, train_configs, start_epoch, len(train_loader))

    # Training loop
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        avg_train_loss = train_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                     train_loader, train_sample, model, revin, 
                                     optimizer, scheduler, criterion, epoch)
        
        if (epoch+1) % val_interval_epochs == 0:
            avg_val_losses = val_epoch(device, model_config, neptune_enabled, run, timestamped_file_name, 
                                    val_loader, model, 
                                    revin, criterion, epoch_pbar, epoch, avg_train_loss)
    
            save_models(config, timestamped_file_name, model, revin, optimizer, scheduler, best_val_loss, num_checkpoints, epoch, avg_train_loss, avg_val_losses)


if __name__ == "__main__":
    main()
