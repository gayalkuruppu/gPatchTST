import sys
sys.path.append('../PatchTST')

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from data import get_tuab_dataloaders
from configs import Config
from get_models import get_patchTST_model
from models.patchtst.layers.revin import RevIN
import argparse
import os
from datetime import datetime
import shutil
from utils.mask_utils import create_patches, create_mask, apply_mask

def train_linear_probe(model, revin, train_loader, optimizer, criterion, device, patch_len, stride):
    """
    Train the linear probe head on the frozen backbone.
    """
    model.train()
    train_loss = 0

    train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
    for batch in train_step_pbar:
        data = batch['past_values'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)

        if revin:
            data = revin(data, mode='norm')
            target = revin(target, mode='norm')

        input_patches, _ = create_patches(data, patch_len, stride)

        optimizer.zero_grad()
        output = model(input_patches)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})

    return train_loss / len(train_loader)


def validate_linear_probe(model, revin, val_loader, criterion, device, patch_len, stride):
    """
    Validate the linear probe head on the frozen backbone.
    """
    model.eval()
    val_loss = 0

    val_step_pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), leave=False)
    with torch.no_grad():
        for batch in val_step_pbar:
            data = batch['past_values'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)

            if revin:
                data = revin(data, mode='norm')
                target = revin(target, mode='norm')

            input_patches, _ = create_patches(data, patch_len, stride)
            output = model(input_patches)
            loss = criterion(output, target)
            val_loss += loss.item()
            val_step_pbar.set_postfix({"Val Loss": val_loss / (val_step_pbar.n + 1)})

    return val_loss / len(val_loader)

def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    """
    Transfer weights from a checkpoint to the model, optionally excluding the head.
    """
    new_state_dict = torch.load(weights_path, map_location=device)['model_state_dict']
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():
        if exclude_head and 'head' in name:
            continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'Check unmatched layers: {unmatched_layers}')
        else:
            print(f"Weights from {weights_path} successfully transferred!")
    model = model.to(device)
    return model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Linear probing or finetuning script for PatchTST.")
    parser.add_argument('--config', type=str, help="Path to the configuration file.", default='configs/tuab_linear_probe_patch_100.yaml')
    parser.add_argument('--checkpoint', type=str, help="Path to the pretrained checkpoint.", default='/home/gayal/ssl-project/gpatchTST/saved_models/pretrain/tuhab_pretrain_tuab_with_cls_token/TUH-101/2025-04-17_21-01-03/checkpoint_epoch_100.pth')
    parser.add_argument('--mode', type=str, choices=['linearprobe', 'finetune'], default='linearprobe', help="Choose between linear probing or finetuning.")
    args = parser.parse_args()

    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config_file_path = args.config
    config = Config(config_file=config_file_path).get()

    data_config = config['data']
    model_config = config['model']
    train_configs = config['train']

    # Create data loaders
    train_loader, val_loader, _ = get_tuab_dataloaders(
        data_config['root_path'],
        data_config['data_path'],
        data_config['csv_path'],
        metadata_csv_path=data_config['metadata_csv_path'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config['pin_memory'],
        drop_last=data_config['drop_last'],
        size=[model_config['seq_len'], 
              model_config['target_dim'],
              model_config['patch_length']],
        seed=data_config['dl_seed'],
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

    # Transfer weights
    model = transfer_weights(args.checkpoint, model, exclude_head=True, device=device)
    print(f"Loaded pretrained weights from {args.checkpoint}")

    # Freeze backbone if mode is linearprobe
    if args.mode == 'linearprobe':
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen for linear probing.")
    else:
        print("Backbone trainable for finetuning.")

    # Initialize RevIN
    if model_config['revin']:
        revin = RevIN(data_config['n_vars'], 
                      float(model_config['revin_eps']),
                      bool(model_config['revin_affine'])).to(device)
    else:
        revin = None

    # Set up optimizer and scheduler
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=float(train_configs['learning_rate']), 
                     weight_decay=float(train_configs['weight_decay']))

    scheduler = OneCycleLR(optimizer,
                           max_lr=float(train_configs['learning_rate']),
                           epochs=train_configs['num_epochs'],
                           steps_per_epoch=len(train_loader),
                           pct_start=0.3,
                           anneal_strategy='cos',
                           cycle_momentum=True,
                           base_momentum=0.85,
                           max_momentum=0.95,
                           div_factor=25)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(train_configs['num_epochs']):
        print(f"Epoch {epoch + 1}/{train_configs['num_epochs']}")

        train_loss = train_linear_probe(model, revin, train_loader, optimizer, criterion, device, model_config['patch_length'], model_config['stride'])
        val_loss = validate_linear_probe(model, revin, val_loader, criterion, device, model_config['patch_length'], model_config['stride'])

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_linear_probe_model.pth")
            print("Saved best model.")

if __name__ == "__main__":
    main()
