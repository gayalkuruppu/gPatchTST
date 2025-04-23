import sys
sys.path.append('../PatchTST')

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from data import get_tuab_dataloaders, get_tuab_alpha_dataloaders
from configs import Config
from get_models import get_patchTST_model
from models.patchtst.layers.revin import RevIN
import argparse
import os
from datetime import datetime
import shutil
from utils.mask_utils import create_patches
import neptune
from sklearn.metrics import roc_auc_score

def calculate_auroc(output, target):
    """
    Calculate AUROC for classification.
    """
    output = torch.softmax(output, dim=1).detach().cpu().numpy()[:, 1]
    target = target.detach().cpu().numpy()

    return roc_auc_score(target, output)#, multi_class='ovr')

def calculate_auroc_epoch(model, revin, data_loader, device, patch_len, stride):
    """
    Calculate AUROC for the entire dataset at the end of an epoch.
    """
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            data = batch['past_values'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)

            if revin:
                data = revin(data, mode='norm')
                target = revin(target, mode='norm')

            input_patches, _ = create_patches(data, patch_len, stride)
            output = model(input_patches)

            all_outputs.append(output)
            all_targets.append(target)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return calculate_auroc(all_outputs, all_targets)

def init_neptune(config):
    """
    Initialize a new Neptune run.
    """
    print("Creating a new Neptune run.")
    return neptune.init_run(
        project=config['neptune']['project'], 
        name=config['neptune']['experiment_name'],
        capture_stdout=False,
        capture_stderr=False
    )

def train_epoch(model, revin, train_loader, optimizer, criterion, device, patch_len, stride, head_type):
    """
    Train the linear probe head on the frozen backbone.
    """
    model.train()
    train_loss = 0

    train_step_pbar = tqdm(train_loader, desc="Training", total=len(train_loader), leave=False)
    for batch in train_step_pbar:
        data = batch['past_values'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)
        # target = target * 10000

        if revin:
            data = revin(data, mode='norm')
            target = revin(target, mode='norm')

        input_patches, _ = create_patches(data, patch_len, stride) # [batch_size, num_patches, num_channels, patch_len]

        if head_type == 'regression':
            # for all the channels calculate loss
            for ch in range(input_patches.shape[2]):
                ch_input_patches = input_patches[:, :, ch, :].unsqueeze(2) # [batch_size, num_patches, patch_len, 1]
                ch_target = target[:, ch].unsqueeze(1)

                optimizer.zero_grad()
                ch_output = model(ch_input_patches)
                loss = criterion(ch_output, ch_target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})

        elif head_type == 'classification':
            optimizer.zero_grad()
            output = model(input_patches)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_step_pbar.set_postfix({"Train Loss": train_loss / (train_step_pbar.n + 1)})

        else:
            raise ValueError(f"Unknown head type: {head_type}. Expected 'regression' or 'classification'.")

    return train_loss / len(train_loader)


def val_epoch(model, revin, val_loader, criterion, device, patch_len, stride, head_type):
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

            if head_type == 'regression':
                for ch in range(input_patches.shape[2]):
                    ch_input_patches = input_patches[:, :, ch, :].unsqueeze(2)
                    ch_target = target[:, ch].unsqueeze(1)
                    ch_output = model(ch_input_patches)
                    loss = criterion(ch_output, ch_target)
                    val_loss += loss.item()
                    val_step_pbar.set_postfix({"Val Loss": val_loss / (val_step_pbar.n + 1)})

            elif head_type == 'classification':
                output = model(input_patches)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_step_pbar.set_postfix({"Val Loss": val_loss / (val_step_pbar.n + 1)})
            else:
                raise ValueError(f"Unknown head type: {head_type}. Expected 'regression' or 'classification'.")

    return val_loss / len(val_loader)

def test_epoch(model, revin, test_loader, criterion, device, patch_len, stride, head_type):
    """
    Test the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", total=len(test_loader), leave=False):
            data = batch['past_values'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)

            if revin:
                data = revin(data, mode='norm')
                target = revin(target, mode='norm')

            input_patches, _ = create_patches(data, patch_len, stride)

            if head_type == 'regression':
                for ch in range(input_patches.shape[2]):
                    ch_input_patches = input_patches[:, :, ch, :].unsqueeze(2)
                    ch_target = target[:, ch].unsqueeze(1)
                    ch_output = model(ch_input_patches)
                    loss = criterion(ch_output, ch_target)
                    test_loss += loss.item()

                    all_outputs.append(ch_output)
                    all_targets.append(ch_target)
            elif head_type == 'classification':
                output = model(input_patches)
                loss = criterion(output, target)
                test_loss += loss.item()

                all_outputs.append(output)
                all_targets.append(target)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    test_auroc = None
    if head_type == 'classification':
        test_auroc = calculate_auroc(all_outputs, all_targets)

    return test_loss / len(test_loader), test_auroc

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

def get_dataloaders(data_config, model_config):
    """
    Get the appropriate data loaders based on the configuration.
    """
    if data_config['dataloader']=='tuab_alpha_powers':
        return get_tuab_alpha_dataloaders(
            data_config['root_path'],
            data_config['data_path'],
            data_config['csv_path'],
            alpha_dict_path=data_config['metadata_dict_path'],
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
    else:
        return get_tuab_dataloaders(
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Linear probing or finetuning script for PatchTST.")
    parser.add_argument('--config', type=str, help="Path to the configuration file.", default='configs/linear_probe/tuab_linear_probe_patch_100.yaml')
    parser.add_argument('--checkpoint', type=str, help="Path to the pretrained checkpoint.", default='/home/gayal/ssl-project/gpatchTST/saved_models/pretrain/tuhab_pretrain_tuab_with_cls_token/TUH-101/2025-04-17_21-01-03/checkpoint_epoch_100.pth')
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
    neptune_config = config['neptune']
    neptune_enabled = neptune_config['enabled']
    
    head_type = model_config['head_type']
    if head_type=='regression':
        assert model_config['target_dim'] == 1, "Regression head type requires target_dim to be 1."
    elif head_type=='classification':
        print("Number of classes:", model_config['target_dim'])
        assert model_config['target_dim'] > 0, "Classification head type requires target_dim to be greater than 0."

    # Initialize Neptune
    run = None
    if neptune_enabled:
        run = init_neptune(config)

    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(data_config, model_config)

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
                                head_type=head_type,
                                use_cls_token=model_config['use_cls_token'],
                            ).to(device)

    # Transfer weights
    model = transfer_weights(args.checkpoint, model, exclude_head=True, device=device)
    print(f"Loaded pretrained weights from {args.checkpoint}")


    # Freeze backbone if mode is linearprobe
    if model_config['mode'] == 'linearprobe':
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen for linear probing.")
        model_config['save_path'] = os.path.join(model_config['save_path'], 'linear_probe')
    else:
        print("Backbone trainable for finetuning.")
        model_config['save_path'] = os.path.join(model_config['save_path'], 'finetune')

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

    criterion = get_loss_function(head_type)

    # Model saving path
    if not os.path.exists(model_config['save_path']):
        os.makedirs(model_config['save_path'])

    # Create a folder with the current date and time
    experiment_name, timestamped_file_name = create_experiment_directory(model_config, args.checkpoint, neptune_config, neptune_enabled, run)

    # Save a copy of the configuration file for reproducibility
    config_backup_path = os.path.join(timestamped_file_name, os.path.basename(config_file_path))
    shutil.copy(config_file_path, config_backup_path)
    print(f"Configuration saved to {config_backup_path}")

    # Training loop
    best_val_loss = float('inf')
    best_val_auroc = 0.0
    for epoch in range(train_configs['num_epochs']):
        print(f"Epoch {epoch + 1}/{train_configs['num_epochs']}")

        train_epoch_loss = train_epoch(model, revin, train_loader, optimizer, criterion, device, model_config['patch_length'], model_config['stride'], head_type)
        val_epoch_loss = val_epoch(model, revin, val_loader, criterion, device, model_config['patch_length'], model_config['stride'], head_type)

        print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if head_type == 'classification':
            train_auroc = calculate_auroc_epoch(model, revin, train_loader, device, model_config['patch_length'], model_config['stride'])
            val_auroc = calculate_auroc_epoch(model, revin, val_loader, device, model_config['patch_length'], model_config['stride'])

            print(f"Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f}")

        scheduler.step()

        # Log metrics to Neptune
        if run:
            run["train/epoch_loss"].log(train_epoch_loss, step=epoch)
            run["val/epoch_loss"].log(val_epoch_loss, step=epoch)
            
            if head_type == 'classification':
                run["train/epoch_auroc"].log(train_auroc, step=epoch)
                run["val/epoch_auroc"].log(val_auroc, step=epoch)

        # Save the best model based on validation loss and AUROC
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_path = os.path.join(timestamped_file_name, "best_linear_probe_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model based on validation loss at {best_model_path}.")

    # Test the model and log results
    test_loss, test_auroc = test_epoch(model, revin, test_loader, criterion, device, 
                                       model_config['patch_length'], model_config['stride'], 
                                       head_type)
    print(f"Test Loss: {test_loss:.4f}")
    if test_auroc is not None:
        print(f"Test AUROC: {test_auroc:.4f}")

    # Save test results to a log file
    log_file_path = os.path.join(timestamped_file_name, "test_results.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Experiment Name: {experiment_name}\n")
        log_file.write(f"Test Loss: {test_loss:.4f}\n")
        if test_auroc is not None:
            log_file.write(f"Test AUROC: {test_auroc:.4f}\n")
    print(f"Test results saved to {log_file_path}")

    if run:
        run.stop()

def extract_pretrain_checkpoint_configs(checkpoint):
    # checkpoint = '/home/gayal/ssl-project/gpatchTST/saved_models/pretrain/tuhab_pretrain_tuab_with_cls_token/TUH-101/2025-04-17_21-01-03/checkpoint_epoch_100.pth'
    dirname = os.path.dirname(checkpoint)
    config_file = [f for f in os.listdir(dirname) if f.endswith('.yaml') or f.endswith('.yml')]
    if len(config_file) == 0:
        raise ValueError(f"No config file found in {dirname}")
    elif len(config_file) > 1:
        raise ValueError(f"Multiple config files found in {dirname}: {config_file}")
    else:
        config_file = os.path.join(dirname, config_file[0])
    print(f"Pretraining config file found: {config_file}")
    pretrain_config = Config(config_file=config_file).get()

    # Extract relevant configurations
    seq_len = pretrain_config['model']['seq_len']
    patch_length = pretrain_config['model']['patch_length']
    stride = pretrain_config['model']['stride']

    checkpoint_name = f"pretrain_seq_len_{seq_len}_patch_length_{patch_length}_stride_{stride}"

    return checkpoint_name

def create_experiment_directory(model_config, checkpoint, neptune_config, neptune_enabled, run):
    experiment_name = neptune_config['experiment_name']
    pretrain_chckpnt_name = extract_pretrain_checkpoint_configs(checkpoint)

    if neptune_enabled:
        experiment_id = run["sys/id"].fetch()
        timestamped_file_name = os.path.join(model_config['save_path'], experiment_name, pretrain_chckpnt_name, str(experiment_id), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        timestamped_file_name = os.path.join(model_config['save_path'], experiment_name, pretrain_chckpnt_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if not os.path.exists(timestamped_file_name):
        os.makedirs(timestamped_file_name)
    return experiment_name,timestamped_file_name

def get_loss_function(head_type):
    if head_type == 'regression':
        criterion = nn.MSELoss()
    elif head_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else: #TODO change this to be taken from the config
        raise ValueError(f"Unknown head type: {head_type}. Expected 'regression' or 'classification'.")
    return criterion

if __name__ == "__main__":
    main()
