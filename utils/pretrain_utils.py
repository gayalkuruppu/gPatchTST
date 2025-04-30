import os
import torch

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
