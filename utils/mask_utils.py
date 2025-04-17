import torch


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