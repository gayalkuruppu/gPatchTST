from data import get_tuh_dataloaders, get_tuab_scalogram_dls, get_celeba_dataloaders

def get_dataloaders(dataset_name, data_config, model_config):
    if dataset_name == 'tuab_timeseries':
        train_loader, val_loader, test_laoder = get_tuh_dataloaders(
        data_config['root_path'],
        data_config['data_path'],
        data_config['csv_path'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config['pin_memory'],
        drop_last=data_config['drop_last'],
        size=[model_config['seq_len'], 
              model_config['target_dim'],
              model_config['patch_length']]
        )

    elif dataset_name == 'tuab_scalogram':
        train_loader, val_loader, test_laoder = get_tuab_scalogram_dls(
        data_config['data_path'],
        data_config['batch_size'],
        data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config['pin_memory'],
        drop_last=data_config['drop_last'],
    )
        
    elif dataset_name == 'celeba':
        train_loader, val_loader, test_laoder = get_celeba_dataloaders(
        data_config['data_path'],
        data_config['batch_size'],
        data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config['pin_memory'],
        drop_last=data_config['drop_last'],
    )        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return train_loader, val_loader, test_laoder
