import yaml

class Config:
    def __init__(self, config_file=None):
        # Default configurations
        self.data = {
            'root_path': '/path/to/data',
            'data_path': '',
            'csv_path': '',
            'batch_size': 32,
            'num_workers': 4,
            'seq_len': 128,
            'label_len': 64, # future
            'pred_len': 64, # patch
        }
        self.model = {
            'model_type': 'Revin',
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'output_size': 1,
        }
        self.train = {
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'scheduler_step_size': 10,
            'scheduler_gamma': 0.1,
        }
        self.neptune = {
            'enabled': False,
        }

        # Load from a file if provided
        if config_file:
            self.load_from_file(config_file)

    def load_from_file(self, config_file):
        """Load configurations from a YAML file."""
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.update(configs)

    def update(self, configs):
        """Update configurations with a dictionary."""
        for key, value in configs.items():
            if hasattr(self, key):
                getattr(self, key).update(value)

    def get(self):
        """Return all configurations as a dictionary."""
        return {
            'data': self.data,
            'model': self.model,
            'train': self.train,
            'neptune': self.neptune,
        }

# Example usage
# config = Config(config_file='config.yaml')
# print(config.get())