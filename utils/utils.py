import neptune

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
