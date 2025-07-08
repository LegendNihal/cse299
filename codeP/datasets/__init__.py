def get_dataset(dataset_name, model_name=None):
    if dataset_name == "CIFAR10":
        if(model_name == "cvae"):
            from .cvae_dataset import load_data
            return load_data()
        else:
            from .dcgan_dataset import load_data
            return load_data()
    elif dataset_name == "mnist":
        from .cgan_dataset import load_data
        return load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
