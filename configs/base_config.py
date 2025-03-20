import torch


class BaseConfig:

    #data
    data_dir = "data/raw/ModelNet10"
    processed_data_dir = "data/processed"

    #model
    model_type = "autoencoder"
    latent_dim = 256

    #Training
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100

    #Hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    #logging
    log_dir = "experiments/logs"
    save_dir = "experiments/checkpoints"