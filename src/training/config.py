class TrainConfig:
    num_epochs: int = 50
    lr: float = 1e-4
    sigma: float = 25.0
    n_attn_head: int = 8
    time_steps: int = 1000
    batch_size: int = 128
    in_channels: int = 1
    n_classes: int = 10
    channels: list = [32, 64, 128, 256]
    t_emb_dim: int = 256
    y_emb_dim: int = 256
    eps: float = 1e-6
    savedir: str = "outputs"