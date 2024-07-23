class TrainConfig:
    num_epochs: int = 100
    lr: float = 1e-4
    n_attn_head: int = 8
    batch_size: int = 128
    in_channels: int = 1
    n_classes: int = 10
    channels: list = [32, 64, 128, 256]
    t_emb_dim: int = 256
    y_emb_dim: int = 768
    eps: float = 1e-5