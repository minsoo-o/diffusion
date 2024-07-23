from kmspy import DataLoader

from preprocessing.dataset import make_dataset_from_csv
from training.config import TrainConfig
from training.trainer import Trainer


if __name__ == "__main__":
    config = TrainConfig()

    train_file = "./data/mnist_train.csv"
    test_file = "./data/mnist_train.csv"

    train_dataset = make_dataset_from_csv(train_file)
    test_dataset = make_dataset_from_csv(test_file)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, prefetch_factor=256, num_workers=8, collate_fn=None, format="torch")
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, prefetch_factor=256, num_workers=8, collate_fn=None, format="torch")

    trainer = Trainer(config)
    trainer.train(train_dataloader)