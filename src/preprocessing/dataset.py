import csv

import numpy as np
from kmspy import Dataset, DataLoader


def parse_fn(data):
    label = int(data[0])
    image = np.array(data[1:], dtype=np.float32)/255.
    image = image.reshape(1, 28, 28)
    return {"image": image, "label": label}


def make_dataset_from_csv(csv_file):
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)  # header 버리기
        dataset = list(map(parse_fn, reader))
    return Dataset(dataset, no_count_keys=["image"])


if __name__ == "__main__":
    train_file = "./data/mnist_train.csv"
    test_file = "./data/mnist_train.csv"

    train_dataset = make_dataset_from_csv(train_file)
    test_dataset = make_dataset_from_csv(test_file)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8, prefetch_factor=256, num_workers=8, collate_fn=None)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, prefetch_factor=256, num_workers=8, collate_fn=None)


