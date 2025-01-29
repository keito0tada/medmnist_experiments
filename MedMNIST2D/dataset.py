import medmnist
from medmnist import INFO
import torchvision
import torch
import numpy as np
import pickle


def main(data_flag, size, index, rate=None):
    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])

    data_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = DataClass(
        split="train", transform=data_transform, download=True, size=size
    )
    val_dataset = DataClass(
        split="val", transform=data_transform, download=True, size=size
    )
    test_dataset = DataClass(
        split="test", transform=data_transform, download=True, size=size
    )

    # train_data_x = []
    # train_data_y = []
    # for X, y in train_dataset:
    #     train_data_x.append(X)
    #     train_data_y.append(y[0])

    # val_data_x = []
    # val_data_y = []
    # for X, y in val_dataset:
    #     val_data_x.append(X)
    #     val_data_y.append(y[0])

    # test_data_x = []
    # test_data_y = []
    # for X, y in test_dataset:
    #     test_data_x.append(X)
    #     test_data_y.append(y[0])

    # train_dataset = torch.utils.data.TensorDataset(
    #     torch.stack(train_data_x), torch.tensor(train_data_y, dtype=torch.int64)
    # )
    # val_dataset = torch.utils.data.TensorDataset(
    #     torch.stack(val_data_x), torch.tensor(val_data_y, dtype=torch.int64)
    # )
    # test_dataset = torch.utils.data.TensorDataset(
    #     torch.stack(test_data_x), torch.tensor(test_data_y, dtype=torch.int64)
    # )

    if rate is not None:
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        with open(
            f"/data1/keito/bachelor/dataset/unlearn_indices/{data_flag}_{rate}_{index}.pkl",
            "wb",
        ) as f:
            pickle.dump(
                (
                    indices[: int(len(train_dataset) * rate)],
                    indices[int(len(train_dataset) * rate) :],
                ),
                f,
            )

        # forget_train_dataset = torch.utils.data.Subset(train_dataset, indices[: int(len(train_dataset) * rate)])
        # retain_train_dataset = torch.utils.data.Subset(train_dataset, indices[int(len(train_dataset) * rate):])

        # forget_train_dataset, retain_train_dataset = torch.utils.data.random_split(
        #     train_dataset,
        #     [
        #         int(len(train_dataset) * rate),
        #         len(train_dataset) - int(len(train_dataset) * rate),
        #     ],
        # )
        # torch.save(
        #     (retain_train_dataset, forget_train_dataset),
        #     f"/data1/keito/bachelor/dataset/{data_flag}_dataset_{rate}_{index}.pt",
        # )
    else:
        # torch.save(
        #     (train_dataset, val_dataset, test_dataset),
        #     f"/data1/keito/bachelor/dataset/{data_flag}_dataset.pt",
        # )
        raise NotImplementedError


DATA_FLAGS = [
    "pathmnist",
    # "chestmnist",
    # "dermamnist",
    "octmnist",
    # "pneumoniamnist",
    # "retinamnist",
    # "breastmnist",
    # "bloodmnist",
    "tissuemnist",
    # "organamnist",
    # "organcmnist",
    # "organsmnist",
]
for data_flag in DATA_FLAGS:
    for i in range(10):
        main(data_flag, 28, i, 0.3)
        main(data_flag, 28, i, 0.5)
