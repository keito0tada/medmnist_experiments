import pickle
import os
import time

import torch
import torchvision
from torcheval.metrics.functional import multiclass_accuracy

import medmnist

from models import ResNet18
from SVC_MIA import get_mia_efficiency

def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (torch.nn.functional.kl_div(p, m, reduction='sum') + torch.nn.functional.kl_div(q, m, reduction='sum'))

def dist_model_output(data_loader, task, model1, model2, device):
    model1 = model1.to(device)
    model1.eval()
    model2 = model2.to(device)
    model2.eval()

    outputs1 = []
    outputs2 = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            output1 = model1(inputs.to(device))
            output2 = model2(inputs.to(device))

            if task == "multi-label, binary-class":
                m = torch.nn.Sigmoid()
                outputs1.append(m(output1).to(device))
                outputs2.append(m(output2).to(device))
            else:
                m = torch.nn.Softmax(dim=1)

            outputs1.append(m(output1).to(device))
            outputs2.append(m(output2).to(device))
    
    return js_div(torch.cat(outputs1), torch.cat(outputs2)).item()


def dist_model_parameter(model1, model2, device):
    dist = torch.tensor(0.0, device=device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    for param in model1.state_dict():
        if 'weight' in param or 'bias' in param:
            diff = torch.subtract(model1.state_dict()[param], model2.state_dict()[param]) 
            pow_diff = torch.pow(diff, 2)
            dist += torch.sum(pow_diff).item()
    return torch.sqrt(dist).item()

def test(model, data_loader, task, criterion, num_classes, device):
    model = model.to(device)
    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)
    targets_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = torch.nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = torch.nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            targets_score = torch.cat((targets_score, targets), 0)

        y_score = y_score.detach().cpu()
        targets_score = torch.squeeze(targets_score.detach().cpu())
        print(y_score.shape)
        print(targets_score.shape)
        acc = multiclass_accuracy(y_score, targets_score, num_classes=num_classes)

        test_loss = sum(total_loss) / len(total_loss)

        return acc


def load_forget_retain_dataset(train_dataset, indices_path):
    with open(indices_path, "rb") as f:
        forget_indices, retain_indices = pickle.load(f)

    forget_dataset = torch.utils.data.Subset(train_dataset, forget_indices)
    retain_dataset = torch.utils.data.Subset(train_dataset, retain_indices)

    return forget_dataset, retain_dataset


def load_forget_retain_loader(
    train_dataset, indices_path, batch_size=128, num_workers=2
):
    forget_dataset, retain_dataset = load_forget_retain_dataset(
        train_dataset, indices_path
    )

    forget_loader = torch.utils.data.DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return forget_loader, retain_loader


def load_dataset(data_flag, size=28):
    info = medmnist.INFO[data_flag]
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

    return train_dataset, val_dataset, test_dataset


def eval(
    retrain_dir_path,
    target_model_path,
    retrain_model_path,
    label,
    n_channels,
    n_classes,
    task,
    criterion,
    val_loader,
    test_loader,
    forget_loader,
    retain_loader,
    mia_forget_dataset,
    mia_retain_dataset,
    mia_val_dataset,
    mia_test_dataset,
):
    target_model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    target_model.load_state_dict(torch.load(target_model_path))
    retrain_model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    retrain_model.load_state_dict(torch.load(retrain_model_path))

    target_result = {
        # "val": test(target_model, val_loader, task, criterion, n_classes, "cuda"),
        "test": test(target_model, test_loader, task, criterion, n_classes, "cuda"),
        "forget": test(target_model, forget_loader, task, criterion, n_classes, "cuda"),
        "retain": test(target_model, retain_loader, task, criterion, n_classes, "cuda"),
        # "mia_val": 100
        # * get_mia_efficiency(
        #     mia_forget_dataset, mia_retain_dataset, mia_val_dataset, target_model
        # ),
        "mia_test": 100
        * get_mia_efficiency(
            mia_forget_dataset, mia_retain_dataset, mia_test_dataset, target_model
        ),
        "weight_dist": dist_model_parameter(target_model, retrain_model, device="cuda"),
        'output_dist': dist_model_output(test_loader, task, target_model, retrain_model, device="cuda")
    }
    print(target_result)
    with open(f"{retrain_dir_path}/eval.csv", "a") as f:
        f.write(
            # f'{label},{target_result["val"]},{target_result["test"]},{target_result["forget"]},{target_result["retain"]},{target_result["mia_val"]},{target_result["mia_test"]}\n'
            f'{label},{target_result["test"]},{target_result["forget"]},{target_result["retain"]},{target_result["mia_test"]},{target_result["weight_dist"]}\n'
        )


def main2(data_flag, rate, index, batch_size=128, num_workers=2):
    start_time = time.perf_counter()
    info = medmnist.INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    indices_path = (
        f"/data1/keito/bachelor/dataset/unlearn_indices/{data_flag}_{rate}_{index}.pkl"
    )

    train_dataset, val_dataset, test_dataset = load_dataset(data_flag)
    forget_dataset, retain_dataset = load_forget_retain_dataset(
        train_dataset, indices_path
    )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    # )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    forget_data_x = []
    forget_data_y = []
    for X, y in forget_dataset:
        forget_data_x.append(X)
        forget_data_y.append(y[0])

    retain_data_x = []
    retain_data_y = []
    for X, y in retain_dataset:
        retain_data_x.append(X)
        retain_data_y.append(y[0])

    val_data_x = []
    val_data_y = []
    for X, y in val_dataset:
        val_data_x.append(X)
        val_data_y.append(y[0])

    test_data_x = []
    test_data_y = []
    for X, y in test_dataset:
        test_data_x.append(X)
        test_data_y.append(y[0])

    mia_forget_dataset = torch.utils.data.TensorDataset(
        torch.stack(forget_data_x), torch.tensor(forget_data_y, dtype=torch.int64)
    )
    mia_retain_dataset = torch.utils.data.TensorDataset(
        torch.stack(retain_data_x), torch.tensor(retain_data_y, dtype=torch.int64)
    )
    mia_val_dataset = torch.utils.data.TensorDataset(
        torch.stack(val_data_x), torch.tensor(val_data_y, dtype=torch.int64)
    )
    mia_test_dataset = torch.utils.data.TensorDataset(
        torch.stack(test_data_x), torch.tensor(test_data_y, dtype=torch.int64)
    )

    if task == "multi-label, binary-class":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    target_model_path = f"/data1/keito/bachelor/model/{data_flag}/target_{index}/best_model.pth"
    retrain_dir_path = f"/data1/keito/bachelor/model/{data_flag}/retrain_{rate}_{index}"
    eval(
        retrain_dir_path,
        target_model_path,
        os.path.join(retrain_dir_path, "best_model.pth"),
        "target",
        n_channels,
        n_classes,
        task,
        criterion,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        mia_forget_dataset,
        mia_retain_dataset,
        mia_val_dataset,
        mia_test_dataset,
    )
    eval(
        retrain_dir_path,
        os.path.join(retrain_dir_path, "best_model.pth"),
        os.path.join(retrain_dir_path, "best_model.pth"),
        "retain",
        n_channels,
        n_classes,
        task,
        criterion,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        mia_forget_dataset,
        mia_retain_dataset,
        mia_val_dataset,
        mia_test_dataset,
    )
    # eval(
    #     retrain_dir_path,
    #     os.path.join(retrain_dir_path, "conmu_2.pth"),
    #     "conmu_2",
    #     n_channels,
    #     n_classes,
    #     task,
    #     criterion,
    #     val_loader,
    #     test_loader,
    #     forget_loader,
    #     retain_loader,
    #     mia_forget_dataset,
    #     mia_retain_dataset,
    #     mia_val_dataset,
    #     mia_test_dataset,
    # )
    # eval(
    #     retrain_dir_path,
    #     os.path.join(retrain_dir_path, "conmu_5.pth"),
    #     "conmu_5",
    #     n_channels,
    #     n_classes,
    #     task,
    #     criterion,
    #     val_loader,
    #     test_loader,
    #     forget_loader,
    #     retain_loader,
    #     mia_forget_dataset,
    #     mia_retain_dataset,
    #     mia_val_dataset,
    #     mia_test_dataset,
    # )
    for i in range(5, 51, 5):
        eval(
            retrain_dir_path,
            os.path.join(retrain_dir_path, f"RL_{i}_model.pth"),
            os.path.join(retrain_dir_path, "best_model.pth"),
            f"RL_{i}",
            n_channels,
            n_classes,
            task,
            criterion,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
            mia_forget_dataset,
            mia_retain_dataset,
            mia_val_dataset,
            mia_test_dataset,
        ) 
        eval(
            retrain_dir_path,
            os.path.join(retrain_dir_path, f"saliency/RL_{i}_0.7_model.pth"),
            os.path.join(retrain_dir_path, "best_model.pth"),
            f"RL_{i}_sal",
            n_channels,
            n_classes,
            task,
            criterion,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
            mia_forget_dataset,
            mia_retain_dataset,
            mia_val_dataset,
            mia_test_dataset,
        ) 
    # eval(
    #     retrain_dir_path,
    #     os.path.join(retrain_dir_path, "RL_10_model.pth"),
    #     os.path.join(retrain_dir_path, "best_model.pth"),
    #     "RL_10",
    #     n_channels,
    #     n_classes,
    #     task,
    #     criterion,
    #     val_loader,
    #     test_loader,
    #     forget_loader,
    #     retain_loader,
    #     mia_forget_dataset,
    #     mia_retain_dataset,
    #     mia_val_dataset,
    #     mia_test_dataset,
    # )
    # eval(
    #     retrain_dir_path,
    #     os.path.join(retrain_dir_path, "RL_20_model.pth"),
    #     os.path.join(retrain_dir_path, "best_model.pth"),
    #     "RL_20",
    #     n_channels,
    #     n_classes,
    #     task,
    #     criterion,
    #     val_loader,
    #     test_loader,
    #     forget_loader,
    #     retain_loader,
    #     mia_forget_dataset,
    #     mia_retain_dataset,
    #     mia_val_dataset,
    #     mia_test_dataset,
    # )
    # eval(
    #     retrain_dir_path,
    #     os.path.join(retrain_dir_path, "saliency/RL_10_0.7_model.pth"),
    #     os.path.join(retrain_dir_path, "best_model.pth"),
    #     "RL_10_sal",
    #     n_channels,
    #     n_classes,
    #     task,
    #     criterion,
    #     val_loader,
    #     test_loader,
    #     forget_loader,
    #     retain_loader,
    #     mia_forget_dataset,
    #     mia_retain_dataset,
    #     mia_val_dataset,
    #     mia_test_dataset,
    # )
    # eval(
    #     retrain_dir_path,
    #     os.path.join(retrain_dir_path, "saliency/RL_20_0.7_model.pth"),
    #     os.path.join(retrain_dir_path, "best_model.pth"),
    #     "RL_20_sal",
    #     n_channels,
    #     n_classes,
    #     task,
    #     criterion,
    #     val_loader,
    #     test_loader,
    #     forget_loader,
    #     retain_loader,
    #     mia_forget_dataset,
    #     mia_retain_dataset,
    #     mia_val_dataset,
    #     mia_test_dataset,
    # )
    print(f"time: {time.perf_counter() - start_time}")


def main(model_path, data_flag, rate=None, index=None):
    info = medmnist.INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    model.load_state_dict(torch.load(model_path))

    train_dataset, val_dataset, test_dataset = load_dataset(data_flag)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    if task == "multi-label, binary-class":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print(f'train: {test(model, train_loader, task, criterion, n_classes, "cuda")}')
    print(f'val: {test(model, val_loader, task, criterion, n_classes, "cuda")}')
    print(f'test: {test(model, test_loader, task, criterion, n_classes, "cuda")}')

    if rate is not None:
        indices_path = f"/data1/keito/bachelor/dataset/unlearn_indices/{data_flag}_{rate}_{index}.pkl"
        forget_loader, retain_loader = load_forget_retain_loader(
            train_dataset, indices_path
        )
        print(
            f'forget: {test(model, forget_loader, task, criterion, n_classes, "cuda")}'
        )
        print(
            f'retain: {test(model, retain_loader, task, criterion, n_classes, "cuda")}'
        )
        print(
            f"mia(val): {100 * get_mia_efficiency(forget_loader.dataset, retain_loader.dataset, val_dataset, model)}"
        )
        print(
            f"mia(test): {100 * get_mia_efficiency(forget_loader.dataset, retain_loader.dataset, test_dataset, model)}"
        )


print('eval')
main2("pathmnist", 0.1, 0)
main2("pathmnist", 0.3, 0)
main2("pathmnist", 0.5, 0)
main2("octmnist", 0.1, 0)
main2("octmnist", 0.3, 0)
main2("octmnist", 0.5, 0)
main2("tissuemnist", 0.1, 0)
main2("tissuemnist", 0.3, 0)
main2("tissuemnist", 0.5, 0)

# main2("tissuemnist", 0.5, 0)
# main("/data1/keito/bachelor/model/pathmnist/target/best_model.pth", "pathmnist")
# main(
#     "/data1/keito/bachelor/model/pathmnist/retrain_0.1_0/best_model.pth",
#     "pathmnist",
#     0.1,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/pathmnist/retrain_0.1_0/saliency/RLmodel.pth",
#     "pathmnist",
#     0.1,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/pathmnist/retrain_0.3_0/best_model.pth",
#     "pathmnist",
#     0.3,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/pathmnist/retrain_0.5_0/best_model.pth",
#     "pathmnist",
#     0.5,
#     0,
# )
# main("/data1/keito/bachelor/model/octmnist/target/best_model.pth", "octmnist")
# main(
#     "/data1/keito/bachelor/model/octmnist/retrain_0.1_0/best_model.pth",
#     "octmnist",
#     0.1,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/octmnist/retrain_0.3_0/best_model.pth",
#     "octmnist",
#     0.3,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/octmnist/retrain_0.5_0/best_model.pth",
#     "octmnist",
#     0.5,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/pathmnist/retrain_0.3_0/best_model.pth",
#     "pathmnist",
#     0.3,
#     0,
# )
# main(
#     "/data1/keito/bachelor/model/pathmnist/retrain_0.5_0/best_model.pth",
#     "pathmnist",
#     0.5,
#     0,
# )
# main('/data1/keito/bachelor/model/pathmnist/retrain_0.1_0/conmu_{rate}_{index}.pth', 'pathmnist', 0.1, 0)
