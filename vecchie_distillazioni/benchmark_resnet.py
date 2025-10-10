import torch
import time
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from utils.energy_mesurement_wrapper import with_global_emission_tracking, global_tracker, cumulative_energy, cumulative_energy_j, runtimes
from utils.flops_tracker import compute_flops_and_macs
from utils.custom_logger import CustomLogger

@with_global_emission_tracking
def evaluate(model, dataloader, device, role):
    model.eval()
    correct = total = 0
    total_time = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)
            end = time.time()
            total_time += (end - start)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_time = total_time / len(dataloader)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(all_labels, all_preds, multi_class='ovo', average='weighted')
    except ValueError:
        roc_auc = float('nan')

    return accuracy, avg_time, precision, recall, f1, roc_auc

def load_dataset(name, batch_size=64):
    if name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    if name == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif name == "CIFAR100":
        dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif name == "SVHN":
        dataset = datasets.SVHN(root="./data", split='test', download=True, transform=transform)
    elif name == "FashionMNIST":
        dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset non supportato.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(name='resnet18'):
    if name == 'resnet18':
        model = models.resnet18(weights=None)
    elif name == 'resnet34':
        model = models.resnet34(weights=None)
    else:
        raise ValueError("Unsupported model")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

student = get_model('resnet18').to(device) 
student_weights = torch.load("./distilled_student_a_07_t_2/student_state_dict.pth", map_location=device)
student.load_state_dict(student_weights)

teacher = get_model('resnet34').to(device)
pretrained_student = get_model('resnet18').to(device)

datasets_to_test = ["CIFAR10", "CIFAR100", "SVHN", "FashionMNIST"]

benchmark_logger = CustomLogger("benchmark", "txt")

print("\n=== BENCHMARK COMPARATIVO ===\n")
for ds_name in datasets_to_test:
    benchmark_logger.log_value(f"--- Dataset: {ds_name} ---")
    dataloader = load_dataset(ds_name)

    num_classes = len(dataloader.dataset.classes) if hasattr(dataloader.dataset, 'classes') else 10


    acc_s, t_s, prec_s, rec_s, f1_s, auc_s = evaluate(student, dataloader, device, "student")
    acc_sp, t_sp, prec_sp, rec_sp, f1_sp, auc_sp = evaluate(pretrained_student, dataloader, device, "pretrained_student")
    acc_t, t_t, prec_t, rec_t, f1_t, auc_t = evaluate(teacher, dataloader, device, "teacher")

    benchmark_logger.log_value(f"Student  | Acc: {acc_s:.4f} | Prec: {prec_s:.4f} | Rec: {rec_s:.4f} | F1: {f1_s:.4f} | AUC: {auc_s:.4f} | Time/batch: {t_s:.4f}s | Params: {count_params(student)/1e6:.2f}M")
    benchmark_logger.log_value(f"Pretrained Student  | Acc: {acc_sp:.4f} | Prec: {prec_sp:.4f} | Rec: {rec_sp:.4f} | F1: {f1_sp:.4f} | AUC: {auc_sp:.4f} | Time/batch: {t_sp:.4f}s | Params: {count_params(pretrained_student)/1e6:.2f}M")
    benchmark_logger.log_value(f"Teacher  | Acc: {acc_t:.4f} | Prec: {prec_t:.4f} | Rec: {rec_t:.4f} | F1: {f1_t:.4f} | AUC: {auc_t:.4f} | Time/batch: {t_t:.4f}s | Params: {count_params(teacher)/1e6:.2f}M")
    benchmark_logger.log_value()

global_emissions = global_tracker.stop()

energy_logger = CustomLogger("energy_measurements", "txt")

energy_logger.log_value(value="Teacher FLOPS MACS")
flops, macs = compute_flops_and_macs(teacher, (3, 224, 224))
energy_logger.log_value(value=f"FLOPS: {flops}| MACS: {macs}")

energy_logger.log_value(value="\nStudent FLOPS MACS")
flops, macs = compute_flops_and_macs(student, (3, 224, 224))
energy_logger.log_value(value=f"FLOPS: {flops}| MACS: {macs}")

energy_logger.log_value(value="\nStudent Pretrained FLOPS MACS")
flops, macs = compute_flops_and_macs(pretrained_student, (3, 224, 224))
energy_logger.log_value(value=f"FLOPS: {flops}| MACS: {macs}")


energy_logger.log_value(value="\n=== CONSUMI CUMULATIVI PER ROLE ===")
for role, energy in cumulative_energy.items():
    energy_logger.log_value(f"{role.capitalize()}: {energy:.6f} kWh")

energy_logger.log_value(value="\n=== CONSUMI CUMULATIVI PER ROLE [J] ===")
for role, energy in cumulative_energy_j.items():
    energy_logger.log_value(value=f"{role.capitalize()}: {energy:.6f} kWh")

energy_logger.log_value(value="\n=== TEMPI DI ESECUZIONE ===")
for role, energy in cumulative_energy_j.items():
    energy_logger.log_value(value=f"{role.capitalize()}: {energy:.6f} s")
