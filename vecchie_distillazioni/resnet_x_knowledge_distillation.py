import os
import torch
import base64
import pandas as pd
from io import BytesIO
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn, optim
from distiller import DistillerBridge
from utils.TaskDir import TaskDir
from utils.directory import ProjectStructure
from utils.save_model import save_model_to_pt
from torchvision.datasets import CIFAR10
from config.constants import BATCH_SIZE
from tqdm import tqdm  # Importa tqdm

# Funzione per salvare CIFAR-10 in formato CSV
def save_cifar10_as_csv(csv_path, split="train"):
    """
    Salva CIFAR-10 in un CSV con immagini codificate in base64 PNG.
    Ogni riga contiene: image (base64), label (int)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    dataset = CIFAR10(root="./data", train=(split == "train"), download=True)
    records = []

    for img_tensor, label in dataset:
        buf = BytesIO()
        img_tensor.save(buf, format="PNG")  # Salviamo direttamente l'immagine come PNG
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        records.append({
            "image": f"data:image/png;base64,{img_base64}",
            "label": label
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Salvato dataset CIFAR10 in {csv_path}")


# Funzione di training del modello teacher con tqdm
#TODO Generalizzare parte di training 
def train_teacher_model(model, trainloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()

    # Barra di progresso per le epoche
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Barra di progresso per i batch
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # Aggiungi il valore della loss alla barra di progresso
                pbar.set_postfix(loss=running_loss/len(pbar))

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader)}")


# Funzione principale
if __name__ == "__main__":
    # Percorsi relativi per il dataset e i modelli
    dataset_relative_path = './datasets/CIFAR10/train.csv'
    models_relative_path = './models/pretrained/student_resnet18.pt'

    # Inizializzazione della struttura del progetto
    project = ProjectStructure()
    output_path = project.create_distillation_folder("resnet152-18", "a_0.5_t_2")
    print (output_path)

    # 1. Salva il dataset CIFAR-10 in CSV se non è già presente
    save_cifar10_as_csv(dataset_relative_path)

    # 2. Carica il modello teacher ResNet-152 e lo allena
    teacher_model_path = './models/pretrained/teacher_resnet152.pt'
    os.makedirs(os.path.dirname(teacher_model_path), exist_ok=True)
    if not os.path.exists(teacher_model_path):
        print("Salvataggio modello teacher (ResNet-152)...")

        teacher = resnet152(pretrained=True)
        teacher.fc = torch.nn.Linear(teacher.fc.in_features, 10)  # Cambiamo l'output per CIFAR-10

        # Setup per il training del teacher
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(teacher.parameters(), lr=0.0001, momentum=0.9)

        # Training del modello teacher con barra di progresso tqdm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #train_teacher_model(teacher, trainloader, criterion, optimizer, device, num_epochs=5)

        # Salvataggio del modello teacher
        torch.save(teacher, teacher_model_path)
        print("Modello teacher (ResNet-152) salvato in teacher_resnet152.pt")

    # 3. Scegli il modello student da una lista di varianti più piccole
    student_model_name = "resnet18"  # Modifica qui per usare resnet34, resnet50, ecc.
    student_model_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152
    }

    if student_model_name in student_model_dict:
        student = student_model_dict[student_model_name](num_classes=10)
    else:
        raise ValueError(f"Modello student {student_model_name} non riconosciuto!")

    # Salva il modello student se non esiste già
    student_model_path = f'./models/pretrained/{student_model_name}_.pt'
    os.makedirs(os.path.dirname(student_model_path), exist_ok=True)
    if not os.path.exists(student_model_path):
        print(f"Salvataggio modello student ({student_model_name})...")
        torch.save(student, student_model_path)
        print(f"Modello student ({student_model_name}) salvato in {student_model_path}")





    # 4. Distillazione: allena lo student usando il teacher
    bridge = DistillerBridge(
        teacher_path=teacher_model_path,
        student_path=student_model_path,
        dataset_path=dataset_relative_path,
        output_path=output_path
    )

    bridge.distill()  # Inizia il processo di distillazione
