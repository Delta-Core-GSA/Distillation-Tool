import os
import torch
import base64
import pandas as pd
from io import BytesIO
from transformers import AutoModelForImageClassification  # Per il modello immagine
from torchvision.datasets import CIFAR10
from distiller import DistillerBridge
from utils.save_model import save_model_to_pt
from utils.TaskDir import TaskDir
from utils.directory import ProjectStructure

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

if __name__ == "__main__":
    # Definizione dei percorsi (～￣▽￣)～
    dataset_relative_path = './datasets/CIFAR10/train.csv'
    models_relative_path = './models/pretrained/vit-base.pt'

    # 1. Salva il dataset CIFAR-10 in CSV se non già presente
    save_cifar10_as_csv(dataset_relative_path)

    # 2. Salva il modello da Hugging Face in formato .pt
    os.makedirs(os.path.dirname(models_relative_path), exist_ok=True)
    model_name = "google/vit-base-patch16-224-in21k"  # Esempio: Vision Transformer
    model = AutoModelForImageClassification.from_pretrained(model_name)
    #Per prendere il modello devi necessariamente fare AutomodelFor<task>
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("vitbase", "cifrar10")



    if not os.path.exists(models_relative_path):
        print("Salvataggio modello Hugging Face...")
        save_model_to_pt(model, models_relative_path)

    # 3. Usa il modello nel processo di distillazione
    bridge = DistillerBridge(
        teacher_path=models_relative_path,
        student_path=models_relative_path,
        dataset_path=dataset_relative_path,
        output_path=output_model_path
    )

    bridge.distill()
