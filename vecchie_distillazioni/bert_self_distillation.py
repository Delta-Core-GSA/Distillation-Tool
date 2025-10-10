import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from utils.save_model import save_model_to_pt
from utils.TaskDir import TaskDir
from utils.directory import ProjectStructure
from distiller import DistillerBridge

def save_sst2_as_csv(csv_path, split="train"):
    """
    Salva il dataset SST-2 in un CSV con due colonne: 'text', 'label'.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    dataset = load_dataset("glue", "sst2", split=split)
    df = pd.DataFrame({
        "text": dataset["sentence"],
        "label": dataset["label"]
    })
    df.to_csv(csv_path, index=False)
    print(f"Salvato SST-2 in {csv_path}")

if __name__ == "__main__":
    # Percorsi
    dataset_relative_path = './datasets/SST2/train.csv'
    models_relative_path = './models/pretrained/bert-base.pt'
    model_name = "bert-base-uncased"

    # 1. Salva il dataset SST-2
    save_sst2_as_csv(dataset_relative_path)

    # 2. Salva il modello teacher (e student) da Hugging Face
    os.makedirs(os.path.dirname(models_relative_path), exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not os.path.exists(models_relative_path):
        print("Salvataggio modello Hugging Face BERT...")
        save_model_to_pt(model, models_relative_path)

    # 3. Percorso di output per la distillazione
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("bertbase", "sst2")

    # 4. Esegui distillazione
    bridge = DistillerBridge(
        teacher_path=models_relative_path,
        student_path=models_relative_path,
        dataset_path=dataset_relative_path,
        output_path=output_model_path,
        tokenizer_name=model_name  # opzionale, se `DistillerBridge` lo usa
    )

    bridge.distill()
