

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from utils.save_model import save_model_to_pt
from utils.directory import ProjectStructure
from distiller import DistillerBridge

def save_sst2_as_csv(csv_path, split="train"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} already exists, skipping.")
        return

    dataset = load_dataset("glue", "sst2", split=split)
    df = pd.DataFrame({
        "text": dataset["sentence"],
        "label": dataset["label"]
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved SST-2 to {csv_path}")

def save_mrpc_as_csv(csv_path, split="train"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} already exists, skipping.")
        return

    dataset = load_dataset("glue", "mrpc", split=split)
    df = pd.DataFrame({
        "text": [f"{s1} [SEP] {s2}" for s1, s2 in zip(dataset["sentence1"], dataset["sentence2"])],
        "label": dataset["label"]
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved MRPC to {csv_path}")

if __name__ == "__main__":
    # Dataset selection: uncomment one
    dataset_name = "sst2"
    dataset_relative_path = './datasets/SST2/train.csv'
    # dataset_name = "mrpc"
    # dataset_relative_path = './datasets/MRPC/train.csv'

    teacher_model_name = "microsoft/deberta-large"
    student_model_name = "microsoft/deberta-base"
    teacher_path = './models/pretrained/deberta-large.pt'
    student_path = './models/pretrained/deberta-base.pt'

    # Save dataset
    if dataset_name == "sst2":
        save_sst2_as_csv(dataset_relative_path)
    else:
        save_mrpc_as_csv(dataset_relative_path)

    # Save teacher model
    os.makedirs(os.path.dirname(teacher_path), exist_ok=True)
    if not os.path.exists(teacher_path):
        print("Saving teacher model...")
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2)
        save_model_to_pt(teacher_model, teacher_path)

    # Save student model
    os.makedirs(os.path.dirname(student_path), exist_ok=True)
    if not os.path.exists(student_path):
        print("Saving student model...")
        student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=2)
        save_model_to_pt(student_model, student_path)

    # Create output path
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("deberta-large-to-base", dataset_name)

    # Run distillation
    bridge = DistillerBridge(
        teacher_path=teacher_path,
        student_path=student_path,
        dataset_path=dataset_relative_path,
        output_path=output_model_path,
        tokenizer_name=student_model_name
    )

    bridge.distill()