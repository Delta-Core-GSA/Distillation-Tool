import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from tqdm import tqdm
import csv
import json
from utils.directory import ProjectStructure
from utils.save_model import save_model_to_pt
from distiller import DistillerBridge
from config.constants import BATCH_SIZE
# AGGIUNGI QUESTO IMPORT
from adapters.dataset_adapter import BaseDatasetAdapter, create_imagenet_mapping_from_train_dir


# ----------------------------- #
# 📄 Genera val_annotations.txt da LOC_val_solution.csv (MANTENUTO)
# ----------------------------- #
def generate_val_annotations_csv_to_txt(dataset_dir):
    csv_path = os.path.join(dataset_dir, 'ILSVRC/LOC_val_solution.csv')
    output_txt = os.path.join(dataset_dir, 'ILSVRC/val_annotations.txt')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[❌] File CSV non trovato: {csv_path}")
    print(f"[📄] Generazione di {output_txt} da {csv_path}...")
    with open(csv_path, 'r') as infile, open(output_txt, 'w') as outfile:
        reader = csv.reader(infile)
        next(reader)  # salta intestazione
        for row in reader:
            img_filename, label = row[0], row[1]
            outfile.write(f"{img_filename} {label}\n")
    print(f"[✅] File val_annotations.txt creato: {output_txt}")

# ----------------------------- #
# 📦 NUOVA FUNZIONE: Crea mapping ImageNet
# ----------------------------- #
def setup_imagenet_mapping(dataset_dir):
    """
    Crea o carica il mapping ImageNet
    """
    raw_train_dir = os.path.join(dataset_dir, 'ILSVRC/Data/CLS-LOC/train')
    mapping_path = os.path.join(dataset_dir, 'imagenet_class_mapping.json')
    
    if not os.path.exists(mapping_path):
        print("[📋] Creazione mapping ImageNet dalle cartelle di training...")
        create_imagenet_mapping_from_train_dir(raw_train_dir, mapping_path)
    else:
        print(f"[✅] Mapping ImageNet già esistente: {mapping_path}")
    
    return mapping_path

# ----------------------------- #
# 📦 FUNZIONE MIGLIORATA: Genera CSV con mapping corretto
# ----------------------------- #
def generate_distillation_csv_with_mapping(dataset_dir, output_csv_path, mapping_path, max_samples_per_class=None):
    """
    Genera CSV per distillazione con mapping corretto delle classi
    """
    raw_train_dir = os.path.join(dataset_dir, 'ILSVRC/Data/CLS-LOC/train')
    raw_val_dir = os.path.join(dataset_dir, 'ILSVRC/Data/CLS-LOC/val')
    val_labels_file = os.path.join(dataset_dir, 'ILSVRC/val_annotations.txt')
    
    # Carica il mapping
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    label_to_idx = mapping_data['label_to_idx']
    
    print(f"[📝] Generazione CSV per distillazione: {output_csv_path}")
    print(f"[📋] Mapping caricato: {len(label_to_idx)} classi")
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])
        
        # Training images
        print("[📂] Elaborazione training images...")
        for class_dir in tqdm(os.listdir(raw_train_dir), desc="Train classes"):
            class_path = os.path.join(raw_train_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # Verifica che la classe sia nel mapping
            if class_dir not in label_to_idx:
                print(f"[WARNING] Classe {class_dir} non trovata nel mapping, skip...")
                continue
            
            img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if max_samples_per_class:
                img_files = img_files[:max_samples_per_class]
            
            for img_file in img_files:
                img_path = os.path.abspath(os.path.join(class_path, img_file))
                # USA IL NOME DELLA CLASSE (sarà convertito in indice dal dataset adapter)
                writer.writerow([img_path, class_dir])
        
        # Validation images (se esistono)
        if os.path.exists(val_labels_file):
            print("[📂] Elaborazione validation images...")
            with open(val_labels_file, 'r') as f:
                for line in tqdm(f, desc="Val images"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename, label = parts[0], parts[1]
                        img_path = os.path.abspath(os.path.join(raw_val_dir, filename))
                        if os.path.exists(img_path) and label in label_to_idx:
                            writer.writerow([img_path, label])
    
    print(f"[✅] File CSV per distillazione generato: {output_csv_path}")

# ----------------------------- #
# 🔄 FUNZIONE MIGLIORATA: DataLoader con adapter
# ----------------------------- #
def get_imagenet_dataloader_with_adapter(csv_path, mapping_path, max_sample=None):
    """
    Crea dataloader usando il BaseDatasetAdapter migliorato
    """
    print(f"[🔄] Creazione dataloader da CSV: {csv_path}")
    
    # Crea adapter con mapping ImageNet
    adapter = BaseDatasetAdapter(
        csv_path=csv_path,
        imagenet_mapping_path=mapping_path,
        max_samples=max_sample
    )
    
    # Mostra info sul mapping
    adapter.print_mapping_info()
    
    # Ottieni dataloader
    dataloader = adapter.get_dataloader()
    num_classes = adapter.get_num_classes()
    
    print(f"[✅] Dataloader creato - Batch size: {BATCH_SIZE}, Classi: {num_classes}")
    
    return dataloader, num_classes, adapter

# ----------------------------- #
# 🧠 Training teacher a chunk (MANTENUTO)
# ----------------------------- #
def train_teacher_model_chunked(model, trainloader, criterion, optimizer, device, chunk_size=500, num_epochs=5):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        chunk_loss = 0.0
        chunk_counter = 0
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                chunk_loss += loss.item()
                chunk_counter += 1
                if chunk_counter >= chunk_size:
                    avg_chunk_loss = chunk_loss / chunk_counter
                    print(f"    🔹 [Chunk completato] Loss media: {avg_chunk_loss:.4f}")
                    chunk_loss = 0.0
                    chunk_counter = 0
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] completata - Loss media: {running_loss/len(trainloader):.4f}")

# ----------------------------- #
# 🚀 MAIN AGGIORNATO
# ----------------------------- #
if __name__ == "__main__":
    dataset_dir = "/home/delta-core/Scaricati"
    
    # 1. Setup mapping ImageNet
    mapping_path = setup_imagenet_mapping(dataset_dir)
    
    # 2. Genera annotations se necessario
    val_labels_file = os.path.join(dataset_dir, 'ILSVRC/val_annotations.txt')
    if not os.path.exists(val_labels_file):
        generate_val_annotations_csv_to_txt(dataset_dir)
    
    # 3. Genera CSV per distillazione con mapping corretto
    distillation_csv_path = os.path.join(dataset_dir, "distillation_dataset.csv")
    generate_distillation_csv_with_mapping(
        dataset_dir, 
        distillation_csv_path,
        mapping_path  # Limita per testing
    )
    
    # 4. Crea dataloader con adapter migliorato
    dataloader, num_classes, adapter = get_imagenet_dataloader_with_adapter(
        distillation_csv_path,
        mapping_path  # Limita il totale per testing
    )
    
    # 5. Setup modelli
    project = ProjectStructure()
    output_path = project.create_distillation_folder("resnet150-101", "a_0.9_t_2.0_imagenet")
    
    # Teacher
        # Teacher
    teacher_model_path = './models/pretrained/teacher_resnet150_imagenet.pt'
    os.makedirs(os.path.dirname(teacher_model_path), exist_ok=True)

    if not os.path.exists(teacher_model_path):
        print("[🚀] Creazione e training teacher da zero (ResNet-152 come placeholder per ResNet-150)...")
        teacher = torchvision.models.resnet152(pretrained=False)
        teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # Esegui il training
        train_teacher_model_chunked(
            model=teacher,
            trainloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            chunk_size=500,    # oppure personalizza
            num_epochs=5       # oppure personalizza
        )

        # Salva il modello dopo il training
        save_model_to_pt(teacher, teacher_model_path)
        print(f"[💾] Teacher addestrato e salvato in: {teacher_model_path}")
    else:
        print(f"[✔] Teacher già addestrato trovato in: {teacher_model_path}")

    
    # Student
student_model_dict = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101
}

for student_model_name, model_fn in student_model_dict.items():
    print(f"\n[🧪] Inizio distillazione per: {student_model_name}")
    
    student_model_path = f'./models/pretrained/{student_model_name}_imagenet.pt'
    
    if not os.path.exists(student_model_path):
        print(f"[🛠️] Creazione student: {student_model_name}")
        student = model_fn(pretrained=False, num_classes=num_classes)
        torch.save(student, student_model_path)
        print(f"[💾] Student salvato: {student_model_path}")
    else:
        print(f"[✔] Student già esistente: {student_model_path}")

    # Crea una cartella di output separata per ogni modello
    model_output_path = project.create_distillation_folder(
        f"resnet150-{student_model_name}", "a_0.9_t_2.0_imagenet"
    )

    # Salva mapping nella cartella specifica
    mapping_output_path = os.path.join(model_output_path, "class_mapping.json")
    adapter.save_mapping(mapping_output_path)

    # Avvia la distillazione
    print("[🔥] Avvio distillazione...")
    bridge = DistillerBridge(
        teacher_path=teacher_model_path,
        student_path=student_model_path,
        dataset_path=distillation_csv_path,
        distillation_strategy="chunked",
        output_path=model_output_path
    )

    bridge.distill()
