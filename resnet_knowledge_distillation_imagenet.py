#!/usr/bin/env python3
# =================== MAIN RESNET MULTI-STUDENT DISTILLATION ===================
# File: main_resnet_multi_distillation.py

import os
import sys
import torch
import torchvision
import json
import time
from datetime import datetime

# Project imports
from utils.directory import ProjectStructure
from utils.save_model import save_model_to_pt
from adapters.dataset_adapter import BaseDatasetAdapter, create_imagenet_mapping_from_train_dir
from distiller import DistillerBridge
from config.constants import BATCH_SIZE

# =================== CONFIGURATION ===================
class DistillationConfig:
    """Configurazione centralizzata per distillazione ResNet"""
    
    def __init__(self):
        # Dataset settings
        self.dataset_dir = "/home/delta-core/Scaricati"
        self.max_samples_per_class = None  # None = tutti, int = limite per testing
        self.max_total_samples = None      # None = tutti, int = limite globale
        
        # Training settings
        self.teacher_epochs = 5
        self.distillation_epochs = 8
        self.batch_size = BATCH_SIZE
        
        # Distillation hyperparameters
        self.temperature = 4.0
        self.alpha = 0.7
        self.learning_rate = 1e-4
        
        # Model settings
        self.teacher_model = "resnet152"
        self.student_models = ["resnet18", "resnet34", "resnet50", "resnet101"]
        
        # Output settings
        self.experiment_name = "resnet_multi_distillation"
        self.verbose = True

# =================== DATASET UTILITIES ===================
class ImageNetPreprocessor:
    """Gestisce preprocessing del dataset ImageNet"""
    
    @staticmethod
    def generate_val_annotations(dataset_dir):
        """Genera val_annotations.txt da LOC_val_solution.csv"""
        import csv
        
        csv_path = os.path.join(dataset_dir, 'ILSVRC/LOC_val_solution.csv')
        output_txt = os.path.join(dataset_dir, 'ILSVRC/val_annotations.txt')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File CSV non trovato: {csv_path}")
        
        print(f"📄 Generazione annotations: {output_txt}")
        with open(csv_path, 'r') as infile, open(output_txt, 'w') as outfile:
            reader = csv.reader(infile)
            next(reader)  # skip header
            for row in reader:
                img_filename, label = row[0], row[1]
                outfile.write(f"{img_filename} {label}\n")
        
        print(f"✅ Annotations create: {output_txt}")
    
    @staticmethod
    def setup_imagenet_mapping(dataset_dir):
        """Crea o carica il mapping ImageNet"""
        raw_train_dir = os.path.join(dataset_dir, 'ILSVRC/Data/CLS-LOC/train')
        mapping_path = os.path.join(dataset_dir, 'imagenet_class_mapping.json')
        
        if not os.path.exists(mapping_path):
            print("📋 Creazione mapping ImageNet...")
            create_imagenet_mapping_from_train_dir(raw_train_dir, mapping_path)
        else:
            print(f"✅ Mapping ImageNet esistente: {mapping_path}")
        
        return mapping_path
    
    @staticmethod
    def generate_distillation_csv(dataset_dir, output_csv_path, mapping_path, config):
        """Genera CSV per distillazione con mapping corretto"""
        from tqdm import tqdm
        import csv
        
        raw_train_dir = os.path.join(dataset_dir, 'ILSVRC/Data/CLS-LOC/train')
        raw_val_dir = os.path.join(dataset_dir, 'ILSVRC/Data/CLS-LOC/val')
        val_labels_file = os.path.join(dataset_dir, 'ILSVRC/val_annotations.txt')
        
        # Carica mapping
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        label_to_idx = mapping_data['label_to_idx']
        
        print(f"📝 Generazione CSV: {output_csv_path}")
        
        total_samples = 0
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'label'])
            
            # Training images
            print("📂 Elaborazione training images...")
            for class_dir in tqdm(os.listdir(raw_train_dir), desc="Train classes"):
                class_path = os.path.join(raw_train_dir, class_dir)
                if not os.path.isdir(class_path) or class_dir not in label_to_idx:
                    continue
                
                img_files = [f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if config.max_samples_per_class:
                    img_files = img_files[:config.max_samples_per_class]
                
                for img_file in img_files:
                    if config.max_total_samples and total_samples >= config.max_total_samples:
                        break
                    
                    img_path = os.path.abspath(os.path.join(class_path, img_file))
                    writer.writerow([img_path, class_dir])
                    total_samples += 1
                
                if config.max_total_samples and total_samples >= config.max_total_samples:
                    break
            
            # Validation images
            if (os.path.exists(val_labels_file) and 
                (not config.max_total_samples or total_samples < config.max_total_samples)):
                
                print("📂 Elaborazione validation images...")
                with open(val_labels_file, 'r') as f:
                    for line in tqdm(f, desc="Val images"):
                        if config.max_total_samples and total_samples >= config.max_total_samples:
                            break
                        
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            filename, label = parts[0], parts[1]
                            img_path = os.path.abspath(os.path.join(raw_val_dir, filename))
                            if os.path.exists(img_path) and label in label_to_idx:
                                writer.writerow([img_path, label])
                                total_samples += 1
        
        print(f"✅ CSV generato: {output_csv_path} ({total_samples} samples)")

# =================== MODEL UTILITIES ===================
class ResNetModelFactory:
    """Factory per creare modelli ResNet"""
    
    MODEL_MAP = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
    }
    
    @classmethod
    def create_model(cls, model_name, num_classes, pretrained=False):
        """Crea modello ResNet con numero di classi specificato"""
        if model_name not in cls.MODEL_MAP:
            raise ValueError(f"Modello {model_name} non supportato. "
                           f"Disponibili: {list(cls.MODEL_MAP.keys())}")
        
        model = cls.MODEL_MAP[model_name](pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    
    @staticmethod
    def count_parameters(model):
        """Conta parametri di un modello"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @classmethod
    def train_teacher(cls, model, dataloader, config, device):
        """Training teacher model"""
        from torch import nn, optim
        from tqdm import tqdm
        
        model.to(device).train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        print(f"🧠 TRAINING TEACHER")
        print(f"📊 Parametri: {cls.count_parameters(model):,}")
        print(f"🎯 Epoche: {config.teacher_epochs}")
        
        for epoch in range(config.teacher_epochs):
            running_loss = 0.0
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.teacher_epochs}") as pbar:
                for batch_idx, (inputs, labels) in enumerate(pbar):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    pbar.set_postfix(loss=running_loss / (batch_idx + 1))
            
            avg_loss = running_loss / len(dataloader)
            print(f"✅ Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# =================== MAIN ORCHESTRATOR ===================
class DistillationPipeline:
    """Pipeline principale per distillazione multi-student"""
    
    def __init__(self, config):
        self.config = config
        self.project = ProjectStructure()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths che verranno inizializzati
        self.dataset_csv = None
        self.mapping_path = None
        self.teacher_path = None
        self.num_classes = None
        
        print(f"🚀 Pipeline inizializzata - Device: {self.device}")
    
    def setup_dataset(self):
        """Prepara dataset ImageNet"""
        print("\n📊 DATASET SETUP")
        print("=" * 30)
        
        # 1. Setup mapping
        self.mapping_path = ImageNetPreprocessor.setup_imagenet_mapping(self.config.dataset_dir)
        
        # 2. Genera annotations se necessario
        val_labels_file = os.path.join(self.config.dataset_dir, 'ILSVRC/val_annotations.txt')
        if not os.path.exists(val_labels_file):
            ImageNetPreprocessor.generate_val_annotations(self.config.dataset_dir)
        
        # 3. Genera CSV per distillazione
        self.dataset_csv = os.path.join(self.config.dataset_dir, "distillation_dataset.csv")
        ImageNetPreprocessor.generate_distillation_csv(
            self.config.dataset_dir, self.dataset_csv, self.mapping_path, self.config)
        
        # 4. Ottieni num_classes
        adapter = BaseDatasetAdapter(
            csv_path=self.dataset_csv,
            imagenet_mapping_path=self.mapping_path,
            max_samples=self.config.max_total_samples
        )
        self.num_classes = adapter.get_num_classes()
        
        print(f"✅ Dataset setup completato - {self.num_classes} classi")
    
    def setup_teacher(self):
        """Crea teacher model (training gestito dal DistillerBridge)"""
        print(f"\n🏫 TEACHER SETUP: {self.config.teacher_model}")
        print("=" * 40)
        
        teacher_dir = "./models/pretrained/"
        os.makedirs(teacher_dir, exist_ok=True)
        self.teacher_path = os.path.join(teacher_dir, 
                                       f"teacher_{self.config.teacher_model}_imagenet.pt")
        
        if not os.path.exists(self.teacher_path):
            print(f"🏗️ Creando teacher model structure...")
            # Crea solo la struttura del modello (training nel DistillerBridge)
            teacher = ResNetModelFactory.create_model(self.config.teacher_model, self.num_classes)
            torch.save(teacher, self.teacher_path)
            print(f"💾 Teacher structure salvata: {self.teacher_path}")
            print(f"ℹ️ Training sarà gestito dal DistillerBridge")
        else:
            print(f"✅ Teacher esistente: {self.teacher_path}")
    
    def create_students(self):
        """Crea tutti i modelli student"""
        print(f"\n🎓 STUDENT SETUP")
        print("=" * 30)
        
        student_paths = {}
        student_dir = "./models/pretrained/"
        
        for student_name in self.config.student_models:
            student_path = os.path.join(student_dir, f"student_{student_name}_imagenet.pt")
            
            if not os.path.exists(student_path):
                print(f"🛠️ Creando {student_name}...")
                student = ResNetModelFactory.create_model(student_name, self.num_classes)
                torch.save(student, student_path)
                print(f"📊 {student_name}: {ResNetModelFactory.count_parameters(student):,} parametri")
            else:
                print(f"✅ {student_name} esistente")
            
            student_paths[student_name] = student_path
        
        return student_paths
    
    def run_distillation(self, student_paths):
        """Esegue distillazione per tutti gli student"""
        print(f"\n🔥 DISTILLATION PHASE")
        print("=" * 30)
        
        results = {}
        
        for student_name, student_path in student_paths.items():
            print(f"\n🧪 DISTILLAZIONE: {self.config.teacher_model} → {student_name}")
            print("-" * 50)
            
            # Setup output folder
            experiment_folder = f"{self.config.teacher_model}_to_{student_name}"
            label = f"a_{self.config.alpha}_t_{self.config.temperature}_imagenet"
            output_path = self.project.create_distillation_folder(experiment_folder, label)
            
            # Configurazione distillazione
            distillation_config = {
                'temperature': self.config.temperature,
                'alpha': self.config.alpha,
                'epochs': self.config.distillation_epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'eval_every': 1,
                'num_classes': self.num_classes
            }
            
            # Esegui distillazione
            try:
                start_time = time.time()
                
                bridge = DistillerBridge(
                    teacher_path=self.teacher_path,
                    student_path=student_path,
                    dataset_path=self.dataset_csv,
                    config=distillation_config,
                    output_path=output_path
                )
                
                final_metrics = bridge.distill()
                distillation_time = time.time() - start_time
                
                results[student_name] = {
                    'success': True,
                    'final_accuracy': final_metrics.get('accuracy', 0.0),
                    'time_minutes': distillation_time / 60,
                    'output_path': output_path
                }
                
                print(f"✅ {student_name} completato!")
                print(f"📊 Accuracy: {final_metrics.get('accuracy', 0.0):.1%}")
                print(f"⏱️ Tempo: {distillation_time/60:.1f}min")
                
            except Exception as e:
                print(f"❌ Errore {student_name}: {e}")
                results[student_name] = {'success': False, 'error': str(e)}
        
        return results
    
    def print_final_report(self, results):
        """Report finale"""
        print(f"\n📋 DISTILLATION REPORT")
        print("=" * 50)
        
        print(f"🏫 Teacher: {self.config.teacher_model}")
        print(f"📊 Dataset: {self.num_classes} classi")
        print(f"🌡️ Temperature: {self.config.temperature}")
        print(f"⚖️ Alpha: {self.config.alpha}")
        
        print(f"\n🎓 RISULTATI:")
        print("-" * 50)
        
        successful = 0
        total_time = 0
        
        for student_name, result in results.items():
            if result['success']:
                print(f"✅ {student_name:<12} | "
                      f"Accuracy: {result['final_accuracy']:.1%} | "
                      f"Time: {result['time_minutes']:.1f}min")
                successful += 1
                total_time += result['time_minutes']
            else:
                print(f"❌ {student_name:<12} | ERROR")
        
        print("-" * 50)
        print(f"📈 Successi: {successful}/{len(results)}")
        print(f"⏱️ Tempo totale: {total_time:.1f} minuti")
        print("\n🎉 DISTILLAZIONE COMPLETATA!")
    
    def run_complete_pipeline(self):
        """Esegue pipeline completa"""
        pipeline_start = time.time()
        
        print("🚀 RESNET MULTI-STUDENT DISTILLATION")
        print("=" * 60)
        
        # Print config
        print(f"📁 Dataset: {self.config.dataset_dir}")
        print(f"🏫 Teacher: {self.config.teacher_model}")
        print(f"🎓 Students: {', '.join(self.config.student_models)}")
        if self.config.max_total_samples:
            print(f"⚠️ Limite samples: {self.config.max_total_samples}")
        
        # Pipeline steps
        self.setup_dataset()
        self.setup_teacher()
        student_paths = self.create_students()
        results = self.run_distillation(student_paths)
        
        # Final report
        pipeline_time = (time.time() - pipeline_start) / 60
        self.print_final_report(results)
        print(f"⏱️ Pipeline totale: {pipeline_time:.1f} minuti")
        
        return results

# =================== MAIN ===================
def main():
    """Main function pulita e semplice"""
    print("🧠 RESNET MULTI-STUDENT KNOWLEDGE DISTILLATION")
    print("=" * 60)
    
    # Configurazione
    config = DistillationConfig()
    
    # 🔧 PERSONALIZZAZIONI PER TESTING (decommentare per test veloce)
    # config.max_samples_per_class = 50   # 50 immagini per classe
    # config.max_total_samples = 2000     # 2000 immagini totali
    # config.teacher_epochs = 2           # Meno epoche per testing
    # config.distillation_epochs = 3
    # config.student_models = ["resnet18"] # Solo un student per testing
    
    # Esegui pipeline
    pipeline = DistillationPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    print(f"✨ DISTILLAZIONE COMPLETATA!")
    return results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️ Interruzione utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)