import os
import torch
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from utils.save_model import save_model
from utils.directory import ProjectStructure
from distiller import DistillerBridge
from PIL import Image
import numpy as np

def count_classes(csv_path):
    """Conta velocemente le classi dal CSV"""
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())

def save_cifar10_as_csv(csv_path, split="train", max_samples=1000):
    """
    Salva CIFAR-10 come CSV (versione ridotta per testing)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    print(f"Scaricando CIFAR-10 {split} split...")
    dataset = load_dataset("cifar10", split=split)
    
    # Directory per salvare le immagini
    images_dir = os.path.join(os.path.dirname(csv_path), "images")
    os.makedirs(images_dir, exist_ok=True)
    
    image_paths = []
    labels = []
    
    print(f"Salvando {min(max_samples, len(dataset))} immagini...")
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
            
        # Salva immagine
        image_path = os.path.join(images_dir, f"cifar10_{i}.png")
        sample["img"].save(image_path)
        image_paths.append(image_path)
        labels.append(sample["label"])
        
        if (i + 1) % 100 == 0:
            print(f"  Salvate {i + 1}/{min(max_samples, len(dataset))} immagini")
    
    # Crea CSV
    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ Salvato CIFAR-10 in {csv_path} con {len(df)} campioni")

def load_and_adapt_model(model_name, num_classes, save_path):
    """
    Carica un modello e lo adatta al numero di classi
    """
    print(f"\n[MODEL] Caricando {model_name}...")
    
    try:
        # Carica con ignore_mismatched_sizes per gestire differenze di classi
        model = AutoModelForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        print(f"[MODEL] ✅ {model_name} caricato e adattato a {num_classes} classi")
        
        # Info sul modello
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] Parametri totali: {total_params:,}")
        print(f"[MODEL] Parametri trainable: {trainable_params:,}")
        
        # Salva il modello
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(save_path):
            print(f"[MODEL] Salvando in {save_path}...")
            save_model(model, save_path)
        else:
            print(f"[MODEL] {save_path} già esistente, skip salvataggio")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Errore nel caricamento di {model_name}: {e}")
        raise

def test_model_compatibility(teacher_model, student_model, test_input):
    """
    Testa la compatibilità tra teacher e student model
    """
    print("\n[TEST] Testando compatibilità modelli...")
    
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        # Test teacher
        try:
            teacher_output = teacher_model(test_input)
            teacher_logits = teacher_output.logits if hasattr(teacher_output, 'logits') else teacher_output
            print(f"[TEST] Teacher output shape: {teacher_logits.shape}")
        except Exception as e:
            print(f"[ERROR] Teacher forward failed: {e}")
            return False
        
        # Test student
        try:
            student_output = student_model(test_input)
            student_logits = student_output.logits if hasattr(student_output, 'logits') else student_output
            print(f"[TEST] Student output shape: {student_logits.shape}")
        except Exception as e:
            print(f"[ERROR] Student forward failed: {e}")
            return False
        
        # Verifica compatibilità
        if teacher_logits.shape == student_logits.shape:
            print(f"[TEST] ✅ Modelli compatibili! Output shape: {teacher_logits.shape}")
            return True
        else:
            print(f"[ERROR] ❌ Modelli incompatibili! Teacher: {teacher_logits.shape} vs Student: {student_logits.shape}")
            return False

if __name__ == "__main__":
    print("🚀 INIZIO TEST CROSS-ARCHITECTURE IMAGE DISTILLATION")
    print("=" * 60)
    
    # =================== CONFIGURAZIONE ===================
    # Dataset
    dataset_relative_path = './datasets/CIFAR10_small/train.csv'
    
    # Modelli leggeri per testing
    # Teacher: MobileNet (più grande)
    teacher_model_name = "google/mobilenet_v2_1.0_224"
    teacher_save_path = './models/pretrained/mobilenet_teacher.pt'
    
    # Student: ViT tiny (più piccolo)
    student_model_name = "google/vit-base-patch16-224-in21k"  # Lo adatteremo
    student_save_path = './models/pretrained/vit_student.pt'
    
    # Output
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("mobilenet_to_vit", "cifar10")
    
    # =================== DATASET PREPARATION ===================
    print("\n📊 PREPARAZIONE DATASET")
    print("-" * 30)
    
    # Salva dataset ridotto per testing veloce
    save_cifar10_as_csv(dataset_relative_path, split="train", max_samples=500)
    
    # Conta classi
    num_classes = count_classes(dataset_relative_path)
    print(f"📊 Numero classi CIFAR-10: {num_classes}")
    
    # =================== MODEL LOADING ===================
    print("\n🤖 CARICAMENTO MODELLI")
    print("-" * 30)
    
    # Carica teacher model (MobileNet)
    teacher_model = load_and_adapt_model(
        teacher_model_name, 
        num_classes, 
        teacher_save_path
    )
    
    # Carica student model (ViT)
    student_model = load_and_adapt_model(
        student_model_name, 
        num_classes, 
        student_save_path
    )
    
    # =================== COMPATIBILITY TEST ===================
    print("\n🔍 TEST COMPATIBILITÀ")
    print("-" * 30)
    
    # Crea input di test
    test_input = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224
    
    # Testa compatibilità
    if not test_model_compatibility(teacher_model, student_model, test_input):
        print("❌ I modelli non sono compatibili. Interrompo.")
        exit(1)
    
    # =================== DISTILLATION SETUP ===================
    print("\n⚙️ CONFIGURAZIONE DISTILLAZIONE")
    print("-" * 30)
    
    # Config per distillazione
    distillation_config = {
        'temperature': 4.0,      # Temperature per soft targets
        'alpha': 0.7,            # Peso hard loss vs soft loss
        'epochs': 2,             # Poche epoche per test veloce
        'learning_rate': 1e-4,   # Learning rate
        'eval_every': 1          # Valuta ogni epoca
    }
    
    print(f"Config distillazione: {distillation_config}")
    
    # =================== DISTILLATION EXECUTION ===================
    print("\n🔥 ESECUZIONE DISTILLAZIONE")
    print("-" * 30)
    
    try:
        # Crea DistillerBridge
        # NOTA: tokenizer_name=None per modelli vision
        bridge = DistillerBridge(
            teacher_path=teacher_save_path,
            student_path=student_save_path,
            dataset_path=dataset_relative_path,
            output_path=output_model_path,
            tokenizer_name=None,  # Importante per vision models!
            config=distillation_config
        )
        
        print(f"🎯 Bridge configurato:")
        print(f"  - Teacher: {teacher_model_name}")
        print(f"  - Student: {student_model_name}")  
        print(f"  - Classi: {bridge.get_num_classes()}")
        print(f"  - Task: {bridge.dataset_info['task_type']}")
        
        # Esegui distillazione
        print("\n🚀 Avvio distillazione...")
        bridge.distill()
        
        print("\n✅ DISTILLAZIONE COMPLETATA CON SUCCESSO!")
        print(f"📁 Modello salvato in: {output_model_path}")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante la distillazione: {e}")
        import traceback
        traceback.print_exc()
        
    # =================== FINAL SUMMARY ===================
    print("\n📋 RIEPILOGO")
    print("=" * 60)
    print(f"Teacher Model: {teacher_model_name}")
    print(f"Student Model: {student_model_name}")
    print(f"Dataset: CIFAR-10 ({num_classes} classi)")
    print(f"Campioni: ~500 (per test veloce)")
    print(f"Cross-Architecture: MobileNet → ViT")
    print(f"Output: {output_model_path}")
    
    # Test finale opzionale
    if os.path.exists(os.path.join(output_model_path, "student.pt")):
        print("✅ Modello student salvato correttamente")
        
        # Caricamento e test veloce del modello distillato
        try:
            print("\n🧪 Test caricamento modello distillato...")
            distilled_model = torch.load(os.path.join(output_model_path, "student.pt"))
            distilled_model.eval()
            
            with torch.no_grad():
                test_output = distilled_model(test_input)
                test_logits = test_output.logits if hasattr(test_output, 'logits') else test_output
                print(f"✅ Modello distillato funziona! Output shape: {test_logits.shape}")
                
                # Predizione di esempio
                probabilities = torch.softmax(test_logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
                
                print(f"Predizione esempio: Classe {predicted_class.item()}, Confidenza: {confidence.item():.3f}")
                
        except Exception as e:
            print(f"⚠️ Errore nel test del modello distillato: {e}")
    
    print("\n🎉 TEST COMPLETATO!")

# =================== VERSIONE ALTERNATIVA CON MODELLI ANCORA PIÙ LEGGERI ===================

def main_super_light():
    """
    Versione con modelli ancora più leggeri per test velocissimi
    """
    print("🚀 VERSIONE SUPER LEGGERA")
    
    # Modelli molto piccoli
    teacher_name = "google/mobilenet_v2_0.35_224"  # MobileNet 0.35 (molto piccolo)
    student_name = "microsoft/dit-base-distilled-patch16-224"  # DeiT distilled (piccolo)
    
    # Dataset ancora più piccolo
    save_cifar10_as_csv('./datasets/CIFAR10_tiny/train.csv', max_samples=100)
    
    # Config veloce
    fast_config = {
        'temperature': 3.0,
        'alpha': 0.8,
        'epochs': 1,  # Solo 1 epoca!
        'learning_rate': 1e-3,
        'eval_every': 1
    }
    
    # Resto del codice uguale...
    print("Configurazione super veloce per debugging!")

# =================== DEBUGGING HELPER ===================

def debug_batch_format(dataloader):
    """
    Helper per debuggare il formato dei batch
    """
    print("\n🐛 DEBUG: Formato batch")
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Type: {type(batch)}")
        
        if isinstance(batch, dict):
            print("  Dict keys:", list(batch.keys()))
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: {v.shape}")
                else:
                    print(f"    {k}: {type(v)}")
        elif isinstance(batch, (list, tuple)):
            print(f"  Tuple/List length: {len(batch)}")
            for j, item in enumerate(batch):
                if hasattr(item, 'shape'):
                    print(f"    [{j}]: {item.shape}")
                else:
                    print(f"    [{j}]: {type(item)}")
        
        if i >= 2:  # Solo primi 3 batch
            break
    
    print("🐛 DEBUG completato")

#Uncomment per versione super veloce:
if __name__ == "__main__":
     main_super_light()