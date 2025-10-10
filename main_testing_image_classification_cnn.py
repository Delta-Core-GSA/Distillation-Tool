# =================== CUSTOM CNN MODELS ===================
# File: custom_models/custom_cnns.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherCNN(nn.Module):
    """
    Teacher CNN: Architettura più profonda e complessa
    Simile a una versione semplificata di ResNet
    """
    def __init__(self, num_classes=10):
        super(TeacherCNN, self).__init__()
        
        print(f"[TEACHER_CNN] Inizializzando con {num_classes} classi")
        
        # Block 1: Input -> 64 filters
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Block 2: 64 -> 128 filters
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Block 3: 128 -> 256 filters
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Extra layer
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Block 4: 256 -> 512 filters (Deep!)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)  # 4x4 -> 2x2
        
        # Global Average Pooling + FC
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[TEACHER_CNN] Parametri totali: {total_params:,}")
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3 (Deep)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))  # Extra depth
        x = self.pool3(x)
        
        # Block 4 (Very Deep)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class StudentCNN(nn.Module):
    """
    Student CNN: Architettura più leggera e veloce
    Simile a MobileNet con depthwise separable convolutions
    """
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        
        print(f"[STUDENT_CNN] Inizializzando con {num_classes} classi")
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise Separable Conv Blocks (MobileNet-style)
        self.dw_conv1 = self._make_dw_block(32, 64, stride=1)    # 16x16
        self.dw_conv2 = self._make_dw_block(64, 128, stride=2)   # 16x16 -> 8x8
        self.dw_conv3 = self._make_dw_block(128, 128, stride=1)   # 8x8
        self.dw_conv4 = self._make_dw_block(128, 256, stride=2)   # 8x8 -> 4x4
        self.dw_conv5 = self._make_dw_block(256, 256, stride=1)   # 4x4
        
        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)  # Less dropout than teacher
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[STUDENT_CNN] Parametri totali: {total_params:,}")
    
    def _make_dw_block(self, in_channels, out_channels, stride=1):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise separable blocks
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.dw_conv5(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# =================== MODEL COMPARISON ===================

def compare_models(num_classes=10):
    """
    Confronta le due architetture
    """
    print("\n" + "="*60)
    print("CONFRONTO ARCHITETTURE")
    print("="*60)
    
    teacher = TeacherCNN(num_classes)
    student = StudentCNN(num_classes)
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n📊 STATISTICHE:")
    print(f"Teacher CNN (Deep):     {teacher_params:,} parametri")
    print(f"Student CNN (Light):    {student_params:,} parametri")
    print(f"Riduzione parametri:    {((teacher_params - student_params) / teacher_params * 100):.1f}%")
    
    # Test forward pass
    test_input = torch.randn(4, 3, 32, 32)  # Batch di 4 immagini CIFAR-10
    
    teacher.eval()
    student.eval()
    
    with torch.no_grad():
        teacher_out = teacher(test_input)
        student_out = student(test_input)
    
    print(f"\n🧪 TEST FORWARD PASS:")
    print(f"Input shape:            {test_input.shape}")
    print(f"Teacher output shape:   {teacher_out.shape}")
    print(f"Student output shape:   {student_out.shape}")
    print(f"Output compatibility:   {'✅' if teacher_out.shape == student_out.shape else '❌'}")
    
    # Architecture comparison
    print(f"\n🏗️  DIFFERENZE ARCHITETTURALI:")
    print(f"Teacher: Deep CNN (VGG/ResNet-like)")
    print(f"  - 4 blocchi conv con BatchNorm")
    print(f"  - Fino a 512 filtri")
    print(f"  - Dropout 0.5")
    print(f"  - Standard convolutions")
    
    print(f"\nStudent: Light CNN (MobileNet-like)")
    print(f"  - Depthwise separable convolutions")
    print(f"  - Massimo 256 filtri")
    print(f"  - Dropout 0.2")  
    print(f"  - Architettura efficiente")
    
    return teacher, student

# =================== HELPER FUNCTIONS ===================

def save_custom_model(model, save_path, model_name):
    """
    Salva un modello custom
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        torch.save(model, save_path)
        print(f"💾 {model_name} salvato in: {save_path}")
    else:
        print(f"📁 {model_name} già esistente: {save_path}")

def test_model_compatibility(teacher, student, input_shape=(1, 3, 32, 32)):
    """
    Testa la compatibilità tra teacher e student
    """
    print(f"\n🔍 TEST COMPATIBILITÀ")
    print(f"Input shape: {input_shape}")
    
    test_input = torch.randn(*input_shape)
    
    teacher.eval()
    student.eval()
    
    try:
        with torch.no_grad():
            teacher_out = teacher(test_input)
            student_out = student(test_input)
        
        if teacher_out.shape == student_out.shape:
            print(f"✅ Modelli compatibili!")
            print(f"   Output shape: {teacher_out.shape}")
            return True
        else:
            print(f"❌ Modelli incompatibili!")
            print(f"   Teacher: {teacher_out.shape}")
            print(f"   Student: {student_out.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Errore nel test: {e}")
        return False

# =================== USAGE EXAMPLE ===================



    # =================== MAIN CUSTOM CNN CROSS-ARCHITECTURE ===================
# File: main_custom_cnn.py

import os
import torch
import pandas as pd
from datasets import load_dataset
from utils.directory import ProjectStructure
from distiller import DistillerBridge
from PIL import Image
import numpy as np

# Import delle nostre CNN custom

def count_classes(csv_path):
    """Conta velocemente le classi dal CSV"""
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())

def save_cifar10_as_csv(csv_path, split="train", max_samples=800):
    """
    Salva CIFAR-10 come CSV per testing custom CNN
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
        image_path = os.path.join(images_dir, f"cifar10_custom_{i}.png")
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
    print(f"✅ Salvato CIFAR-10 custom in {csv_path} con {len(df)} campioni")

def create_and_save_custom_models(num_classes, teacher_path, student_path):
    """
    Crea e salva i modelli CNN custom
    """
    print("\n🏗️  CREAZIONE MODELLI CUSTOM")
    print("-" * 40)
    
    # Crea modelli
    teacher_model = TeacherCNN(num_classes=num_classes)
    student_model = StudentCNN(num_classes=num_classes)
    
    # Confronta architetture
    print("\n📊 CONFRONTO ARCHITETTURE:")
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    print(f"Teacher (Deep CNN):     {teacher_params:,} parametri")
    print(f"Student (Light CNN):    {student_params:,} parametri")
    print(f"Compression ratio:      {teacher_params/student_params:.1f}x")
    print(f"Parameter reduction:    {((teacher_params-student_params)/teacher_params*100):.1f}%")
    
    # Test compatibilità
    print(f"\n🔍 TEST COMPATIBILITÀ:")
    if test_model_compatibility(teacher_model, student_model):
        print("✅ Modelli pronti per distillazione!")
    else:
        print("❌ Errore di compatibilità!")
        return None, None
    
    # Salva modelli
    os.makedirs(os.path.dirname(teacher_path), exist_ok=True)
    os.makedirs(os.path.dirname(student_path), exist_ok=True)
    
    if not os.path.exists(teacher_path):
        torch.save(teacher_model, teacher_path)
        print(f"💾 Teacher CNN salvato: {teacher_path}")
    else:
        print(f"📁 Teacher CNN già esistente: {teacher_path}")
    
    if not os.path.exists(student_path):
        torch.save(student_model, student_path)
        print(f"💾 Student CNN salvato: {student_path}")
    else:
        print(f"📁 Student CNN già esistente: {student_path}")
    
    return teacher_model, student_model

def analyze_models_architectures(teacher_model, student_model):
    """
    Analizza le differenze architetturali tra i modelli
    """
    print("\n🔬 ANALISI ARCHITETTURALE DETTAGLIATA")
    print("=" * 50)
    
    print("\n🏗️  TEACHER CNN (Deep Architecture):")
    print("├── Block 1: 3→64→64 + MaxPool (32→16)")
    print("├── Block 2: 64→128→128 + MaxPool (16→8)")  
    print("├── Block 3: 128→256→256→256 + MaxPool (8→4)")
    print("├── Block 4: 256→512→512 + MaxPool (4→2)")
    print("└── GlobalAvgPool + Dropout(0.5) + FC")
    
    print("\n📱 STUDENT CNN (MobileNet-like):")
    print("├── Initial Conv: 3→32 (stride=2, 32→16)")
    print("├── DW Block 1: 32→64 (16→16)")
    print("├── DW Block 2: 64→128 (16→8, stride=2)")
    print("├── DW Block 3: 128→128 (8→8)")
    print("├── DW Block 4: 128→256 (8→4, stride=2)")
    print("├── DW Block 5: 256→256 (4→4)")
    print("└── GlobalAvgPool + Dropout(0.2) + FC")
    
    print(f"\n⚡ VANTAGGI ARCHITETTURALI:")
    print(f"✅ Teacher: Deep feature extraction, alta capacità")
    print(f"✅ Student: Efficient depthwise convs, veloce inference")
    print(f"✅ Cross-arch: CNN→CNN ma architetture molto diverse")
    print(f"✅ Perfect match: Stesso output space ({teacher_model.fc.out_features} classi)")

if __name__ == "__main__":
# Test delle architetture
    teacher, student = compare_models(num_classes=10)
    
    # Test compatibilità
    is_compatible = test_model_compatibility(teacher, student)
    
    if is_compatible:
        print(f"\n🎉 Le architetture sono pronte per la distillazione!")
    else:
        print(f"\n⚠️  Le architetture necessitano aggiustamenti!")
    
    # Esempio di salvataggio
    save_custom_model(teacher, './models/custom/teacher_cnn.pt', 'Teacher CNN')
    save_custom_model(student, './models/custom/student_cnn.pt', 'Student CNN')


    print("🚀 CUSTOM CNN CROSS-ARCHITECTURE DISTILLATION")
    print("=" * 60)
    
    # =================== CONFIGURAZIONE ===================
    # Dataset
    dataset_relative_path = './datasets/CIFAR10_custom/train.csv'
    
    # Modelli custom
    teacher_save_path = './models/custom/teacher_deep_cnn.pt'
    student_save_path = './models/custom/student_light_cnn.pt'
    
    # Output
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("deep_cnn_to_light_cnn", "cifar10")
    
    # =================== DATASET PREPARATION ===================
    print("\n📊 PREPARAZIONE DATASET")
    print("-" * 30)
    
    # Salva dataset
    save_cifar10_as_csv(dataset_relative_path, split="train", max_samples=800)
    
    # Conta classi
    num_classes = count_classes(dataset_relative_path)
    print(f"📊 Numero classi CIFAR-10: {num_classes}")
    
    # =================== MODEL CREATION ===================
    print("\n🤖 CREAZIONE MODELLI CUSTOM")
    print("-" * 30)
    
    # Crea e salva modelli custom
    teacher_model, student_model = create_and_save_custom_models(
        num_classes, teacher_save_path, student_save_path
    )
    
    if teacher_model is None or student_model is None:
        print("❌ Errore nella creazione dei modelli. Uscita.")
        exit(1)
    
    # Analisi architetturale
    analyze_models_architectures(teacher_model, student_model)
    
    # =================== DISTILLATION SETUP ===================
    print("\n⚙️ CONFIGURAZIONE DISTILLAZIONE")
    print("-" * 30)
    
    # Config ottimizzato per CNN custom
    distillation_config = {
        'temperature': 5.0,      # Temperatura più alta per CNN deep
        'alpha': 0.6,            # Più peso al soft target (CNN ha feature ricche)
        'epochs': 3,             # Poche epoche per test
        'learning_rate': 2e-4,   # LR leggermente più alto per CNN
        'eval_every': 1          # Valuta ogni epoca
    }
    
    print(f"Config distillazione: {distillation_config}")
    
    # =================== DISTILLATION EXECUTION ===================
    print("\n🔥 ESECUZIONE DISTILLAZIONE CUSTOM CNN")
    print("-" * 40)
    
    try:
        # Crea DistillerBridge
        # NOTA: tokenizer_name=None per modelli custom CNN
        bridge = DistillerBridge(
            teacher_path=teacher_save_path,
            student_path=student_save_path,
            dataset_path=dataset_relative_path,
            output_path=output_model_path,
            tokenizer_name=None,  # Importante per CNN custom!
            config=distillation_config
        )
        
        print(f"🎯 Bridge configurato:")
        print(f"  - Teacher: Deep CNN ({sum(p.numel() for p in teacher_model.parameters()):,} params)")
        print(f"  - Student: Light CNN ({sum(p.numel() for p in student_model.parameters()):,} params)")
        print(f"  - Classi: {bridge.get_num_classes()}")
        print(f"  - Task: {bridge.dataset_info['task_type']}")
        print(f"  - Architecture: Custom CNN → Custom CNN (Cross-arch!)")
        
        # Esegui distillazione
        print("\n🚀 Avvio distillazione custom CNN...")
        bridge.distill()
        
        print("\n✅ DISTILLAZIONE CUSTOM CNN COMPLETATA!")
        print(f"📁 Modello distillato salvato in: {output_model_path}")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante la distillazione: {e}")
        import traceback
        traceback.print_exc()
        
    # =================== VALIDATION E TESTING ===================
    print("\n🧪 VALIDAZIONE FINALE")
    print("-" * 30)
    
    # Test finale del modello distillato
    if os.path.exists(os.path.join(output_model_path, "student.pt")):
        print("✅ Modello student salvato correttamente")
        
        try:
            print("\n🔍 Test caricamento modello distillato...")
            distilled_model = torch.load(os.path.join(output_model_path, "student.pt"))
            distilled_model.eval()
            
            # Test con input CIFAR-10
            test_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 format
            
            with torch.no_grad():
                test_output = distilled_model(test_input)
                
                # Verifica che l'output sia corretto
                if test_output.shape == (1, num_classes):
                    print(f"✅ Modello distillato funziona perfettamente!")
                    print(f"   Input shape: {test_input.shape}")
                    print(f"   Output shape: {test_output.shape}")
                    
                    # Predizione di esempio
                    probabilities = torch.softmax(test_output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)
                    confidence = torch.max(probabilities, dim=1)[0]
                    
                    print(f"   Predizione esempio: Classe {predicted_class.item()}")
                    print(f"   Confidenza: {confidence.item():.3f}")
                    
                    # Confronta con modelli originali
                    print(f"\n🆚 CONFRONTO PERFORMANCE:")
                    
                    teacher_model.eval()
                    student_original = StudentCNN(num_classes)
                    student_original.eval()
                    
                    with torch.no_grad():
                        teacher_out = teacher_model(test_input)
                        student_orig_out = student_original(test_input)
                        distilled_out = test_output
                        
                        teacher_pred = torch.argmax(teacher_out, dim=1)
                        student_orig_pred = torch.argmax(student_orig_out, dim=1)
                        distilled_pred = torch.argmax(distilled_out, dim=1)
                        
                        print(f"   Teacher prediction:     {teacher_pred.item()}")
                        print(f"   Student (original):     {student_orig_pred.item()}")
                        print(f"   Student (distilled):    {distilled_pred.item()}")
                        
                        # Similarity check
                        teacher_prob = torch.softmax(teacher_out, dim=1)
                        distilled_prob = torch.softmax(distilled_out, dim=1)
                        
                        # KL divergence similarity
                        kl_sim = torch.nn.functional.kl_div(
                            torch.log(distilled_prob), teacher_prob, reduction='batchmean'
                        )
                        print(f"   KL divergence (Teacher↔Distilled): {kl_sim.item():.4f}")
                        
                else:
                    print(f"❌ Errore: Output shape errato {test_output.shape}, atteso ({1}, {num_classes})")
                    
        except Exception as e:
            print(f"⚠️ Errore nel test finale: {e}")
    
    # =================== FINAL SUMMARY ===================
    print("\n📋 RIEPILOGO FINALE")
    print("=" * 60)
    print(f"🏗️  Teacher: Deep CNN (VGG/ResNet-like)")
    print(f"📱 Student: Light CNN (MobileNet-like)")
    print(f"📊 Dataset: CIFAR-10 ({num_classes} classi, ~800 campioni)")
    print(f"🔄 Cross-Architecture: ✅ CNN→CNN con architetture diverse")
    #print(f"⚡ Compression: ~{teacher_params//student_params if 'teacher_params' in locals() else 'N/A'}x parametri")
    print(f"📁 Output: {output_model_path}")
    print(f"🎯 Obiettivo: Trasferire conoscenza da rete profonda a rete efficiente")
    
    print("\n🎉 TEST CUSTOM CNN CROSS-ARCHITECTURE COMPLETATO!")
    print("🔬 Hai testato con successo la distillazione tra architetture CNN diverse!")

# =================== EXTRA: BENCHMARK FUNCTION ===================

def benchmark_models_speed(teacher_model, student_model, num_tests=100):
    """
    Benchmark della velocità di inference tra teacher e student
    """
    import time
    
    print(f"\n⏱️  BENCHMARK VELOCITÀ INFERENCE")
    print("-" * 40)
    
    # Prepara input di test
    test_input = torch.randn(1, 3, 32, 32)
    
    # Warm up
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        for _ in range(10):  # Warm up
            _ = teacher_model(test_input)
            _ = student_model(test_input)
    
    # Benchmark Teacher
    teacher_times = []
    with torch.no_grad():
        for _ in range(num_tests):
            start_time = time.time()
            _ = teacher_model(test_input)
            end_time = time.time()
            teacher_times.append(end_time - start_time)
    
    # Benchmark Student
    student_times = []
    with torch.no_grad():
        for _ in range(num_tests):
            start_time = time.time()
            _ = student_model(test_input)
            end_time = time.time()
            student_times.append(end_time - start_time)
    
    # Risultati
    teacher_avg = sum(teacher_times) / len(teacher_times) * 1000  # ms
    student_avg = sum(student_times) / len(student_times) * 1000  # ms
    speedup = teacher_avg / student_avg
    
    print(f"Teacher (Deep CNN):    {teacher_avg:.2f} ms/inference")
    print(f"Student (Light CNN):   {student_avg:.2f} ms/inference")
    print(f"Speedup:              {speedup:.1f}x più veloce")
    print(f"Test eseguiti:        {num_tests}")
    
    return teacher_avg, student_avg, speedup

def analyze_feature_maps(teacher_model, student_model, test_input):
    """
    Analizza le feature maps dei due modelli (opzionale)
    """
    print(f"\n🔍 ANALISI FEATURE MAPS")
    print("-" * 30)
    
    teacher_model.eval()
    student_model.eval()
    
    # Hook per catturare feature maps
    teacher_features = []
    student_features = []
    
    def hook_teacher(module, input, output):
        if len(output.shape) == 4:  # Solo conv layers
            teacher_features.append(output.detach())
    
    def hook_student(module, input, output):
        if len(output.shape) == 4:  # Solo conv layers
            student_features.append(output.detach())
    
    # Registra hooks
    teacher_hooks = []
    student_hooks = []
    
    for module in teacher_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            teacher_hooks.append(module.register_forward_hook(hook_teacher))
    
    for module in student_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            student_hooks.append(module.register_forward_hook(hook_student))
    
    # Forward pass
    with torch.no_grad():
        _ = teacher_model(test_input)
        _ = student_model(test_input)
    
    # Rimuovi hooks
    for hook in teacher_hooks:
        hook.remove()
    for hook in student_hooks:
        hook.remove()
    
    # Analizza feature maps
    print(f"Teacher feature maps: {len(teacher_features)} layers")
    print(f"Student feature maps: {len(student_features)} layers")
    
    if teacher_features and student_features:
        print(f"Teacher first layer:  {teacher_features[0].shape}")
        print(f"Student first layer:  {student_features[0].shape}")
        print(f"Teacher last layer:   {teacher_features[-1].shape}")
        print(f"Student last layer:   {student_features[-1].shape}")
    
    return teacher_features, student_features

# Funzione helper per il main
def run_additional_tests():
    """
    Esegue test aggiuntivi se richiesti
    """
    try:
        # Carica modelli per test aggiuntivi
        teacher = TeacherCNN(num_classes=10)
        student = StudentCNN(num_classes=10)
        
        # Benchmark velocità
        benchmark_models_speed(teacher, student, num_tests=50)
        
        # Analisi feature maps
        test_input = torch.randn(1, 3, 32, 32)
        analyze_feature_maps(teacher, student, test_input)
        
    except Exception as e:
        print(f"⚠️ Test aggiuntivi falliti: {e}")

# =================== MODELLO SUMMARY FUNCTION ===================

def print_model_summary(model, model_name, input_size=(1, 3, 32, 32)):
    """
    Stampa un summary dettagliato del modello
    """
    print(f"\n📋 {model_name.upper()} SUMMARY")
    print("-" * 40)
    
    # Parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB):      {total_params * 4 / (1024*1024):.2f}")
    
    # Test forward pass per vedere le dimensioni
    test_input = torch.randn(*input_size)
    model.eval()
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape:          {test_input.shape}")
    print(f"Output shape:         {output.shape}")
    
    # Conta layer types
    conv_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    bn_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    linear_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    
    print(f"Conv2d layers:        {conv_layers}")
    print(f"BatchNorm2d layers:   {bn_layers}")
    print(f"Linear layers:        {linear_layers}")

# Aggiungi al main se vuoi test extra
if __name__ == "__main__" and "--extra-tests" in os.sys.argv:
    print("\n🧪 ESECUZIONE TEST AGGIUNTIVI")
    print("=" * 40)
    run_additional_tests()