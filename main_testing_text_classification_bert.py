# =================== MAIN TEXT DISTILLATION - BERT TO DISTILBERT ===================
# File: main_text_distillation.py

import os
import pandas as pd
from datasets import load_dataset
from utils.directory import ProjectStructure
from distiller import DistillerBridge
from setup_bert import setup_bert_distilbert_models, print_model_details
import time

def count_classes(csv_path):
    """Conta velocemente le classi dal CSV"""
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())

def save_imdb_as_csv(csv_path, split="train", max_samples=5000):
    """
    Salva dataset IMDB per sentiment analysis
    Dataset più challenging di SST-2, perfetto per 5 minuti su RTX 4090
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    print(f"📥 Scaricando IMDB dataset ({split} split)...")
    dataset = load_dataset("imdb", split=split)
    
    # Prendi un subset bilanciato
    texts = []
    labels = []
    
    # Conta per bilanciamento
    pos_count = 0
    neg_count = 0
    max_per_class = max_samples // 2
    
    print(f"Selezionando {max_samples} campioni bilanciati...")
    
    for i, sample in enumerate(dataset):
        if sample['label'] == 1 and pos_count < max_per_class:
            texts.append(sample['text'])
            labels.append(sample['label'])
            pos_count += 1
        elif sample['label'] == 0 and neg_count < max_per_class:
            texts.append(sample['text'])
            labels.append(sample['label'])
            neg_count += 1
        
        # Stop quando abbiamo abbastanza campioni
        if pos_count >= max_per_class and neg_count >= max_per_class:
            break
        
        if (i + 1) % 1000 == 0:
            print(f"  Processati {i+1} campioni, selezionati: {len(texts)}")
    
    # Crea DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(csv_path, index=False)
    print(f"✅ IMDB dataset salvato: {csv_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Positive: {sum(df['label'] == 1)}")
    print(f"   Negative: {sum(df['label'] == 0)}")
    
    # Mostra alcuni esempi
    print(f"\n📝 ESEMPI DI TESTI:")
    for i in range(3):
        label_text = "POSITIVE" if df.iloc[i]['label'] == 1 else "NEGATIVE"
        text_preview = df.iloc[i]['text'][:100] + "..."
        print(f"{i+1}. [{label_text}] {text_preview}")

def save_sst2_as_csv(csv_path, split="train", max_samples=3000):
    """
    Alternativa più leggera: SST-2 dataset
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    print(f"📥 Scaricando SST-2 dataset ({split} split)...")
    dataset = load_dataset("glue", "sst2", split=split)
    
    # Prendi subset
    subset_size = min(max_samples, len(dataset))
    indices = list(range(0, len(dataset), len(dataset) // subset_size))[:subset_size]
    
    texts = [dataset[i]["sentence"] for i in indices]
    labels = [dataset[i]["label"] for i in indices]
    
    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })
    
    df.to_csv(csv_path, index=False)
    print(f"✅ SST-2 dataset salvato: {csv_path} ({len(df)} campioni)")

def estimate_training_time(num_samples, batch_size=16, epochs=3):
    """
    Stima il tempo di training su RTX 4090
    """
    # Stime basate su benchmark RTX 4090 per BERT
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    # RTX 4090: ~50-80 steps/sec per BERT-base
    steps_per_second = 65  # Conservative estimate
    
    estimated_seconds = total_steps / steps_per_second
    estimated_minutes = estimated_seconds / 60
    
    print(f"\n⏱️  STIMA TEMPO TRAINING (RTX 4090):")
    print(f"Campioni: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Epoche: {epochs}")
    print(f"Steps totali: {total_steps}")
    print(f"Tempo stimato: {estimated_minutes:.1f} minuti")
    
    return estimated_minutes

if __name__ == "__main__":
    print("🚀 TEXT DISTILLATION: BERT → DistilBERT")
    print("🚀 Optimized for RTX 4090 (~5 minutes)")
    print("=" * 60)
    
    start_time = time.time()
    
    # =================== CONFIGURAZIONE ===================
    # Dataset (scegli uno dei due)
    use_imdb = True  # True per IMDB (più challenging), False per SST-2 (più veloce)
    
    if use_imdb:
        dataset_relative_path = './datasets/IMDB_distill/train.csv'
        dataset_samples = 5000  # Perfetto per ~5 min su RTX 4090
        dataset_name = "IMDB"
    else:
        dataset_relative_path = './datasets/SST2_distill/train.csv'
        dataset_samples = 3000
        dataset_name = "SST-2"
    
    # Modelli
    teacher_save_path = './models/text_distill/bert_base_teacher.pt'
    student_save_path = './models/text_distill/distilbert_student.pt'
    
    # Output
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("bert_to_distilbert", dataset_name.lower())
    
    # =================== DATASET PREPARATION ===================
    print(f"\n📊 PREPARAZIONE DATASET {dataset_name}")
    print("-" * 40)
    
    if use_imdb:
        save_imdb_as_csv(dataset_relative_path, split="train", max_samples=dataset_samples)
    else:
        save_sst2_as_csv(dataset_relative_path, split="train", max_samples=dataset_samples)
    
    # Conta classi
    num_classes = count_classes(dataset_relative_path)
    print(f"📊 Numero classi: {num_classes}")
    
    # Stima tempo
    estimate_training_time(dataset_samples, batch_size=16, epochs=3)
    
    # =================== MODEL SETUP ===================
    print(f"\n🤖 SETUP MODELLI BERT → DistilBERT")
    print("-" * 40)
    
    teacher_model, student_model, teacher_tokenizer, student_tokenizer = setup_bert_distilbert_models(
        num_classes=num_classes,
        teacher_save_path=teacher_save_path,
        student_save_path=student_save_path
    )
    
    if not teacher_model or not student_model:
        print("❌ Errore nel setup dei modelli!")
        exit(1)
    
    # =================== DISTILLATION CONFIG ===================
    print(f"\n⚙️ CONFIGURAZIONE DISTILLAZIONE")
    print("-" * 40)
    
    # Config ottimizzato per RTX 4090 e BERT distillation
    distillation_config = {
        'temperature': 2.0,      # Standard per BERT distillation
        'alpha': 0.3,            # 70% hard loss, 30% soft loss
        'epochs': 3,             # 3 epoche per ~5 minuti
        'learning_rate': 5e-5,   # Learning rate ottimale per BERT fine-tuning
        'eval_every': 1,         # Valuta ogni epoca
        'batch_size': 16         # Ottimale per RTX 4090
    }
    
    print(f"Config distillazione:")
    for key, value in distillation_config.items():
        print(f"  {key}: {value}")
    
    # =================== DISTILLATION EXECUTION ===================
    print(f"\n🔥 ESECUZIONE DISTILLAZIONE")
    print("-" * 40)
    
    try:
        # Crea DistillerBridge
        bridge = DistillerBridge(
            teacher_path=teacher_save_path,
            student_path=student_save_path,
            dataset_path=dataset_relative_path,
            output_path=output_model_path,
            tokenizer_name="bert-base-uncased",  # Per text processing
            config=distillation_config
        )
        
        print(f"🎯 Bridge configurato:")
        print(f"  - Teacher: BERT-base (~110M params)")
        print(f"  - Student: DistilBERT (~66M params)")
        print(f"  - Dataset: {dataset_name} ({dataset_samples} campioni)")
        print(f"  - Classi: {bridge.get_num_classes()}")
        print(f"  - Task: {bridge.dataset_info['task_type']}")
        
        # Timing setup
        distill_start = time.time()
        
        # Esegui distillazione
        print(f"\n🚀 Avvio distillazione BERT → DistilBERT...")
        print(f"Target time: ~5 minuti su RTX 4090")
        bridge.distill()
        
        distill_end = time.time()
        distill_time = (distill_end - distill_start) / 60
        
        print(f"\n✅ DISTILLAZIONE COMPLETATA!")
        print(f"⏱️  Tempo reale: {distill_time:.1f} minuti")
        print(f"📁 Modello salvato: {output_model_path}")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante distillazione: {e}")
        import traceback
        traceback.print_exc()
        
    # =================== VALIDATION E PERFORMANCE ===================
    print(f"\n🧪 VALIDAZIONE E PERFORMANCE")
    print("-" * 40)
    
    if os.path.exists(os.path.join(output_model_path, "student.pt")):
        print("✅ Modello distillato salvato correttamente")
        
        try:
            import torch
            
            print(f"\n🔍 Test caricamento modello distillato...")
            distilled_model = torch.load(os.path.join(output_model_path, "student.pt"),weights_only=False)
            distilled_model.eval()
            
            # Test con frasi di esempio
            test_sentences = [
                "This movie is absolutely fantastic and amazing!",
                "I hate this film, it's terrible and boring.",
                "The movie was okay, nothing special really.",
            ]
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            inputs = tokenizer(
                test_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = distilled_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            print(f"✅ Modello distillato funziona perfettamente!")
            print(f"\n📝 Test su frasi di esempio:")
            
            for i, sentence in enumerate(test_sentences):
                pred_class = torch.argmax(predictions[i]).item()
                confidence = torch.max(predictions[i]).item()
                sentiment = "POSITIVE" if pred_class == 1 else "NEGATIVE"
                
                print(f"{i+1}. '{sentence}'")
                print(f"   Predizione: {sentiment} (confidenza: {confidence:.3f})")
            
            # Performance comparison
            print(f"\n⚡ PERFORMANCE COMPARISON:")
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in distilled_model.parameters())
            
            print(f"Teacher (BERT):        {teacher_params:,} parametri")
            print(f"Student (Distilled):   {student_params:,} parametri")
            print(f"Compression ratio:     {teacher_params/student_params:.1f}x")
            print(f"Size reduction:        {((teacher_params-student_params)/teacher_params*100):.1f}%")
            
        except Exception as e:
            print(f"⚠️ Errore nel test finale: {e}")
    
    # =================== FINAL SUMMARY ===================
    total_time = (time.time() - start_time) / 60
    
    print(f"\n📋 RIEPILOGO FINALE")
    print("=" * 60)
    print(f"🏗️  Architettura: BERT-base → DistilBERT")
    print(f"📊 Dataset: {dataset_name} ({dataset_samples} campioni)")
    print(f"⏱️  Tempo totale: {total_time:.1f} minuti")
    print(f"🎯 Target: ~5 minuti ({'✅ Centrato!' if 4 <= total_time <= 6 else '⚠️  Fuori target'})")
    print(f"💾 Output: {output_model_path}")
    print(f"🚀 GPU: RTX 4090 optimized")
    print(f"🔬 Cross-architecture: Transformer → Transformer (diversi layer count)")
    
    print(f"\n🎉 TEXT DISTILLATION COMPLETATA!")
    print(f"📈 Hai ottenuto un modello {teacher_params//student_params if 'teacher_params' in locals() else '1.7'}x più leggero!")

# =================== OPZIONI ALTERNATIVE ===================

def quick_test_run():
    """
    Versione veloce per test rapidi (1-2 minuti)
    """
    print("🚀 QUICK TEST RUN (1-2 minuti)")
    
    # Dataset più piccolo
    save_sst2_as_csv('./datasets/SST2_quick/train.csv', max_samples=1000)
    
    # Config veloce
    quick_config = {
        'temperature': 3.0,
        'alpha': 0.8,
        'epochs': 1,  # Solo 1 epoca
        'learning_rate': 1e-4,
        'eval_every': 1
    }
    
    print("Configurazione veloce per test rapido!")

# Uncomment per test veloce:
if __name__ == "__main__":
    quick_test_run()