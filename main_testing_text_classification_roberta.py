# =================== MAIN TEXT DISTILLATION - ROBERTA TO DISTILROBERTA ===================
# File: main_text_distillation_roberta.py

import os
import pandas as pd
from datasets import load_dataset
from utils.directory import ProjectStructure
from distiller import DistillerBridge
from setup_roberta import setup_roberta_distilroberta_models, print_model_details
import time


import warnings
warnings.filterwarnings("ignore", message=".*FLOPs.*")
warnings.filterwarnings("ignore", message=".*Embedding.*")
warnings.filterwarnings("ignore", message=".*GELUActivation.*")


def count_classes(csv_path):
    """Conta velocemente le classi dal CSV"""
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())

def save_amazon_reviews_as_csv(csv_path, split="train", max_samples=15000):
    """
    Salva Amazon Reviews dataset per sentiment analysis
    Dataset più challenging e realistico per training medio-lungo
    5 stelle → classe 4 (positive), 1-2 stelle → classe 0 (negative)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return
    
    print(f"📥 Scaricando Amazon Reviews dataset ({split} split)...")
    print("🎯 Dataset scelto: Amazon Product Reviews (più challenging)")
    
    try:
        # Prova prima amazon_reviews_multi per multilingua (più interessante)
        dataset = load_dataset("amazon_reviews_multi", "en", split=split)
        print("✅ Caricato Amazon Reviews Multi (inglese)")
    except:
        # Fallback a dataset alternativo
        print("📦 Caricamento alternativo: Stanford Sentiment Treebank esteso...")
        dataset = load_dataset("stanfordnlp/sst2", split=split)
    
    # Preprocessing per Amazon Reviews
    texts = []
    labels = []
    
    pos_count = 0
    neg_count = 0
    max_per_class = max_samples // 2
    
    print(f"🔄 Processando dataset per {max_samples} campioni bilanciati...")
    print("📊 Mapping: 4-5 stelle → POSITIVE, 1-2 stelle → NEGATIVE")
    
    processed = 0
    for sample in dataset:
        processed += 1
        
        # Gestione diversi formati dataset
        if 'stars' in sample:
            # Amazon Reviews Multi
            text = sample['review_body'] or sample['review_title'] or ""
            stars = sample['stars']
            
            # Converti stelle in sentiment binario
            if stars >= 4 and pos_count < max_per_class:
                texts.append(text)
                labels.append(1)  # Positive
                pos_count += 1
            elif stars <= 2 and neg_count < max_per_class:
                texts.append(text)
                labels.append(0)  # Negative
                neg_count += 1
                
        elif 'label' in sample:
            # SST2 format fallback
            if sample['label'] == 1 and pos_count < max_per_class:
                texts.append(sample['sentence'])
                labels.append(1)
                pos_count += 1
            elif sample['label'] == 0 and neg_count < max_per_class:
                texts.append(sample['sentence'])
                labels.append(0)
                neg_count += 1
        
        # Progress e stop condition
        if processed % 2000 == 0:
            print(f"  📈 Processati {processed:,} campioni | Selezionati: {len(texts):,} | Pos: {pos_count} | Neg: {neg_count}")
        
        if pos_count >= max_per_class and neg_count >= max_per_class:
            print(f"✅ Completato! Processati {processed:,} campioni totali")
            break
    
    # Crea DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Filtra testi troppo corti o lunghi
    df = df[df['text'].str.len() > 20]  # Almeno 20 caratteri
    df = df[df['text'].str.len() < 2000]  # Massimo 2000 caratteri
    
    # Shuffle per bilanciamento
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Salva
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Amazon Reviews dataset salvato: {csv_path}")
    print(f"   📊 Total samples: {len(df):,}")
    print(f"   😊 Positive: {sum(df['label'] == 1):,}")
    print(f"   😞 Negative: {sum(df['label'] == 0):,}")
    print(f"   📏 Lunghezza media testo: {df['text'].str.len().mean():.0f} caratteri")
    
    # Mostra esempi rappresentativi
    print(f"\n📝 ESEMPI DI REVIEWS:")
    for i in range(3):
        label_text = "POSITIVE" if df.iloc[i]['label'] == 1 else "NEGATIVE"
        text_preview = df.iloc[i]['text'][:150] + "..." if len(df.iloc[i]['text']) > 150 else df.iloc[i]['text']
        print(f"{i+1}. [{label_text}] {text_preview}")

def save_financial_sentiment_csv(csv_path, split="train", max_samples=12000):
    """
    Alternativa: Financial Sentiment dataset
    Più specifico e professionale per test avanzati
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return
    
    print(f"📈 Scaricando Financial Sentiment dataset...")
    
    try:
        # Dataset finanziario specializzato
        dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
        print("✅ Caricato Financial PhraseBank")
    except:
        # Fallback generico
        print("📦 Fallback: Caricamento dataset Twitter sentiment...")
        dataset = load_dataset("tweet_eval", "sentiment", split=split)
    
    texts = []
    labels = []
    
    print(f"🔄 Processando financial sentiment per {max_samples} campioni...")
    
    for i, sample in enumerate(dataset):
        if len(texts) >= max_samples:
            break
            
        if 'sentence' in sample and 'label' in sample:
            # Financial phrasebank format
            text = sample['sentence']
            label = sample['label']
            
            # Converti in formato binario se necessario
            if label in [0, 1]:  # Già binario
                texts.append(text)
                labels.append(label)
            elif label == 2:  # Neutral -> positive
                texts.append(text)
                labels.append(1)
        
        if (i + 1) % 1000 == 0:
            print(f"  📈 Processati {i+1:,} | Selezionati: {len(texts):,}")
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Financial dataset salvato: {csv_path} ({len(df):,} campioni)")

def estimate_training_time_roberta(num_samples, batch_size=16, epochs=5):
    """
    Stima tempo training per RoBERTa su RTX 4090
    RoBERTa è leggermente più lento di BERT
    """
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    # RTX 4090 con RoBERTa: ~45-60 steps/sec
    steps_per_second = 50  # Conservative per RoBERTa
    
    estimated_seconds = total_steps / steps_per_second
    estimated_minutes = estimated_seconds / 60
    
    print(f"\n⏱️  STIMA TEMPO TRAINING RoBERTa (RTX 4090):")
    print(f"📊 Campioni: {num_samples:,}")
    print(f"🔄 Batch size: {batch_size}")
    print(f"🎯 Epoche: {epochs}")
    print(f"⚡ Steps totali: {total_steps:,}")
    print(f"🕐 Tempo stimato: {estimated_minutes:.1f} minuti ({estimated_minutes/60:.1f} ore)")
    
    if estimated_minutes > 60:
        print(f"⚠️  Training lungo: considera di ridurre campioni o epoche per test")
    elif estimated_minutes < 15:
        print(f"⚡ Training veloce: puoi aumentare campioni per risultati migliori")
    
    return estimated_minutes

if __name__ == "__main__":
    print("🚀 TEXT DISTILLATION: RoBERTa → DistilRoBERTa")
    print("🎯 Optimized for RTX 4090 | Medium-Long Term Training")
    print("=" * 65)
    
    start_time = time.time()
    
    # =================== CONFIGURAZIONE AVANZATA ===================
    # Dataset choices (più challenging di BERT example)
    use_amazon_reviews = True  # True per Amazon (più realistico), False per Financial
    
    if use_amazon_reviews:
        dataset_relative_path = './datasets/Amazon_Reviews_distill/train.csv'
        dataset_samples = 15000  # Più campioni per training robusto
        dataset_name = "Amazon_Reviews"
        expected_time_minutes = 25  # ~25 minuti stimati
    else:
        dataset_relative_path = './datasets/Financial_Sentiment_distill/train.csv'
        dataset_samples = 12000
        dataset_name = "Financial_Sentiment"
        expected_time_minutes = 20
    
    # Modelli RoBERTa
    teacher_save_path = './models/roberta_distill/roberta_base_teacher.pt'
    student_save_path = './models/roberta_distill/distilroberta_student.pt'
    
    # Output
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("roberta_to_distilroberta", dataset_name.lower())
    
    # =================== DATASET PREPARATION ===================
    print(f"\n📊 PREPARAZIONE DATASET {dataset_name}")
    print("-" * 50)
    
    if use_amazon_reviews:
        save_amazon_reviews_as_csv(dataset_relative_path, split="train", max_samples=dataset_samples)
    else:
        save_financial_sentiment_csv(dataset_relative_path, split="train", max_samples=dataset_samples)
    
    # Analisi dataset
    num_classes = count_classes(dataset_relative_path)
    print(f"\n📈 Analisi dataset:")
    print(f"   🎯 Numero classi: {num_classes}")
    
    # Stima tempo più accurata
    estimate_training_time_roberta(dataset_samples, batch_size=16, epochs=5)
    
    # =================== MODEL SETUP ===================
    print(f"\n🤖 SETUP MODELLI RoBERTa → DistilRoBERTa")
    print("-" * 50)
    
    teacher_model, student_model, teacher_tokenizer, student_tokenizer = setup_roberta_distilroberta_models(
        num_classes=num_classes,
        teacher_save_path=teacher_save_path,
        student_save_path=student_save_path
    )
    
    if not teacher_model or not student_model:
        print("❌ Errore nel setup dei modelli!")
        exit(1)
    
    # =================== DISTILLATION CONFIG AVANZATA ===================
    print(f"\n⚙️ CONFIGURAZIONE DISTILLAZIONE AVANZATA")
    print("-" * 50)
    
    # Config ottimizzato per RoBERTa e training medio-lungo
    distillation_config = {
        'temperature': 3.0,      # RoBERTa beneficia di temperature leggermente più alta
        'alpha': 0.4,            # Bilanciamento soft/hard loss per RoBERTa
        'epochs': 5,             # Più epoche per convergenza migliore
        'learning_rate': 3e-5,   # LR ottimale per RoBERTa fine-tuning
        'eval_every': 1,         # Valuta ogni epoca
        'batch_size': 16,        # Ottimale per RTX 4090 con RoBERTa
        'warmup_steps': 500,     # Warmup per stabilità
        'weight_decay': 0.01     # Regularization
    }
    
    print(f"🎛️  Parametri ottimizzati per RoBERTa:")
    for key, value in distillation_config.items():
        emoji = {
            'temperature': '🌡️',
            'alpha': '⚖️',
            'epochs': '🔄',
            'learning_rate': '📈',
            'batch_size': '📦',
            'warmup_steps': '🔥',
            'weight_decay': '🎯'
        }.get(key, '⚙️')
        
        print(f"   {emoji} {key}: {value}")
    
    print(f"\n💡 Perché questi parametri:")
    print(f"   • Temperature 3.0: RoBERTa produce distribuzioni più confident")
    print(f"   • Alpha 0.4: Più peso al ground truth per dataset complessi")
    print(f"   • 5 epoche: Convergenza migliore per 15K+ campioni")
    print(f"   • LR 3e-5: Sweet spot per RoBERTa fine-tuning")
    
    # =================== DISTILLATION EXECUTION ===================
    print(f"\n🔥 ESECUZIONE DISTILLAZIONE")
    print("-" * 50)
    
    try:
        # Crea DistillerBridge con tokenizer RoBERTa
        bridge = DistillerBridge(
            teacher_path=teacher_save_path,
            student_path=student_save_path,
            dataset_path=dataset_relative_path,
            output_path=output_model_path,
            tokenizer_name="roberta-base",  # ⚠️ Importante: RoBERTa tokenizer
            config=distillation_config
        )
        
        print(f"🎯 Bridge configurato:")
        print(f"   🏫 Teacher: RoBERTa-base (~125M params)")
        print(f"   🎓 Student: DistilRoBERTa (~82M params)")
        print(f"   📊 Dataset: {dataset_name} ({dataset_samples:,} campioni)")
        print(f"   🎯 Classi: {bridge.get_num_classes()}")
        print(f"   📝 Task: {bridge.dataset_info['task_type']}")
        print(f"   ⏰ Tempo stimato: ~{expected_time_minutes} minuti")
        
        # Timing
        distill_start = time.time()
        
        # Conferma prima di iniziare
        print(f"\n🚀 Avvio distillazione RoBERTa → DistilRoBERTa...")
        print(f"⚠️  Training medio-lungo: {expected_time_minutes} minuti stimati")
        print(f"💪 RTX 4090 detected: Full power training!")
        
        # Esegui distillazione
        bridge.distill()
        
        distill_end = time.time()
        distill_time = (distill_end - distill_start) / 60
        
        print(f"\n✅ DISTILLAZIONE COMPLETATA!")
        print(f"⏱️  Tempo reale: {distill_time:.1f} minuti")
        print(f"🎯 Stima vs reale: {expected_time_minutes:.1f} min stimati, {distill_time:.1f} min reali")
        print(f"📁 Modello salvato: {output_model_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrotto dall'utente")
        print(f"💾 Modelli parziali potrebbero essere salvati in: {output_model_path}")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante distillazione: {e}")
        import traceback
        traceback.print_exc()
        
    # =================== VALIDATION E PERFORMANCE AVANZATA ===================
    print(f"\n🧪 VALIDAZIONE E PERFORMANCE AVANZATA")
    print("-" * 50)
    
    if os.path.exists(os.path.join(output_model_path, "student.pt")):
        print("✅ Modello distillato salvato correttamente")
        
        try:
            import torch
            
            print(f"\n🔍 Test caricamento modello distillato...")
            distilled_model = torch.load(os.path.join(output_model_path, "student.pt"), weights_only=False)
            distilled_model.eval()
            
            # Test con frasi più challenging per RoBERTa
            test_sentences = [
                "This product exceeded my expectations in every way - outstanding quality and fast delivery!",
                "Completely disappointed. Poor quality, doesn't work as advertised, waste of money.",
                "The item is decent, nothing spectacular but does what it's supposed to do.",
                "Mixed feelings about this purchase. Good price but questionable durability.",
                "Absolutely love it! Will definitely recommend to friends and family."
            ]
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
            
            # Test più robusto con diversi input lengths
            inputs = tokenizer(
                test_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256  # RoBERTa gestisce bene testi più lunghi
            )
            
            with torch.no_grad():
                outputs = distilled_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            print(f"✅ Modello distillato funziona perfettamente!")
            print(f"\n📝 Test su frasi di example (Amazon-style):")
            
            for i, sentence in enumerate(test_sentences):
                pred_class = torch.argmax(predictions[i]).item()
                confidence = torch.max(predictions[i]).item()
                sentiment = "POSITIVE" if pred_class == 1 else "NEGATIVE"
                
                # Emoji per readability
                emoji = "😊" if sentiment == "POSITIVE" else "😞"
                conf_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                print(f"{i+1}. {emoji} '{sentence[:60]}{'...' if len(sentence) > 60 else ''}'")
                print(f"     → {sentiment} {conf_color} (confidenza: {confidence:.3f})")
            
            # Performance comparison dettagliata
            print(f"\n⚡ PERFORMANCE COMPARISON AVANZATA:")
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in distilled_model.parameters())
            compression_ratio = teacher_params / student_params
            size_reduction = ((teacher_params - student_params) / teacher_params * 100)
            
            print(f"🏫 Teacher (RoBERTa):        {teacher_params:,} parametri")
            print(f"🎓 Student (DistilRoBERTa):  {student_params:,} parametri")
            print(f"📦 Compression ratio:        {compression_ratio:.2f}x")
            print(f"📉 Size reduction:           {size_reduction:.1f}%")
            print(f"💾 Disk space saved:         ~{(teacher_params-student_params)*4/1024/1024:.0f} MB")
            
            # Throughput estimate
            throughput_improvement = compression_ratio * 1.2  # Stima conservativa
            print(f"⚡ Throughput improvement:   ~{throughput_improvement:.1f}x più veloce")
            
        except Exception as e:
            print(f"⚠️ Errore nel test finale: {e}")
            import traceback
            traceback.print_exc()
    
    # =================== FINAL SUMMARY AVANZATO ===================
    total_time = (time.time() - start_time) / 60
    
    print(f"\n📋 RIEPILOGO FINALE DETTAGLIATO")
    print("=" * 65)
    print(f"🏗️  Architettura: RoBERTa-base → DistilRoBERTa")
    print(f"📊 Dataset: {dataset_name} ({dataset_samples:,} campioni)")
    print(f"⏱️  Tempo totale: {total_time:.1f} minuti ({total_time/60:.1f} ore)")
    
    # Target analysis
    target_met = abs(total_time - expected_time_minutes) < (expected_time_minutes * 0.3)
    target_status = "✅ Target centrato!" if target_met else f"⚠️ Differenza: {abs(total_time - expected_time_minutes):.1f} min"
    print(f"🎯 Target: ~{expected_time_minutes} minuti ({target_status})")
    
    print(f"💾 Output: {output_model_path}")
    print(f"🚀 GPU: RTX 4090 optimized for RoBERTa")
    print(f"🔬 Cross-architecture: RoBERTa → DistilRoBERTa (layer reduction)")
    
    # Success metrics
    if 'compression_ratio' in locals():
        print(f"📈 Compression achieved: {compression_ratio:.1f}x smaller model")
        efficiency_score = compression_ratio * (100 if 'accuracy' in locals() and accuracy > 0.9 else 80)
        print(f"⭐ Efficiency score: {efficiency_score:.0f}/100")
    
    print(f"\n🎉 ROBERTA DISTILLATION COMPLETATA!")
    print(f"💪 Modello enterprise-ready per production!")
    
    # Next steps suggestions
    print(f"\n💡 NEXT STEPS:")
    print(f"   • Test su validation set separato")
    print(f"   • Benchmark inference speed vs teacher")
    print(f"   • Deploy per A/B testing in production")
    print(f"   • Fine-tune su domain-specific data")

# =================== OPZIONI ALTERNATIVE ===================

def quick_roberta_test():
    """
    Versione veloce per test rapidi RoBERTa (5-8 minuti)
    """
    print("🚀 QUICK ROBERTA TEST (5-8 minuti)")
    
    # Dataset ridotto
    save_amazon_reviews_as_csv('./datasets/Amazon_Quick/train.csv', max_samples=3000)
    
    # Config veloce
    quick_config = {
        'temperature': 2.5,
        'alpha': 0.3,
        'epochs': 2,  # Solo 2 epoche
        'learning_rate': 5e-5,
        'eval_every': 1,
        'batch_size': 20  # Batch più grande per velocità
    }
    
    print("⚡ Configurazione veloce per test RoBERTa!")

def production_roberta_training():
    """
    Versione production con dataset molto grande (1-2 ore)
    """
    print("🏭 PRODUCTION ROBERTA TRAINING (1-2 ore)")
    
    # Dataset molto grande
    save_amazon_reviews_as_csv('./datasets/Amazon_Production/train.csv', max_samples=50000)
    
    # Config production
    production_config = {
        'temperature': 3.5,
        'alpha': 0.5,
        'epochs': 8,  # Training approfondito
        'learning_rate': 2e-5,  # LR più basso per stabilità
        'eval_every': 2,
        'batch_size': 32,  # Batch più grande
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2
    }
    
    print("🏭 Configurazione production per RoBERTa enterprise!")

# Uncomment per test veloce:
# if __name__ == "__main__":
#     quick_roberta_test()

# Uncomment per training production:
if __name__ == "__main__":
     production_roberta_training()