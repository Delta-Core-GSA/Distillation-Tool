# =================== SETUP ROBERTA MODELS ===================
# File: setup_roberta.py

import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    RobertaTokenizer
)


def print_model_details():
    """
    Stampa dettagli sui modelli RoBERTa vs DistilRoBERTa
    """
    print("\n🤖 SETUP ROBERTA → DISTILROBERTA DISTILLATION")
    print("=" * 50)
    
    print("📚 TEACHER MODEL: RoBERTa-base")
    print("   • Sviluppato da: Facebook AI (2019)")
    print("   • Miglioramento di BERT con:")
    print("     - Rimozione Next Sentence Prediction")
    print("     - Tokenizzazione Byte-Pair Encoding")
    print("     - Training su più dati")
    print("     - Ottimizzazioni hyperparameter")
    
    print("\n🎓 STUDENT MODEL: DistilRoBERTa-base")
    print("   • Sviluppato da: Hugging Face")
    print("   • Architettura ottimizzata:")
    print("     - 6 layers invece di 12 (50% riduzione)")
    print("     - Stessa dimensione hidden (768)")
    print("     - Stesso tokenizer di RoBERTa")
    print("     - Mantiene performance simili")
'''
def estimate_flops_roberta():
    """
    Calcola FLOPs per RoBERTa e DistilRoBERTa
    """
    print("\n⚡ ANALISI COMPUTATIONAL COMPLEXITY")
    print("-" * 40)
    
    try:
        # Crea modelli dummy per analysis
        roberta_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", 
            num_labels=2,
            torchscript=False
        )
        
        distilroberta_model = AutoModelForSequenceClassification.from_pretrained(
            "distilroberta-base",
            num_labels=2, 
            torchscript=False
        )
        
        # Input dummy per FLOPs calculation
        dummy_input = torch.randint(0, 1000, (1, 128))  # Sequence length 128
        
        # Calcola FLOPs
        roberta_flops, roberta_macs = flopth(roberta_model, inputs=(dummy_input,))
        distil_flops, distil_macs = flopth(distilroberta_model, inputs=(dummy_input,))
        
        print(f"RoBERTa-base: FLOPs={roberta_flops/1e9:.2f} GFLOPs, MACs={roberta_macs/1e6:.2f} MMACs")
        print(f"DistilRoBERTa: FLOPs={distil_flops/1e9:.2f} GFLOPs, MACs={distil_macs/1e6:.2f} MMACs")
        print(f"⚡ Speedup FLOPs: {roberta_flops/distil_flops:.1f}x")
        print(f"⚡ Speedup MACs: {roberta_macs/distil_macs:.1f}x")
        
    except Exception as e:
        print(f"⚠️ FLOPs calculation failed: {e}")
        print("📊 Estimated values:")
        print("RoBERTa-base: ~22 GFLOPs")
        print("DistilRoBERTa: ~11 GFLOPs")
        print("⚡ Estimated speedup: ~2x")
'''

def setup_roberta_distilroberta_models(num_classes=2, teacher_save_path=None, student_save_path=None):
    """
    Setup completo per distillazione RoBERTa → DistilRoBERTa
    
    Args:
        num_classes (int): Numero di classi per classification
        teacher_save_path (str): Path per salvare/caricare teacher
        student_save_path (str): Path per salvare/caricare student
    
    Returns:
        tuple: (teacher_model, student_model, teacher_tokenizer, student_tokenizer)
    """
    
    print_model_details()
    #estimate_flops_roberta()
    
    print(f"\n📚 Caricamento Teacher: RoBERTa-base")
    
    # =================== TEACHER MODEL SETUP ===================
    try:
        if teacher_save_path and os.path.exists(teacher_save_path):
            print(f"📁 Caricamento teacher esistente: {teacher_save_path}")
            teacher_model = torch.load(teacher_save_path, weights_only=False)
            teacher_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        else:
            print("🔄 Caricamento RoBERTa-base da Hugging Face...")
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=num_classes,
                problem_type="single_label_classification"
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            
            # Salva se path specificato
            if teacher_save_path:
                os.makedirs(os.path.dirname(teacher_save_path), exist_ok=True)
                torch.save(teacher_model, teacher_save_path)
                print(f"💾 Teacher salvato: {teacher_save_path}")
        
        # Info teacher
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"✅ Teacher RoBERTa caricato: {teacher_params:,} parametri")
        
    except Exception as e:
        print(f"❌ Errore caricamento teacher: {e}")
        return None, None, None, None
    
    # =================== STUDENT MODEL SETUP ===================
    print(f"\n🎓 Caricamento Student: DistilRoBERTa-base")
    
    try:
        if student_save_path and os.path.exists(student_save_path):
            print(f"📁 Caricamento student esistente: {student_save_path}")
            student_model = torch.load(student_save_path, weights_only=False)
            student_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        else:
            print("🔄 Caricamento DistilRoBERTa-base da Hugging Face...")
            student_model = AutoModelForSequenceClassification.from_pretrained(
                "distilroberta-base",
                num_labels=num_classes,
                problem_type="single_label_classification"
            )
            student_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
            
            # Salva se path specificato
            if student_save_path:
                os.makedirs(os.path.dirname(student_save_path), exist_ok=True)
                torch.save(student_model, student_save_path)
                print(f"💾 Student salvato: {student_save_path}")
        
        # Info student
        student_params = sum(p.numel() for p in student_model.parameters())
        print(f"✅ Student DistilRoBERTa caricato: {student_params:,} parametri")
        
    except Exception as e:
        print(f"❌ Errore caricamento student: {e}")
        return None, None, None, None
    
    # =================== MODELS COMPARISON ===================
    print(f"\n📊 CONFRONTO MODELLI:")
    print(f"Teacher (RoBERTa):       {teacher_params:,} parametri")
    print(f"Student (DistilRoBERTa): {student_params:,} parametri")
    print(f"Compression ratio:       {teacher_params/student_params:.1f}x")
    print(f"Parameter reduction:     {((teacher_params-student_params)/teacher_params*100):.1f}%")
    
    # =================== ARCHITECTURAL DETAILS ===================
    print(f"\n🏗️  DETTAGLI ARCHITETTURALI:")
    
    if hasattr(teacher_model, 'config'):
        teacher_config = teacher_model.config
        print(f"RoBERTa-base:")
        print(f"  - {teacher_config.num_hidden_layers} transformer layers")
        print(f"  - {teacher_config.hidden_size} hidden size") 
        print(f"  - {teacher_config.num_attention_heads} attention heads")
        print(f"  - {teacher_config.vocab_size:,} vocab size")
        print(f"  - {teacher_config.max_position_embeddings:,} max sequence length")
    
    if hasattr(student_model, 'config'):
        student_config = student_model.config
        print(f"\nDistilRoBERTa:")
        print(f"  - {student_config.num_hidden_layers} transformer layers ({teacher_config.num_hidden_layers-student_config.num_hidden_layers} layers removed)")
        print(f"  - {student_config.hidden_size} hidden size (same)")
        print(f"  - {student_config.num_attention_heads} attention heads (same)")
        print(f"  - {student_config.vocab_size:,} vocab size (same)")
        print(f"  - {student_config.max_position_embeddings:,} max sequence length (same)")
    
    # =================== COMPATIBILITY TEST ===================
    print(f"\n🔍 TEST COMPATIBILITÀ:")
    
    try:
        # Test con input dummy più realistico per RoBERTa
        test_text = "This is a test sentence for RoBERTa compatibility check."
        
        # Test teacher
        teacher_inputs = teacher_tokenizer(
            test_text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
        
        # Test student  
        student_inputs = student_tokenizer(
            test_text,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            student_outputs = student_model(**student_inputs)
            student_logits = student_outputs.logits
        
        # Verifica compatibilità
        if teacher_logits.shape == student_logits.shape:
            print(f"✅ Modelli compatibili!")
            print(f"   Output shape: {teacher_logits.shape}")
            print(f"   Classi: {teacher_logits.shape[1]}")
        else:
            print(f"❌ Incompatibilità shape: Teacher {teacher_logits.shape} vs Student {student_logits.shape}")
            return None, None, None, None
            
        # Test tokenizer compatibility
        teacher_vocab_size = len(teacher_tokenizer.get_vocab())
        student_vocab_size = len(student_tokenizer.get_vocab())
        
        if teacher_vocab_size == student_vocab_size:
            print(f"   Stesso tokenizer vocabulary: ✅")
        else:
            print(f"   ⚠️ Vocabulari diversi: T={teacher_vocab_size}, S={student_vocab_size}")
        
        # Test special tokens
        teacher_special = teacher_tokenizer.special_tokens_map
        student_special = student_tokenizer.special_tokens_map
        
        if teacher_special == student_special:
            print(f"   Special tokens allineati: ✅")
        else:
            print(f"   ⚠️ Special tokens differenti")
        
    except Exception as e:
        print(f"❌ Errore test compatibilità: {e}")
        return None, None, None, None
    
    # =================== ROBERTA SPECIFIC CHECKS ===================
    print(f"\n🔧 CONTROLLI SPECIFICI ROBERTA:")
    
    # Check RoBERTa features
    print(f"   📝 Tokenizzazione: Byte-Pair Encoding (BPE)")
    print(f"   🚫 No Next Sentence Prediction task")
    print(f"   🎭 Masked Language Modeling only")
    
    # Verifica che non ci siano token_type_ids (RoBERTa non li usa)
    sample_encoding = teacher_tokenizer("Test", return_tensors="pt")
    if 'token_type_ids' not in sample_encoding:
        print(f"   ✅ No token_type_ids (corretto per RoBERTa)")
    else:
        print(f"   ⚠️ token_type_ids presente (inaspettato)")
    
    # GPU compatibility check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\n🚀 GPU COMPATIBILITY:")
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        
        # Memory requirements estimate
        teacher_memory_gb = teacher_params * 4 / 1024**3  # 4 bytes per param (float32)
        student_memory_gb = student_params * 4 / 1024**3
        total_memory_needed = (teacher_memory_gb + student_memory_gb) * 2  # x2 for gradients
        
        print(f"   Teacher memory: ~{teacher_memory_gb:.1f} GB")
        print(f"   Student memory: ~{student_memory_gb:.1f} GB") 
        print(f"   Training memory needed: ~{total_memory_needed:.1f} GB")
        
        if gpu_memory >= total_memory_needed:
            print(f"   ✅ GPU memory sufficient for training")
        else:
            print(f"   ⚠️ GPU memory might be tight, consider smaller batch size")
    
    print(f"\n🎯 MODELLI PRONTI PER DISTILLAZIONE!")
    
    return teacher_model, student_model, teacher_tokenizer, student_tokenizer

def verify_roberta_setup():
    """
    Verifica rapida che tutto sia configurato correttamente
    """
    print("\n🔍 VERIFICA SETUP ROBERTA")
    print("-" * 30)
    
    try:
        # Test import transformers
        from transformers import __version__
        print(f"✅ Transformers: v{__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️ CUDA non disponibile - training su CPU sarà lento")
        
        # Test modelli accessibili
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        print(f"✅ RoBERTa-base accessibile")
        
        tokenizer_distil = AutoTokenizer.from_pretrained("distilroberta-base")
        print(f"✅ DistilRoBERTa-base accessibile")
        
        print(f"🎉 Setup verification completato!")
        return True
        
    except Exception as e:
        print(f"❌ Errore setup: {e}")
        return False

def get_roberta_training_recommendations():
    """
    Raccomandazioni per training ottimale RoBERTa
    """
    print("\n💡 RACCOMANDAZIONI TRAINING ROBERTA")
    print("=" * 45)
    
    print("🎯 HYPERPARAMETERS OTTIMALI:")
    print("   • Learning Rate: 1e-5 to 5e-5 (più basso di BERT)")
    print("   • Batch Size: 16-32 (RTX 4090 può gestire 32)")
    print("   • Warmup: 6% of total steps")
    print("   • Weight Decay: 0.01")
    print("   • Temperature: 2.5-4.0 per distillation")
    print("   • Alpha: 0.3-0.5 (più peso al hard loss)")
    
    print("\n⚡ PERFORMANCE TIPS:")
    print("   • Usa gradient accumulation per batch effettivi più grandi")
    print("   • Mixed precision (fp16) per velocità extra")
    print("   • Sequence length 256-512 max per memory efficiency")
    print("   • Early stopping su validation loss")
    
    print("\n🔧 TROUBLESHOOTING COMUNE:")
    print("   • Se OOM: riduci batch_size o sequence length")
    print("   • Se convergenza lenta: aumenta learning rate")
    print("   • Se overfitting: aumenta weight decay")
    print("   • Se underfitting: più epoche o dati")

if __name__ == "__main__":
    # Test del modulo
    print("🧪 TEST SETUP ROBERTA MODULE")
    print("=" * 40)
    
    if verify_roberta_setup():
        print("\n🎯 Test setup modelli:")
        teacher, student, teacher_tok, student_tok = setup_roberta_distilroberta_models(
            num_classes=2
        )
        
        if teacher and student:
            print("✅ Setup completato con successo!")
            get_roberta_training_recommendations()
        else:
            print("❌ Setup fallito")
    else:
        print("❌ Verification fallita")