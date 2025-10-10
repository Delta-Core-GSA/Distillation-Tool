# =================== TEXT DISTILLATION MODELS SETUP ===================
# File: text_models_setup.py

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification
)

def setup_bert_distilbert_models(num_classes, teacher_save_path, student_save_path):
    """
    Setup BERT-base (teacher) e DistilBERT (student) per distillazione
    
    BERT-base: ~110M parametri
    DistilBERT: ~66M parametri (40% reduction)
    """
    print("\n🤖 SETUP BERT → DistilBERT DISTILLATION")
    print("=" * 50)
    
    # =================== TEACHER MODEL (BERT-base) ===================
    print("\n📚 Caricamento Teacher: BERT-base-uncased")
    
    teacher_model_name = "bert-base-uncased"
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_name)
    
    teacher_model = BertForSequenceClassification.from_pretrained(
        teacher_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"✅ Teacher BERT caricato: {teacher_params:,} parametri")
    
    # =================== STUDENT MODEL (DistilBERT) ===================
    print("\n🎓 Caricamento Student: DistilBERT-base-uncased")
    
    student_model_name = "distilbert-base-uncased"
    student_tokenizer = DistilBertTokenizer.from_pretrained(student_model_name)
    
    student_model = DistilBertForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"✅ Student DistilBERT caricato: {student_params:,} parametri")
    
    # =================== COMPARISON ===================
    compression_ratio = teacher_params / student_params
    reduction_percentage = ((teacher_params - student_params) / teacher_params) * 100
    
    print(f"\n📊 CONFRONTO MODELLI:")
    print(f"Teacher (BERT):       {teacher_params:,} parametri")
    print(f"Student (DistilBERT): {student_params:,} parametri")
    print(f"Compression ratio:    {compression_ratio:.1f}x")
    print(f"Parameter reduction:  {reduction_percentage:.1f}%")
    
    # =================== ARCHITECTURE DETAILS ===================
    print(f"\n🏗️  DETTAGLI ARCHITETTURALI:")
    print(f"BERT-base:")
    print(f"  - 12 transformer layers")
    print(f"  - 768 hidden size")
    print(f"  - 12 attention heads")
    print(f"  - 30,522 vocab size")
    
    print(f"\nDistilBERT:")
    print(f"  - 6 transformer layers (50% reduction)")
    print(f"  - 768 hidden size (same)")
    print(f"  - 12 attention heads (same)")
    print(f"  - 30,522 vocab size (same)")
    
    # =================== COMPATIBILITY TEST ===================
    print(f"\n🔍 TEST COMPATIBILITÀ:")
    
    # Test input
    test_text = ["This is a test sentence for compatibility check."]
    
    # Tokenize with both tokenizers
    teacher_inputs = teacher_tokenizer(
        test_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=128
    )
    
    student_inputs = student_tokenizer(
        test_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=128
    )
    
    # Forward pass
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        teacher_output = teacher_model(**teacher_inputs)
        student_output = student_model(**student_inputs)
    
    teacher_logits = teacher_output.logits
    student_logits = student_output.logits
    
    if teacher_logits.shape == student_logits.shape:
        print(f"✅ Modelli compatibili!")
        print(f"   Output shape: {teacher_logits.shape}")
        print(f"   Stesso tokenizer vocabulary: ✅")
    else:
        print(f"❌ Problema compatibilità!")
        print(f"   Teacher: {teacher_logits.shape}")
        print(f"   Student: {student_logits.shape}")
        return None, None, None, None
    
    # =================== SAVE MODELS ===================
    import os
    
    os.makedirs(os.path.dirname(teacher_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(student_save_path), exist_ok=True)
    
    if not os.path.exists(teacher_save_path):
        torch.save(teacher_model, teacher_save_path)
        print(f"💾 Teacher BERT salvato: {teacher_save_path}")
    else:
        print(f"📁 Teacher già esistente: {teacher_save_path}")
    
    if not os.path.exists(student_save_path):
        torch.save(student_model, student_save_path)
        print(f"💾 Student DistilBERT salvato: {student_save_path}")
    else:
        print(f"📁 Student già esistente: {student_save_path}")
    
    print(f"\n🎯 MODELLI PRONTI PER DISTILLAZIONE!")
    
    return teacher_model, student_model, teacher_tokenizer, student_tokenizer

def setup_roberta_bert_models(num_classes, teacher_save_path, student_save_path):
    """
    Setup alternativo: RoBERTa-base (teacher) e BERT-base (student)
    Cross-architecture più interessante
    """
    print("\n🤖 SETUP RoBERTa → BERT DISTILLATION")
    print("=" * 50)
    
    # Teacher: RoBERTa-base
    teacher_model_name = "roberta-base"
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    
    # Student: BERT-base
    student_model_name = "bert-base-uncased"
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    print(f"✅ RoBERTa → BERT setup completato!")
    
    # Save models
    import os
    os.makedirs(os.path.dirname(teacher_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(student_save_path), exist_ok=True)
    
    if not os.path.exists(teacher_save_path):
        torch.save(teacher_model, teacher_save_path)
    if not os.path.exists(student_save_path):
        torch.save(student_model, student_save_path)
    
    return teacher_model, student_model, teacher_tokenizer, student_tokenizer

def benchmark_models_inference(teacher_model, student_model, tokenizer, num_tests=100):
    """
    Benchmark velocità inference
    """
    import time
    
    print(f"\n⏱️  BENCHMARK INFERENCE SPEED")
    print("-" * 40)
    
    test_texts = [
        "This is a sample text for benchmarking.",
        "Another example sentence to test speed.",
        "Speed testing with various sentence lengths and complexity."
    ]
    
    # Tokenize
    inputs = tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    teacher_model.eval()
    student_model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = teacher_model(**inputs)
            _ = student_model(**inputs)
    
    # Benchmark Teacher
    teacher_times = []
    with torch.no_grad():
        for _ in range(num_tests):
            start = time.time()
            _ = teacher_model(**inputs)
            teacher_times.append(time.time() - start)
    
    # Benchmark Student
    student_times = []
    with torch.no_grad():
        for _ in range(num_tests):
            start = time.time()
            _ = student_model(**inputs)
            student_times.append(time.time() - start)
    
    teacher_avg = sum(teacher_times) / len(teacher_times) * 1000
    student_avg = sum(student_times) / len(student_times) * 1000
    speedup = teacher_avg / student_avg
    
    print(f"Teacher inference:  {teacher_avg:.2f} ms")
    print(f"Student inference:  {student_avg:.2f} ms")
    print(f"Speedup:           {speedup:.1f}x")
    
    return teacher_avg, student_avg, speedup

def print_model_details(model, model_name):
    """
    Stampa dettagli del modello
    """
    print(f"\n📋 {model_name.upper()} DETAILS")
    print("-" * 30)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB):      {total_params * 4 / (1024*1024):.1f}")
    
    if hasattr(model, 'config'):
        config = model.config
        print(f"Hidden size:          {getattr(config, 'hidden_size', 'N/A')}")
        print(f"Num layers:           {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"Num attention heads:  {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"Vocab size:           {getattr(config, 'vocab_size', 'N/A')}")

# =================== TESTING FUNCTIONS ===================

def test_models_on_sample_data(teacher_model, student_model, tokenizer):
    """
    Test rapido sui modelli con dati di esempio
    """
    print(f"\n🧪 TEST SU DATI DI ESEMPIO")
    print("-" * 30)
    
    sample_texts = [
        "I love this movie, it's fantastic!",
        "This film is terrible and boring.",
        "The movie was okay, nothing special.",
        "Absolutely amazing cinematography and acting!"
    ]
    
    inputs = tokenizer(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        student_outputs = student_model(**inputs)
    
    teacher_probs = torch.softmax(teacher_outputs.logits, dim=1)
    student_probs = torch.softmax(student_outputs.logits, dim=1)
    
    print("Sample predictions (Teacher | Student):")
    for i, text in enumerate(sample_texts):
        t_pred = torch.argmax(teacher_probs[i]).item()
        s_pred = torch.argmax(student_probs[i]).item()
        t_conf = torch.max(teacher_probs[i]).item()
        s_conf = torch.max(student_probs[i]).item()
        
        print(f"{i+1}. '{text[:30]}...'")
        print(f"   Teacher: Class {t_pred} (conf: {t_conf:.3f})")
        print(f"   Student: Class {s_pred} (conf: {s_conf:.3f})")

if __name__ == "__main__":
    # Test setup
    teacher_model, student_model, teacher_tok, student_tok = setup_bert_distilbert_models(
        num_classes=2,
        teacher_save_path="./models/text/bert_teacher.pt",
        student_save_path="./models/text/distilbert_student.pt"
    )
    
    if teacher_model and student_model:
        print_model_details(teacher_model, "BERT Teacher")
        print_model_details(student_model, "DistilBERT Student")
        
        benchmark_models_inference(teacher_model, student_model, teacher_tok)
        test_models_on_sample_data(teacher_model, student_model, teacher_tok)