# =================== SETUP CROSS-ARCHITECTURE MODELS ===================
# File: setup_cross_models.py

import torch
import torch.nn as nn
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification
)


def print_architecture_comparison():
    """
    Stampa confronto tra architetture Transformer vs MLP
    """
    print("\n🏗️  CROSS-ARCHITECTURE DISTILLATION SETUP")
    print("=" * 55)
    
    print("🏫 TEACHER MODEL: Transformer (DistilBERT)")
    print("   • Architettura: Multi-head attention + feedforward")
    print("   • Layers: 6 transformer blocks")
    print("   • Attention heads: 12")
    print("   • Hidden size: 768")
    print("   • Parameters: ~67M")
    print("   • Input: Tokenized text (input_ids, attention_mask)")
    
    print("\n🎓 STUDENT MODEL: Multi-Layer Perceptron (MLP)")
    print("   • Architettura: Pure feedforward network")
    print("   • Layers: 3-4 linear layers + activations")
    print("   • No attention mechanism")
    print("   • Hidden sizes: 768 → 512 → 256 → num_classes")
    print("   • Parameters: ~500K-1M (molto più piccolo)")
    print("   • Input: Aggregated embeddings from tokenizer")
    
    print("\n🔬 SFIDA CROSS-ARCHITECTURE:")
    print("   • Teacher: Processa sequenze con attention")
    print("   • Student: Processa feature aggregate senza attention")
    print("   • Knowledge transfer: Da sequential a feedforward")
    print("   • Test: Compatibility tra paradigmi diversi")

class MLPStudentModel(nn.Module):
    """
    Student MLP Model per test cross-architecture
    Input: Embeddings aggregati dal tokenizer
    Output: Logits per classification
    """
    
    def __init__(self, vocab_size=30522, embed_dim=768, hidden_dim=512, num_classes=2, dropout=0.1):
        super().__init__()
        
        # Embedding layer (simile a transformer ma più semplice)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # ✅ FIX: Config serializzabile (dict invece di classe dinamica)
        self.config = {
            'num_labels': num_classes,
            'vocab_size': vocab_size,
            'hidden_size': embed_dim
        }
        
        # Aggiungi attributi come property per compatibility
        for key, value in self.config.items():
            setattr(self, key, value)
        
        print(f"[MLP_STUDENT] Creato con {self.count_parameters():,} parametri")
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass che simula interfaccia transformer
        ma usa solo MLP interno
        """
        # Embedding dei token
        embeddings = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embeddings = self.dropout(embeddings)
        
        # Aggregazione semplice: mean pooling
        if attention_mask is not None:
            # Maschera padding
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded
            sum_embeddings = torch.sum(embeddings, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Semplice mean se no mask
            pooled = torch.mean(embeddings, dim=1)  # [batch, embed_dim]
        
        # Forward attraverso MLP
        logits = self.mlp(pooled)  # [batch, num_classes]
        
        # Return in formato compatibile con transformers
        return type('ModelOutput', (), {'logits': logits})()
    
    def count_parameters(self):
        """Conta parametri del modello"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_transformer_teacher(num_classes=2, model_name="distilbert-base-uncased"):
    """
    Crea teacher transformer (DistilBERT)
    """
    print(f"🏫 Creando teacher transformer: {model_name}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            problem_type="single_label_classification"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        teacher_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Teacher transformer creato: {teacher_params:,} parametri")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Errore creazione teacher transformer: {e}")
        return None, None

def create_mlp_student(num_classes=2, vocab_size=30522):
    """
    Crea student MLP
    """
    print(f"🎓 Creando student MLP...")
    
    try:
        model = MLPStudentModel(
            vocab_size=vocab_size,
            embed_dim=768,  # Compatibile con DistilBERT
            hidden_dim=512,
            num_classes=num_classes,
            dropout=0.1
        )
        
        # Usa stesso tokenizer del teacher per compatibility
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        print(f"✅ Student MLP creato: architettura feedforward")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Errore creazione student MLP: {e}")
        return None, None

def test_model_compatibility(teacher_model, student_model, tokenizer):
    """
    Test preliminare di compatibility tra teacher e student
    """
    print(f"\n🧪 TEST PRELIMINARY COMPATIBILITY")
    print("-" * 40)
    
    try:
        # Test input
        test_text = "This is a compatibility test for cross-architecture distillation."
        
        # Tokenize
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        print(f"📝 Test text: '{test_text[:50]}...'")
        print(f"🔤 Tokenized shape: {inputs['input_ids'].shape}")
        
        # Test teacher
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        print(f"🏫 Teacher output: {teacher_logits.shape}")
        print(f"   Values: {teacher_logits[0].tolist()}")
        
        # Test student
        student_model.eval()
        with torch.no_grad():
            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits
        
        print(f"🎓 Student output: {student_logits.shape}")
        print(f"   Values: {student_logits[0].tolist()}")
        
        # Compatibility check
        if teacher_logits.shape == student_logits.shape:
            print(f"✅ Output shapes compatible: {teacher_logits.shape}")
            
            # Test diversi input lengths
            print(f"\n🔄 Testing different input lengths...")
            
            test_texts = [
                "Short",
                "This is a medium length sentence for testing.",
                "This is a much longer sentence that tests the model's ability to handle variable length inputs with proper padding and attention masking."
            ]
            
            for i, text in enumerate(test_texts):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                
                with torch.no_grad():
                    t_out = teacher_model(**inputs).logits
                    s_out = student_model(**inputs).logits
                
                if t_out.shape == s_out.shape:
                    print(f"   ✅ Length {len(text)} chars: {t_out.shape}")
                else:
                    print(f"   ❌ Length {len(text)} chars: Shape mismatch!")
                    return False
            
            print(f"🎯 PRELIMINARY COMPATIBILITY: ✅ PASS")
            return True
            
        else:
            print(f"❌ Output shapes incompatible!")
            print(f"   Teacher: {teacher_logits.shape}")
            print(f"   Student: {student_logits.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_transformer_mlp_models(num_classes=2, teacher_save_path=None, student_save_path=None):
    """
    Setup completo per distillazione Transformer → MLP
    
    Args:
        num_classes (int): Numero di classi per classification
        teacher_save_path (str): Path per salvare/caricare teacher
        student_save_path (str): Path per salvare/caricare student
    
    Returns:
        tuple: (teacher_model, student_model, teacher_tokenizer, student_tokenizer)
    """
    
    print_architecture_comparison()
    
    # =================== TEACHER SETUP ===================
    print(f"\n🏫 TEACHER SETUP: Transformer")
    
    try:
        if teacher_save_path and os.path.exists(teacher_save_path):
            print(f"📁 Caricamento teacher esistente: {teacher_save_path}")
            teacher_model = torch.load(teacher_save_path, weights_only=False)
            teacher_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            print("🔄 Creando nuovo teacher transformer...")
            teacher_model, teacher_tokenizer = create_transformer_teacher(num_classes)
            
            if teacher_model and teacher_save_path:
                os.makedirs(os.path.dirname(teacher_save_path), exist_ok=True)
                torch.save(teacher_model, teacher_save_path)
                print(f"💾 Teacher salvato: {teacher_save_path}")
        
        if not teacher_model:
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Errore setup teacher: {e}")
        return None, None, None, None

    # =================== STUDENT SETUP ===================
    print(f"\n🎓 STUDENT SETUP: MLP")

    try:
        if student_save_path and os.path.exists(student_save_path):
            print(f"📁 Caricamento student esistente: {student_save_path}")
            student_model = torch.load(student_save_path, weights_only=False)
            student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            print("🔄 Creando nuovo student MLP...")
            
            # Determina vocab_size dal teacher tokenizer
            vocab_size = len(teacher_tokenizer.get_vocab()) if teacher_tokenizer else 30522
            print(f"🔍 Debug: Vocab size rilevato: {vocab_size}")
            
            student_model, student_tokenizer = create_mlp_student(
                num_classes=num_classes, 
                vocab_size=vocab_size
            )
            
            if student_model and student_save_path:
                os.makedirs(os.path.dirname(student_save_path), exist_ok=True)
                
                # ✅ FIX COMPLETO PER ERRORE PICKLE
                print(f"🔄 Tentativo salvataggio student...")
                try:
                    torch.save(student_model, student_save_path)
                    print(f"💾 Student salvato con successo: {student_save_path}")
                    
                except Exception as save_error:
                    print(f"❌ Errore salvataggio: {save_error}")
                    print(f"🔧 Creando modello con config serializzabile...")
                    
                    class SerializableMLPStudentModel(nn.Module):
                        """
                        Identico all'MLP originale ma con config serializzabile
                        """
                        def __init__(self, original_model):
                            super().__init__()
                            
                            # Copia TUTTI i parametri e layer dall'originale
                            self.embedding = original_model.embedding
                            self.dropout = original_model.dropout  
                            self.mlp = original_model.mlp
                            
                            # ✅ Config serializzabile (dict semplice invece di classe dinamica)
                            self.config = {
                                'num_labels': getattr(original_model.config, 'num_labels', num_classes) if hasattr(original_model.config, 'num_labels') else getattr(original_model, 'num_labels', num_classes),
                                'vocab_size': getattr(original_model.config, 'vocab_size', vocab_size) if hasattr(original_model.config, 'vocab_size') else getattr(original_model, 'vocab_size', vocab_size),
                                'hidden_size': getattr(original_model.config, 'hidden_size', 768) if hasattr(original_model.config, 'hidden_size') else getattr(original_model, 'hidden_size', 768)
                            }
                            
                            # Aggiungi attributi come property per compatibility
                            for key, value in self.config.items():
                                setattr(self, key, value)
                        
                        def forward(self, input_ids, attention_mask=None, **kwargs):
                            """Forward identico all'originale"""
                            # Embedding dei token
                            embeddings = self.embedding(input_ids)
                            embeddings = self.dropout(embeddings)
                            
                            # Aggregazione con attention mask
                            if attention_mask is not None:
                                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                                embeddings = embeddings * mask_expanded
                                sum_embeddings = torch.sum(embeddings, 1)
                                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                                pooled = sum_embeddings / sum_mask
                            else:
                                pooled = torch.mean(embeddings, dim=1)
                            
                            # Forward attraverso MLP
                            logits = self.mlp(pooled)
                            
                            # Return compatibile con transformers
                            return type('ModelOutput', (), {'logits': logits})()
                        
                        def count_parameters(self):
                            """Mantieni metodo di utilità"""
                            return sum(p.numel() for p in self.parameters() if p.requires_grad)
                    
                    # Crea versione serializzabile
                    print(f"🔄 Clonando modello con config serializzabile...")
                    serializable_model = SerializableMLPStudentModel(student_model)
                    
                    # Verifica che sia identico
                    print(f"🔍 Verifica identità modelli...")
                    test_input = {
                        'input_ids': torch.randint(0, 1000, (1, 10)),
                        'attention_mask': torch.ones(1, 10)
                    }
                    
                    with torch.no_grad():
                        original_output = student_model(**test_input)
                        serializable_output = serializable_model(**test_input)
                        
                        # Verifica che gli output siano identici
                        if torch.allclose(original_output.logits, serializable_output.logits, atol=1e-6):
                            print(f"✅ Modello clonato è identico all'originale")
                            
                            # Salva la versione serializzabile
                            torch.save(serializable_model, student_save_path)
                            
                            # Verifica salvataggio
                            file_size = os.path.getsize(student_save_path)
                            print(f"💾 Modello serializzabile salvato: {student_save_path} ({file_size} bytes)")
                            
                            # Test caricamento
                            try:
                                loaded_model = torch.load(student_save_path, weights_only=False)
                                print(f"✅ Test caricamento: SUCCESS")
                                
                                # Sostituisci il modello originale con quello serializzabile
                                student_model = serializable_model
                                
                            except Exception as load_test_error:
                                print(f"❌ Test caricamento fallito: {load_test_error}")
                                
                        else:
                            print(f"❌ Modello clonato NON è identico - differenza: {torch.max(torch.abs(original_output.logits - serializable_output.logits))}")
        
        # ✅ VALIDATION STUDENT MODEL
        if student_model:
            print(f"🔍 Validazione student model...")
            try:
                # Test del forward pass
                test_input = {
                    'input_ids': torch.randint(0, 1000, (1, 10)),
                    'attention_mask': torch.ones(1, 10)
                }
                
                with torch.no_grad():
                    test_output = student_model(**test_input)
                    print(f"✅ Student validation OK - Output: {test_output.logits.shape}")
                    
                    # Debug config dopo validation
                    if hasattr(student_model, 'config'):
                        print(f"🔍 Config dopo validation: {type(student_model.config)}")
                    
            except Exception as val_error:
                print(f"❌ Student validation failed: {val_error}")
                print(f"🔍 Debug validation error:")
                import traceback
                traceback.print_exc()
                student_model = None
        
        if not student_model:
            print(f"❌ Student model non disponibile")
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Errore setup student: {e}")
        print(f"🔍 Debug exception type: {type(e)}")
        print(f"🔍 Debug exception args: {e.args}")
        
        # Debug dettagliato per errori pickle
        if "pickle" in str(e).lower() or "config" in str(e).lower():
            print(f"🔧 Errore di serialization rilevato!")
            print(f"💡 Suggerimento: Problema con config class dinamica")
            if 'student_model' in locals() and hasattr(student_model, 'config'):
                print(f"🔍 Config problematico: {student_model.config}")
                print(f"🔍 Config class: {student_model.config.__class__}")
        
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # =================== COMPATIBILITY VERIFICATION ===================
    print(f"\n🔬 VERIFICA COMPATIBILITY CROSS-ARCHITECTURE")
    
    # Test compatibility preliminare
    compatibility_ok = test_model_compatibility(
        teacher_model, student_model, teacher_tokenizer
    )
    
    if not compatibility_ok:
        print(f"❌ Test compatibility fallito!")
        return None, None, None, None
    
    # =================== DETAILED COMPARISON ===================
    print(f"\n📊 CONFRONTO DETTAGLIATO MODELLI:")
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params
    
    print(f"🏫 Teacher (Transformer):    {teacher_params:,} parametri")
    print(f"🎓 Student (MLP):            {student_params:,} parametri")
    print(f"📦 Compression ratio:        {compression_ratio:.1f}x")
    print(f"📉 Size reduction:           {((teacher_params-student_params)/teacher_params*100):.1f}%")
    
    # =================== ARCHITECTURE DETAILS ===================
    print(f"\n🏗️  DETTAGLI ARCHITETTURALI:")
    
    if hasattr(teacher_model, 'config'):
        teacher_config = teacher_model.config
        print(f"Teacher (Transformer):")
        print(f"  - {getattr(teacher_config, 'num_hidden_layers', 'N/A')} transformer layers")
        print(f"  - {getattr(teacher_config, 'hidden_size', 'N/A')} hidden size")
        print(f"  - {getattr(teacher_config, 'num_attention_heads', 'N/A')} attention heads")
        print(f"  - {getattr(teacher_config, 'vocab_size', 'N/A'):,} vocab size")
        print(f"  - Multi-head attention mechanism")
    
    if hasattr(student_model, 'config'):
        student_config = student_model.config
        print(f"\nStudent (MLP):")
        print(f"  - Pure feedforward architecture")
        if isinstance(student_config, dict):
            print(f"  - {student_config.get('hidden_size', 'N/A')} input embedding dim")
            print(f"  - {student_config.get('vocab_size', 'N/A'):,} vocab size (shared)")
        else:
            print(f"  - {getattr(student_config, 'hidden_size', 'N/A')} input embedding dim")
            print(f"  - {getattr(student_config, 'vocab_size', 'N/A'):,} vocab size (shared)")
        print(f"  - No attention mechanism")
        print(f"  - Mean pooling aggregation")
    
    # =================== INPUT COMPATIBILITY TEST ===================
    print(f"\n🔧 TEST INPUT COMPATIBILITY:")
    
    # Test diversi tipi di input che il sistema potrebbe ricevere
    test_scenarios = [
        ("input_ids only", {"input_ids": torch.randint(0, 1000, (2, 10))}),
        ("input_ids + attention_mask", {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10)
        }),
        ("with padding", {
            "input_ids": torch.tensor([[101, 2023, 2003, 102, 0, 0], [101, 7592, 2088, 1999, 102, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
        })
    ]
    
    all_compatible = True
    
    for scenario_name, test_input in test_scenarios:
        try:
            with torch.no_grad():
                teacher_out = teacher_model(**test_input)
                student_out = student_model(**test_input)
            
            if teacher_out.logits.shape == student_out.logits.shape:
                print(f"   ✅ {scenario_name}: Compatible")
            else:
                print(f"   ❌ {scenario_name}: Shape mismatch")
                all_compatible = False
                
        except Exception as e:
            print(f"   ❌ {scenario_name}: Error - {e}")
            all_compatible = False
    
    # =================== FINAL VALIDATION ===================
    if all_compatible:
        print(f"\n🎯 CROSS-ARCHITECTURE SETUP: ✅ SUCCESS")
        print(f"   • Input compatibility: ✅ All scenarios pass")
        print(f"   • Output alignment: ✅ Shape matching confirmed")
        print(f"   • Architecture diversity: ✅ Transformer → MLP")
        print(f"   • Compression potential: ✅ {compression_ratio:.1f}x reduction")
        print(f"   • Ready for distillation: 🚀 GO!")
        
        return teacher_model, student_model, teacher_tokenizer, student_tokenizer
    else:
        print(f"\n❌ CROSS-ARCHITECTURE SETUP: FAILED")
        print(f"   • Compatibility issues detected")
        print(f"   • Check model implementations")
        return None, None, None, None

def verify_cross_architecture_setup():
    """
    Verifica rapida che tutto sia configurato per cross-architecture
    """
    print("\n🧪 VERIFICA CROSS-ARCHITECTURE SETUP")
    print("-" * 45)
    
    try:
        # Test import necessari
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: disponibile")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️ CUDA non disponibile - training su CPU")
        
        # Test creazione modelli base
        print(f"🧪 Test creazione modelli...")
        
        # Test teacher
        try:
            teacher, teacher_tok = create_transformer_teacher(num_classes=2)
            if teacher:
                print(f"✅ Teacher transformer: Creabile")
            else:
                print(f"❌ Teacher transformer: Fallito")
        except Exception as e:
            print(f"❌ Teacher transformer: {e}")
        
        # Test student
        try:
            student, student_tok = create_mlp_student(num_classes=2)
            if student:
                print(f"✅ Student MLP: Creabile")
            else:
                print(f"❌ Student MLP: Fallito")
        except Exception as e:
            print(f"❌ Student MLP: {e}")
        
        print(f"🎉 Verifica cross-architecture completata!")
        return True
        
    except Exception as e:
        print(f"❌ Errore verifica: {e}")
        return False

def get_cross_architecture_recommendations():
    """
    Raccomandazioni per training cross-architecture ottimale
    """
    print("\n💡 RACCOMANDAZIONI CROSS-ARCHITECTURE")
    print("=" * 50)
    
    print("🎯 HYPERPARAMETERS OTTIMALI:")
    print("   • Learning Rate: 5e-5 to 1e-4 (più alto per MLP)")
    print("   • Temperature: 3.5-5.0 (alta per smooth knowledge)")
    print("   • Alpha: 0.4-0.6 (bilanciato per architetture diverse)")
    print("   • Batch Size: 16-32 (standard)")
    print("   • Epochs: 4-8 (MLP converge più veloce)")
    
    print("\n⚡ PERFORMANCE TIPS:")
    print("   • MLP student converge più velocemente")
    print("   • Attention knowledge è hard da trasferire")
    print("   • Focus su feature-level distillation")
    print("   • Mean pooling vs attention pooling")
    
    print("\n🔧 TROUBLESHOOTING CROSS-ARCH:")
    print("   • Se student converge troppo veloce: riduci LR")
    print("   • Se knowledge transfer è poor: aumenta temperature")
    print("   • Se unstable training: riduci batch size")
    print("   • Se underfitting: più epochs o hidden layers")
    
    print("\n🚀 OPTIMIZATION STRATEGIES:")
    print("   • Progressive distillation: step-by-step layer reduction")
    print("   • Feature matching: intermediate layer alignment")
    print("   • Attention transfer: knowledge distillation from attention maps")
    print("   • Ensemble distillation: multiple teachers")

if __name__ == "__main__":
    # Test del modulo cross-architecture
    print("🧪 TEST CROSS-ARCHITECTURE MODULE")
    print("=" * 45)
    
    if verify_cross_architecture_setup():
        print("\n🎯 Test setup modelli cross-architecture:")
        teacher, student, teacher_tok, student_tok = setup_transformer_mlp_models(
            num_classes=2
        )
        
        if teacher and student:
            print("✅ Cross-architecture setup completato!")
            get_cross_architecture_recommendations()
        else:
            print("❌ Cross-architecture setup fallito")
    else:
        print("❌ Verifica cross-architecture fallita")