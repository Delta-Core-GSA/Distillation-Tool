# =================== TEXT CLASSIFICATION TASK - VERSIONE UNIFICATA ===================
# File: dataset_by_tasking/text_classification.py

from typing import Dict, Any, Set
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch
from torch.utils.data import DataLoader

class TextClassificationTask(BaseTask):
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        super().__init__(config)
        self.task_type = TaskType.TEXT_CLASSIFICATION
        self.num_classes = config.get('num_classes', 2)
        
        self._teacher_model = teacher_model
        self._student_model = student_model
        
        # FEATURE AVANZATA: Rilevamento automatico input supportati
        self.teacher_supported_inputs = self._detect_supported_inputs(teacher_model, "Teacher")
        self.student_supported_inputs = self._detect_supported_inputs(student_model, "Student")
        self.common_inputs = self.teacher_supported_inputs & self.student_supported_inputs
        
        print(f"[TEXT_TASK] ✅ Inizializzato con {self.num_classes} classi")
        print(f"[TEXT_TASK] Teacher inputs: {sorted(self.teacher_supported_inputs)}")
        print(f"[TEXT_TASK] Student inputs: {sorted(self.student_supported_inputs)}")
        print(f"[TEXT_TASK] 🎯 Common inputs: {sorted(self.common_inputs)}")
    
    def _detect_supported_inputs(self, model: torch.nn.Module, model_name: str) -> Set[str]:
        """
        Rileva dinamicamente quali input supporta un modello
        """
        print(f"[DYNAMIC] Rilevando input per {model_name}...")
        
        # Rileva device del modello
        model_device = next(model.parameters()).device
        
        # Crea tensori base sul device corretto
        base_tensors = {
            'input_ids': torch.randint(0, 1000, (1, 10)).to(model_device),
            'attention_mask': torch.ones(1, 10).to(model_device),
            'token_type_ids': torch.zeros(1, 10, dtype=torch.long).to(model_device),
        }
        
        supported_inputs = set()
        model.eval()
        
        # ✅ FIX: Test input essenziali INSIEME (non separatamente)
        try:
            with torch.no_grad():
                # Test input_ids + attention_mask insieme (standard per transformer)
                _ = model(
                    input_ids=base_tensors['input_ids'],
                    attention_mask=base_tensors['attention_mask']
                )
            supported_inputs.update(['input_ids', 'attention_mask'])
            print(f"[DYNAMIC] ✅ {model_name} supporta: input_ids + attention_mask")
        except Exception as e:
            print(f"[DYNAMIC] ❌ {model_name} NON supporta input base: {e}")
            
            # Fallback: prova solo input_ids
            try:
                with torch.no_grad():
                    _ = model(input_ids=base_tensors['input_ids'])
                supported_inputs.add('input_ids')
                print(f"[DYNAMIC] ✅ {model_name} supporta: input_ids (solo)")
            except Exception as e2:
                print(f"[DYNAMIC] ❌ {model_name} NON supporta nemmeno input_ids: {e2}")
        
        # ✅ FIX: Test token_type_ids AGGIUNTIVO (se ha già input base)
        if 'input_ids' in supported_inputs:
            try:
                with torch.no_grad():
                    test_inputs = {
                        'input_ids': base_tensors['input_ids'],
                        'token_type_ids': base_tensors['token_type_ids']
                    }
                    # Aggiungi attention_mask se supportato
                    if 'attention_mask' in supported_inputs:
                        test_inputs['attention_mask'] = base_tensors['attention_mask']
                    
                    _ = model(**test_inputs)
                supported_inputs.add('token_type_ids')
                print(f"[DYNAMIC] ✅ {model_name} supporta: token_type_ids")
            except Exception as e:
                print(f"[DYNAMIC] ❌ {model_name} NON supporta: token_type_ids - {e}")
        
        return supported_inputs
    
    def prepare_dataset(self, dataset_adapter) -> DataLoader:
        """
        CRUCIALE: Prepara il dataset con input comuni rilevati automaticamente
        """
        if dataset_adapter.mode != "text":
            raise ValueError("Dataset must be in text mode for TextClassificationTask")
        
        # 🎯 FIX PRINCIPALE: Imposta gli input supportati nell'adapter
        print(f"[TEXT_TASK] Impostando input supportati nell'adapter: {sorted(self.common_inputs)}")
        if hasattr(dataset_adapter, 'set_supported_inputs'):
            dataset_adapter.set_supported_inputs(self.common_inputs)
        else:
            print(f"[WARNING] Dataset adapter non supporta set_supported_inputs")
        
        return dataset_adapter.get_text_loader()
    
    def forward_pass(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        """
        Forward pass avanzato con filtraggio automatico e gestione device
        """
        model_device = next(model.parameters()).device
        
        if isinstance(inputs, dict):
            # Sposta inputs sul device corretto
            device_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    device_inputs[k] = v.to(model_device)
                else:
                    device_inputs[k] = v
            
            # SICUREZZA: Filtra input non supportati
            filtered_inputs = self._safe_filter_inputs(device_inputs, model)
            
            # Forward pass per modelli transformer
            outputs = model(**filtered_inputs)
            
            # Estrai logits dall'output
            if hasattr(outputs, 'logits'):
                return outputs.logits
            else:
                return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        else:
            # Input diretto (meno comune per text)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(model_device)
            outputs = model(inputs)
            
            if hasattr(outputs, 'logits'):
                return outputs.logits
            else:
                return outputs
    
    def _safe_filter_inputs(self, inputs: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Filtro di sicurezza avanzato per evitare input non supportati
        """
        if model is self._teacher_model:
            supported = self.teacher_supported_inputs
            model_name = "Teacher"
        elif model is self._student_model:
            supported = self.student_supported_inputs
            model_name = "Student"
        else:
            # Usa input comuni come fallback
            supported = self.common_inputs
            model_name = "Unknown"
        
        filtered = {}
        for key, value in inputs.items():
            if key in supported:
                filtered[key] = value

            #else:
                #print(f"[SAFETY] Rimuovendo '{key}' per {model_name} (non supportato)")
        
        # Assicurati che ci siano almeno input_ids e attention_mask
        essential = ['input_ids', 'attention_mask']
        for key in essential:
            if key not in filtered and key in inputs:
                filtered[key] = inputs[key]
                #print(f"[SAFETY] Aggiunto input essenziale: {key}")
        
        return filtered
    
    def compute_distillation_loss(self, teacher_logits: torch.Tensor, 
                                student_logits: torch.Tensor, 
                                labels: torch.Tensor, 
                                config: Dict[str, Any]) -> torch.Tensor:
        """
        Loss di distillazione ottimizzata per text classification
        """
        import torch.nn.functional as F
        
        # Assicurati che tutti i tensori siano sullo stesso device
        if teacher_logits.device != student_logits.device:
            teacher_logits = teacher_logits.to(student_logits.device)
        if labels.device != student_logits.device:
            labels = labels.to(student_logits.device)
        
        # Parametri di distillazione
        temperature = config.get('temperature', 3.0)
        alpha = config.get('alpha', 0.8)
        
        # Soft distillation loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
        
        distillation_loss = F.kl_div(
            soft_predictions, 
            soft_targets, 
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Hard loss (standard cross entropy)
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Weighted combination
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
        
        return total_loss
    
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluation avanzata con gestione input sicura e device management
        """
        model.eval()
        correct = 0
        total = 0
        model_device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Prepara inputs spostando su device corretto
                    inputs = {}
                    for k, v in batch.items():
                        if k != 'labels':
                            if isinstance(v, torch.Tensor):
                                inputs[k] = v.to(model_device)
                            else:
                                inputs[k] = v
                    
                    labels = batch['labels'].to(model_device)
                    
                    # Usa forward_pass per consistenza e sicurezza
                    logits = self.forward_pass(model, inputs)
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Calcola accuracy
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                except Exception as e:
                    print(f"[WARNING] Errore evaluation batch text: {e}")
                    continue
        
        # Rimetti model in training mode
        model.train()
        
        if total > 0:
            return {
                'accuracy': correct / total,
                'correct': correct,
                'total': total
            }
        else:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}
    
    # =================== METODI LEGACY (per compatibilità) ===================
    
    def get_teacher_model(self) -> torch.nn.Module:
        """Ritorna il modello teacher (per compatibilità)"""
        return self._teacher_model

    def get_student_model(self) -> torch.nn.Module:
        """Ritorna il modello student (per compatibilità)"""
        return self._student_model
    
    # =================== METODI HELPER AVANZATI ===================
    
    def get_tokenizer_info(self):
        """
        Ritorna informazioni dettagliate sul tokenizer e modelli
        """
        info = {
            'task_type': 'text_classification',
            'num_classes': self.num_classes,
            'model_type': 'transformer_based',
            'supported_inputs': {
                'teacher': sorted(self.teacher_supported_inputs),
                'student': sorted(self.student_supported_inputs),
                'common': sorted(self.common_inputs)
            }
        }
        
        # Cerca di inferire il tipo di modello dal teacher
        if hasattr(self._teacher_model, 'config'):
            model_config = self._teacher_model.config
            if hasattr(model_config, 'model_type'):
                info['teacher_architecture'] = model_config.model_type
            if hasattr(model_config, 'vocab_size'):
                info['vocab_size'] = model_config.vocab_size
        
        # Informazioni sul student se disponibili
        if hasattr(self._student_model, 'config'):
            model_config = self._student_model.config
            if hasattr(model_config, 'model_type'):
                info['student_architecture'] = model_config.model_type
        
        return info
    
    def check_model_compatibility(self) -> bool:
        """
        Verifica avanzata di compatibilità tra teacher e student
        """
        try:
            # Test con input dummy usando solo input comuni
            dummy_input = {}
            if 'input_ids' in self.common_inputs:
                dummy_input['input_ids'] = torch.randint(0, 1000, (2, 10))
            if 'attention_mask' in self.common_inputs:
                dummy_input['attention_mask'] = torch.ones(2, 10)
            if 'token_type_ids' in self.common_inputs:
                dummy_input['token_type_ids'] = torch.zeros(2, 10, dtype=torch.long)
            
            if not dummy_input:
                print(f"[TEXT_TASK] ❌ Nessun input comune trovato!")
                return False
            
            teacher_logits = self.forward_pass(self._teacher_model, dummy_input)
            student_logits = self.forward_pass(self._student_model, dummy_input)
            
            # Verifica shape compatibility
            if teacher_logits.shape == student_logits.shape:
                print(f"[TEXT_TASK] ✅ Modelli compatibili: {teacher_logits.shape}")
                print(f"[TEXT_TASK] ✅ Input comuni utilizzati: {sorted(dummy_input.keys())}")
                return True
            else:
                print(f"[TEXT_TASK] ❌ Shape mismatch: Teacher {teacher_logits.shape} vs Student {student_logits.shape}")
                return False
                
        except Exception as e:
            print(f"[TEXT_TASK] ❌ Errore compatibilità: {e}")
            return False
    
    def debug_compatibility(self):
        """
        Helper completo per debug della compatibilità con diagnostics dettagliate
        """
        print(f"\n🔍 DEBUG COMPATIBILITÀ COMPLETO")
        print(f"=" * 50)
        print(f"Teacher supporta: {sorted(self.teacher_supported_inputs)}")
        print(f"Student supporta: {sorted(self.student_supported_inputs)}")
        print(f"Input comuni: {sorted(self.common_inputs)}")
        print(f"Num classi: {self.num_classes}")
        
        # Analisi input specifici
        if 'token_type_ids' not in self.common_inputs:
            print("✅ token_type_ids ESCLUSO dai comuni (raccomandato per compatibilità)")
        else:
            print("⚠️ token_type_ids INCLUSO nei comuni (potrebbe causare errori)")
        
        # Test compatibilità
        compatibility = self.check_model_compatibility()
        print(f"Test compatibilità: {'✅ PASS' if compatibility else '❌ FAIL'}")
        
        # Diagnostics aggiuntive
        essential_inputs = {'input_ids', 'attention_mask'}
        has_essentials = essential_inputs.issubset(self.common_inputs)
        print(f"Input essenziali presenti: {'✅ SÌ' if has_essentials else '❌ NO'}")
        
        print(f"=" * 50)
        return {
            'compatibility': compatibility,
            'has_essentials': has_essentials,
            'common_inputs_count': len(self.common_inputs),
            'recommended': len(self.common_inputs) >= 2 and compatibility
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Informazioni utili per il training setup
        """
        return {
            'task_type': self.task_type,
            'num_classes': self.num_classes,
            'input_compatibility': {
                'teacher_inputs': len(self.teacher_supported_inputs),
                'student_inputs': len(self.student_supported_inputs),
                'common_inputs': len(self.common_inputs)
            },
            'models_compatible': self.check_model_compatibility(),
            'recommended_config': {
                'temperature': 3.0,
                'alpha': 0.8,
                'batch_size': 16
            }
        }