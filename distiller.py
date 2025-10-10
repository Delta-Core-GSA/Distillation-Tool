from factories.adapter_factory import AdapterFactory
from utils.save_model import save_model
import os
import torch
from config import TEMPERATURE, ALPHA
from tqdm import tqdm

class DistillerBridge:
    def __init__(self, teacher_path, student_path, dataset_path, output_path, 
                 tokenizer_name=None, config=None):
        
        print("[BRIDGE] === INIZIALIZZAZIONE DISTILLER BRIDGE ===")
        
        # 1. Analizza il dataset per ottenere le info
        print("[BRIDGE] Analizzando dataset...")
        self.dataset_adapter, self.dataset_info = AdapterFactory.create_dataset_adapter(
            dataset_path, tokenizer_name
        )
        
        # 2. Estrai il numero di classi automaticamente
        self.num_classes = self.dataset_info['num_classes']
        print(f"[BRIDGE] 🎯 Numero classi rilevate: {self.num_classes}")
        
        # 3. Crea model adapter
        self.student_adapter = AdapterFactory.create_model_adapter(student_path)
        self.teacher_adapter = AdapterFactory.create_model_adapter(teacher_path)
        
        # 4. NUOVO: Setup device management
        self.device = self._setup_device()
        print(f"[BRIDGE] Device configurato: {self.device}")
        
        # 5. Sposta modelli sul device corretto
        self._move_models_to_device()
        
        # 6. Config setup
        if config is None:
            self.config = {
                'temperature': TEMPERATURE,
                'alpha': ALPHA,
                'num_classes': self.num_classes,
                'epochs': 3,
                'learning_rate': 1e-4
            }
            print("[BRIDGE] Config di default creato")
        else:
            self.config = config.copy()
            self.config['num_classes'] = self.num_classes
            print("[BRIDGE] Config fornito aggiornato con numero classi automatico")
        
        print(f"[BRIDGE] Config finale: {self.config}")
        
        # 7. Crea task adapter
        print("[BRIDGE] Creando task adapter...")
        self.task_adapter = AdapterFactory.create_task_adapter(
            dataset_path, 
            self.config,
            self.teacher_adapter.model, 
            self.student_adapter.model
        )
        
        self.output_path = output_path
        
        print(f"[BRIDGE] === CONFIGURAZIONE COMPLETATA ===")
        print(f"  - Task: {self.dataset_info['task_type']}")
        print(f"  - Classi: {self.num_classes}")
        print(f"  - Campioni: {self.dataset_info['num_samples']}")
        print(f"  - Device: {self.device}")
        print(f"  - Task Handler: {type(self.task_adapter).__name__}")

    def _setup_device(self):
        """
        NUOVO: Configura il device appropriato
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[BRIDGE] CUDA disponibile: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print(f"[BRIDGE] Usando CPU")
        
        return device
    
    def _move_models_to_device(self):
        """
        NUOVO: Sposta i modelli sul device corretto
        """
        print(f"[BRIDGE] Spostando modelli su {self.device}...")
        
        # Sposta teacher model
        self.teacher_adapter.model = self.teacher_adapter.model.to(self.device)
        print(f"[BRIDGE] ✅ Teacher model su {self.device}")
        
        # Sposta student model
        self.student_adapter.model = self.student_adapter.model.to(self.device)
        print(f"[BRIDGE] ✅ Student model su {self.device}")
        
        # Aggiorna device negli adapter se necessario
        if hasattr(self.teacher_adapter, 'device'):
            self.teacher_adapter.device = self.device
        if hasattr(self.student_adapter, 'device'):
            self.student_adapter.device = self.device

    def get_num_classes(self):
        """Getter per il numero di classi"""
        return self.num_classes
    
    def get_config(self):
        """Getter per il config completo"""
        return self.config

    def distill(self):
        """
        AGGIORNATO: Esegue la distillazione con device management corretto
        """
        print("[BRIDGE] === INIZIO DISTILLAZIONE ===")
        
        # 1. Prepara il dataloader usando il task handler
        print("[BRIDGE] Preparando dataloader...")
        data_loader = self.task_adapter.prepare_dataset(self.dataset_adapter)
        
        # 2. Setup training
        optimizer = torch.optim.Adam(
            self.student_adapter.model.parameters(), 
            lr=self.config.get('learning_rate', 1e-4)
        )
        
        epochs = self.config.get('epochs', 3)
        
        # 3. IMPORTANTE: Setup model modes
        print(f"[BRIDGE] Configurando model modes...")
        self.teacher_adapter.model.eval()   # Teacher in evaluation mode
        self.student_adapter.model.train()  # Student in training mode
        print(f"[BRIDGE] ✅ Teacher: eval(), Student: train()")
        
        print(f"[BRIDGE] Inizio training per {epochs} epoche su device: {self.device}")
        
        # 4. Training loop con device management
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Assicurati che student sia in training mode
            self.student_adapter.model.train()
            
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                try:
                    # CRUCIALE: Prepara batch con device management
                    inputs, labels = self._prepare_batch_with_device(batch)
                    
                    # Teacher forward pass (senza gradient)
                    with torch.no_grad():
                        teacher_logits = self.task_adapter.forward_pass(
                            self.teacher_adapter.model, inputs
                        )
                    
                    # Student forward pass (con gradient)
                    student_logits = self.task_adapter.forward_pass(
                        self.student_adapter.model, inputs
                    )
                    
                    # Loss computation
                    loss = self.task_adapter.compute_distillation_loss(
                        teacher_logits, student_logits, labels, self.config
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Tracking
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Aggiorna progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}'
                    })
                    
                except Exception as e:
                    print(f"[ERROR] Errore nel batch: {e}")
                    print(f"[DEBUG] Batch type: {type(batch)}")
                    if isinstance(batch, (list, tuple)):
                        print(f"[DEBUG] Batch length: {len(batch)}")
                        for i, item in enumerate(batch):
                            if hasattr(item, 'device'):
                                print(f"[DEBUG] Item {i} device: {item.device}")
                            if hasattr(item, 'shape'):
                                print(f"[DEBUG] Item {i} shape: {item.shape}")
                    raise
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"[BRIDGE] Epoch {epoch+1} completata - Avg Loss: {avg_loss:.4f}")
            
            # Evaluation con device management
            if (epoch + 1) % self.config.get('eval_every', 1) == 0:
                print("[BRIDGE] Eseguendo evaluation...")
                try:
                    eval_results = self._evaluate_with_device_management(data_loader)
                    print(f"[BRIDGE] Evaluation risultati: {eval_results}")
                except Exception as e:
                    print(f"[WARNING] Errore durante evaluation: {e}")
        
        print("[BRIDGE] === DISTILLAZIONE COMPLETATA ===")
        
        # Valutazione finale
        print("[BRIDGE] Valutazione finale...")
        try:
            final_results = self._evaluate_with_device_management(data_loader)
            print(f"[BRIDGE] Risultati finali: {final_results}")
        except Exception as e:
            print(f"[WARNING] Errore valutazione finale: {e}")
            final_results = None
        
        # Salvataggio
        self._save_model(final_results)
    
    def _prepare_batch_with_device(self, batch):
        """
        NUOVO: Prepara il batch gestendo correttamente i device
        """
        if isinstance(batch, dict):
            # Formato dict (text o image con processor)
            inputs = {}
            for k, v in batch.items():
                if k != 'labels':
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                    else:
                        inputs[k] = v
            
            labels = batch['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
                
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            # Formato tuple (image standard)
            inputs, labels = batch
            
            # Sposta inputs su device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Sposta labels su device
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
                
        else:
            raise ValueError(f"Formato batch non supportato: {type(batch)}")
        
        return inputs, labels
    
    def _evaluate_with_device_management(self, dataloader):
        """
        NUOVO: Evaluation con device management corretto
        """
        # Metti student in evaluation mode
        self.student_adapter.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Prepara batch con device management
                    inputs, labels = self._prepare_batch_with_device(batch)
                    
                    # Forward pass
                    logits = self.task_adapter.forward_pass(
                        self.student_adapter.model, inputs
                    )
                    
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Assicurati che labels e predictions siano sullo stesso device
                    if labels.device != predictions.device:
                        labels = labels.to(predictions.device)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                except Exception as e:
                    print(f"[WARNING] Errore in evaluation batch: {e}")
                    continue
        
        # Rimetti student in training mode
        self.student_adapter.model.train()
        
        if total > 0:
            return {
                'accuracy': correct / total,
                'correct': correct,
                'total': total
            }
        else:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}
    
    def _save_model(self, eval_results=None):
        """Salva il modello e i metadati"""
        print("[BRIDGE] Salvando modello...")
        
        # Sposta modello su CPU prima del salvataggio
        model_to_save = self.student_adapter.model.cpu()
        
        model_save_path = os.path.join(self.output_path, "student.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        try:
            torch.save(model_to_save, model_save_path)
            print(f"💾 Modello student salvato in: {model_save_path}")
            
            # Rimetti modello su device originale
            self.student_adapter.model = model_to_save.to(self.device)
            
            # Salva metadati completi
            metadata = {
                'model_path': model_save_path,
                'dataset_info': self.dataset_info,
                'config': self.config,
                'task_handler': type(self.task_adapter).__name__,
                'device_used': str(self.device),
                'model_parameters': sum(p.numel() for p in model_to_save.parameters()),
                'final_evaluation': eval_results
            }
            
            metadata_path = os.path.join(self.output_path, "distillation_metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"📋 Metadati salvati in: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Errore durante il salvataggio: {e}")
            raise