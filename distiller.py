from factories.adapter_factory import AdapterFactory
from utils.save_model import save_model
import os
import torch
from config import TEMPERATURE, ALPHA
from tqdm import tqdm


class DistillerBridge:
    """
    Main bridge class for knowledge distillation workflow.
    Manages dataset loading, model setup, device management, and training pipeline.
    """
    
    def __init__(self, teacher_path, student_path, dataset_path, output_path, 
                 tokenizer_name=None, config=None):
        """
        Initialize distiller bridge with automatic dataset analysis and device setup.
        
        Args:
            teacher_path: Path to pre-trained teacher model
            student_path: Path to student model to be trained
            dataset_path: Path to training dataset CSV
            output_path: Directory where trained model and metadata will be saved
            tokenizer_name: HuggingFace tokenizer name (optional, for text tasks)
            config: Training configuration dict (optional, uses defaults if None)
        """
        print("[BRIDGE] === DISTILLER BRIDGE INITIALIZATION ===")
        
        # Step 1: Analyze dataset to obtain information
        print("[BRIDGE] Analyzing dataset...")
        self.dataset_adapter, self.dataset_info = AdapterFactory.create_dataset_adapter(
            dataset_path, tokenizer_name
        )
        
        # Step 2: Extract number of classes automatically
        self.num_classes = self.dataset_info['num_classes']
        print(f"[BRIDGE] Detected number of classes: {self.num_classes}")
        
        # Step 3: Create model adapters
        self.student_adapter = AdapterFactory.create_model_adapter(student_path)
        self.teacher_adapter = AdapterFactory.create_model_adapter(teacher_path)
        
        # Step 4: Setup device management
        self.device = self._setup_device()
        print(f"[BRIDGE] Device configured: {self.device}")
        
        # Step 5: Move models to correct device
        self._move_models_to_device()
        
        # Step 6: Configuration setup
        if config is None:
            self.config = {
                'temperature': TEMPERATURE,
                'alpha': ALPHA,
                'num_classes': self.num_classes,
                'epochs': 3,
                'learning_rate': 1e-4
            }
            print("[BRIDGE] Default config created")
        else:
            self.config = config.copy()
            self.config['num_classes'] = self.num_classes
            print("[BRIDGE] Provided config updated with automatic class count")
        
        print(f"[BRIDGE] Final config: {self.config}")
        
        # Step 7: Create task adapter
        print("[BRIDGE] Creating task adapter...")
        self.task_adapter = AdapterFactory.create_task_adapter(
            dataset_path, 
            self.config,
            self.teacher_adapter.model, 
            self.student_adapter.model,
            self.dataset_info
        )
        
        self.output_path = output_path
        
        print(f"[BRIDGE] === CONFIGURATION COMPLETE ===")
        print(f"  - Task: {self.dataset_info['task_type']}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Samples: {self.dataset_info['num_samples']}")
        print(f"  - Device: {self.device}")
        print(f"  - Task Handler: {type(self.task_adapter).__name__}")

    def _setup_device(self):
        """
        Configure appropriate device (CUDA GPU if available, otherwise CPU).
        
        Returns:
            torch.device: Configured device
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[BRIDGE] CUDA available: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print(f"[BRIDGE] Using CPU")
        
        return device
    
    def _move_models_to_device(self):
        """
        Move teacher and student models to the configured device.
        Updates device references in adapters if they have device attributes.
        """
        print(f"[BRIDGE] Moving models to {self.device}...")
        
        # Move teacher model
        self.teacher_adapter.model = self.teacher_adapter.model.to(self.device)
        print(f"[BRIDGE] Teacher model on {self.device}")
        
        # Move student model
        self.student_adapter.model = self.student_adapter.model.to(self.device)
        print(f"[BRIDGE] Student model on {self.device}")
        
        # Update device in adapters if necessary
        if hasattr(self.teacher_adapter, 'device'):
            self.teacher_adapter.device = self.device
        if hasattr(self.student_adapter, 'device'):
            self.student_adapter.device = self.device

    def get_num_classes(self):
        """
        Get the number of classes in the dataset.
        
        Returns:
            int: Number of classes
        """
        return self.num_classes
    
    def get_config(self):
        """
        Get the complete configuration dictionary.
        
        Returns:
            dict: Configuration with all training parameters
        """
        return self.config

    def distill(self):
        """
        Execute knowledge distillation training loop with device management.
        Trains student model using teacher's soft targets and ground truth labels.
        """
        print("[BRIDGE] === START DISTILLATION ===")
        
        # Step 1: Prepare dataloader using task handler
        print("[BRIDGE] Preparing dataloader...")
        data_loader = self.task_adapter.prepare_dataset(self.dataset_adapter)
        
        # Step 2: Setup training
        optimizer = torch.optim.Adam(
            self.student_adapter.model.parameters(), 
            lr=self.config.get('learning_rate', 1e-4)
        )
        
        epochs = self.config.get('epochs', 3)
        
        # Step 3: Setup model modes
        print(f"[BRIDGE] Configuring model modes...")
        self.teacher_adapter.model.eval()   # Teacher in evaluation mode
        self.student_adapter.model.train()  # Student in training mode
        print(f"[BRIDGE] Teacher: eval(), Student: train()")
        
        print(f"[BRIDGE] Starting training for {epochs} epochs on device: {self.device}")
        
        # Step 4: Training loop with device management
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Ensure student is in training mode
            self.student_adapter.model.train()
            
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                try:
                    # Prepare batch with device management
                    inputs, labels = self._prepare_batch_with_device(batch)
                    
                    # Teacher forward pass (without gradient)
                    with torch.no_grad():
                        teacher_logits = self.task_adapter.forward_pass(
                            self.teacher_adapter.model, inputs
                        )
                    
                    # Student forward pass (with gradient)
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
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}'
                    })
                    
                except Exception as e:
                    print(f"[ERROR] Batch error: {e}")
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
            print(f"[BRIDGE] Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
            
            # Evaluation with device management
            if (epoch + 1) % self.config.get('eval_every', 1) == 0:
                print("[BRIDGE] Running evaluation...")
                try:
                    eval_results = self._evaluate_with_device_management(data_loader)
                    print(f"[BRIDGE] Evaluation results: {eval_results}")
                except Exception as e:
                    print(f"[WARNING] Error during evaluation: {e}")
        
        print("[BRIDGE] === DISTILLATION COMPLETE ===")
        
        # Final evaluation
        print("[BRIDGE] Final evaluation...")
        try:
            final_results = self._evaluate_with_device_management(data_loader)
            print(f"[BRIDGE] Final results: {final_results}")
        except Exception as e:
            print(f"[WARNING] Final evaluation error: {e}")
            final_results = None
        
        # Save model
        self._save_model(final_results)
    
    def _prepare_batch_with_device(self, batch):
        """
        Prepare batch by moving all tensors to the correct device.
        
        Args:
            batch: Batch data (dict or tuple format)
            
        Returns:
            Tuple of (inputs, labels) with all tensors on correct device
            
        Raises:
            ValueError: If batch format is not supported
        """
        if isinstance(batch, dict):
            # Dict format (text or image with processor)
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
            # Tuple format (standard image)
            inputs, labels = batch
            
            # Move inputs to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Move labels to device
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
                
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        return inputs, labels
    
    def _evaluate_with_device_management(self, dataloader):
        """
        Evaluate student model with correct device management.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary with accuracy, correct predictions, and total samples
        """
        # Set student to evaluation mode
        self.student_adapter.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Prepare batch with device management
                    inputs, labels = self._prepare_batch_with_device(batch)
                    
                    # Forward pass
                    logits = self.task_adapter.forward_pass(
                        self.student_adapter.model, inputs
                    )
                    
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Ensure labels and predictions are on same device
                    if labels.device != predictions.device:
                        labels = labels.to(predictions.device)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                except Exception as e:
                    print(f"[WARNING] Evaluation batch error: {e}")
                    continue
        
        # Return student to training mode
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
        """
        Save trained student model and metadata to disk.
        
        Args:
            eval_results: Optional evaluation results to include in metadata
        """
        print("[BRIDGE] Saving model...")
        
        # Move model to CPU before saving
        model_to_save = self.student_adapter.model.cpu()
        
        model_save_path = os.path.join(self.output_path, "student.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        try:
            torch.save(model_to_save, model_save_path)
            print(f"[BRIDGE] Student model saved to: {model_save_path}")
            
            # Return model to original device
            self.student_adapter.model = model_to_save.to(self.device)
            
            # Save complete metadata
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
            
            print(f"[BRIDGE] Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"[BRIDGE] Error during save: {e}")
            raise