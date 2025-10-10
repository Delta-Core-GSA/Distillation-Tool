from typing import Dict, Any, Set
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch
from torch.utils.data import DataLoader


class TextClassificationTask(BaseTask):
    """
    Text classification task with automatic input compatibility detection.
    Dynamically detects which inputs (input_ids, attention_mask, token_type_ids)
    are supported by teacher and student models to ensure safe distillation.
    """
    
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        """
        Initialize text classification task with automatic compatibility detection.
        
        Args:
            config: Task configuration including num_classes, temperature, alpha
            teacher_model: Pre-trained teacher model for distillation
            student_model: Student model to be trained
        """
        super().__init__(config)
        self.task_type = TaskType.TEXT_CLASSIFICATION
        self.num_classes = config.get('num_classes', 2)
        
        self._teacher_model = teacher_model
        self._student_model = student_model
        
        # Automatic detection of supported inputs for both models
        self.teacher_supported_inputs = self._detect_supported_inputs(teacher_model, "Teacher")
        self.student_supported_inputs = self._detect_supported_inputs(student_model, "Student")
        self.common_inputs = self.teacher_supported_inputs & self.student_supported_inputs
        
        print(f"[TEXT_TASK] Initialized with {self.num_classes} classes")
        print(f"[TEXT_TASK] Teacher inputs: {sorted(self.teacher_supported_inputs)}")
        print(f"[TEXT_TASK] Student inputs: {sorted(self.student_supported_inputs)}")
        print(f"[TEXT_TASK] Common inputs: {sorted(self.common_inputs)}")
    
    def _detect_supported_inputs(self, model: torch.nn.Module, model_name: str) -> Set[str]:
        """
        Dynamically detect which inputs a model supports through testing.
        
        Args:
            model: Model to test
            model_name: Name for logging purposes
            
        Returns:
            Set of supported input keys (e.g., 'input_ids', 'attention_mask')
        """
        print(f"[DYNAMIC] Detecting inputs for {model_name}...")
        
        # Detect model device
        model_device = next(model.parameters()).device
        
        # Create base tensors on correct device
        base_tensors = {
            'input_ids': torch.randint(0, 1000, (1, 10)).to(model_device),
            'attention_mask': torch.ones(1, 10).to(model_device),
            'token_type_ids': torch.zeros(1, 10, dtype=torch.long).to(model_device),
        }
        
        supported_inputs = set()
        model.eval()
        
        # Test essential inputs together (standard for transformers)
        try:
            with torch.no_grad():
                _ = model(
                    input_ids=base_tensors['input_ids'],
                    attention_mask=base_tensors['attention_mask']
                )
            supported_inputs.update(['input_ids', 'attention_mask'])
            print(f"[DYNAMIC] {model_name} supports: input_ids + attention_mask")
        except Exception as e:
            print(f"[DYNAMIC] {model_name} does NOT support base inputs: {e}")
            
            # Fallback: try only input_ids
            try:
                with torch.no_grad():
                    _ = model(input_ids=base_tensors['input_ids'])
                supported_inputs.add('input_ids')
                print(f"[DYNAMIC] {model_name} supports: input_ids (only)")
            except Exception as e2:
                print(f"[DYNAMIC] {model_name} does NOT support even input_ids: {e2}")
        
        # Test token_type_ids as additional input (if has base inputs)
        if 'input_ids' in supported_inputs:
            try:
                with torch.no_grad():
                    test_inputs = {
                        'input_ids': base_tensors['input_ids'],
                        'token_type_ids': base_tensors['token_type_ids']
                    }
                    # Add attention_mask if supported
                    if 'attention_mask' in supported_inputs:
                        test_inputs['attention_mask'] = base_tensors['attention_mask']
                    
                    _ = model(**test_inputs)
                supported_inputs.add('token_type_ids')
                print(f"[DYNAMIC] {model_name} supports: token_type_ids")
            except Exception as e:
                print(f"[DYNAMIC] {model_name} does NOT support: token_type_ids - {e}")
        
        return supported_inputs
    
    def prepare_dataset(self, dataset_adapter) -> DataLoader:
        """
        Prepare dataset with automatically detected common inputs.
        Sets supported inputs in the adapter to filter tokenizer output.
        
        Args:
            dataset_adapter: Dataset adapter containing text data
            
        Returns:
            DataLoader configured for text data
            
        Raises:
            ValueError: If dataset is not in text mode
        """
        if dataset_adapter.mode != "text":
            raise ValueError("Dataset must be in text mode for TextClassificationTask")
        
        # Set supported inputs in adapter for filtering
        print(f"[TEXT_TASK] Setting supported inputs in adapter: {sorted(self.common_inputs)}")
        if hasattr(dataset_adapter, 'set_supported_inputs'):
            dataset_adapter.set_supported_inputs(self.common_inputs)
        else:
            print(f"[WARNING] Dataset adapter does not support set_supported_inputs")
        
        return dataset_adapter.get_text_loader()
    
    def forward_pass(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        """
        Execute forward pass with automatic input filtering and device management.
        
        Args:
            model: Neural network model (teacher or student)
            inputs: Input data (dictionary or tensor)
            
        Returns:
            Model logits as torch.Tensor
        """
        model_device = next(model.parameters()).device
        
        if isinstance(inputs, dict):
            # Move inputs to correct device
            device_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    device_inputs[k] = v.to(model_device)
                else:
                    device_inputs[k] = v
            
            # Filter unsupported inputs for safety
            filtered_inputs = self._safe_filter_inputs(device_inputs, model)
            
            # Forward pass for transformer models
            outputs = model(**filtered_inputs)
            
            # Extract logits from output
            if hasattr(outputs, 'logits'):
                return outputs.logits
            else:
                return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        else:
            # Direct input (less common for text)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(model_device)
            outputs = model(inputs)
            
            if hasattr(outputs, 'logits'):
                return outputs.logits
            else:
                return outputs
    
    def _safe_filter_inputs(self, inputs: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Advanced safety filter to prevent unsupported inputs from being passed to model.
        
        Args:
            inputs: Dictionary of input tensors
            model: Model to filter inputs for
            
        Returns:
            Filtered dictionary containing only supported inputs
        """
        # Determine which inputs this specific model supports
        if model is self._teacher_model:
            supported = self.teacher_supported_inputs
        elif model is self._student_model:
            supported = self.student_supported_inputs
        else:
            # Use common inputs as fallback
            supported = self.common_inputs
        
        filtered = {}
        for key, value in inputs.items():
            if key in supported:
                filtered[key] = value
        
        # Ensure at least essential inputs are present
        essential = ['input_ids', 'attention_mask']
        for key in essential:
            if key not in filtered and key in inputs:
                filtered[key] = inputs[key]
        
        return filtered
    
    def compute_distillation_loss(self, teacher_logits: torch.Tensor, 
                                   student_logits: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   config: Dict[str, Any]) -> torch.Tensor:
        """
        Calculate optimized distillation loss for text classification.
        Combines soft distillation loss (KL divergence) with hard cross-entropy loss.
        
        Args:
            teacher_logits: Teacher model output logits
            student_logits: Student model output logits
            labels: Ground truth labels
            config: Configuration with temperature and alpha parameters
            
        Returns:
            Combined distillation loss as torch.Tensor
        """
        import torch.nn.functional as F
        
        # Ensure all tensors are on the same device
        if teacher_logits.device != student_logits.device:
            teacher_logits = teacher_logits.to(student_logits.device)
        if labels.device != student_logits.device:
            labels = labels.to(student_logits.device)
        
        # Distillation parameters
        temperature = config.get('temperature', 3.0)
        alpha = config.get('alpha', 0.8)
        
        # Soft distillation loss (KL divergence between teacher and student)
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
        
        distillation_loss = F.kl_div(
            soft_predictions, 
            soft_targets, 
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Hard loss (standard cross entropy with ground truth)
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Weighted combination of soft and hard losses
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
        
        return total_loss
    
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model with safe input handling and device management.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary with accuracy, correct predictions count, and total samples
        """
        model.eval()
        correct = 0
        total = 0
        model_device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Prepare inputs by moving to correct device
                    inputs = {}
                    for k, v in batch.items():
                        if k != 'labels':
                            if isinstance(v, torch.Tensor):
                                inputs[k] = v.to(model_device)
                            else:
                                inputs[k] = v
                    
                    labels = batch['labels'].to(model_device)
                    
                    # Use forward_pass for consistency and safety
                    logits = self.forward_pass(model, inputs)
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Calculate accuracy
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                except Exception as e:
                    print(f"[WARNING] Text evaluation batch error: {e}")
                    continue
        
        # Return model to training mode
        model.train()
        
        if total > 0:
            return {
                'accuracy': correct / total,
                'correct': correct,
                'total': total
            }
        else:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}
    
    def get_teacher_model(self) -> torch.nn.Module:
        """
        Retrieve the teacher model.
        
        Returns:
            Teacher model instance
        """
        return self._teacher_model

    def get_student_model(self) -> torch.nn.Module:
        """
        Retrieve the student model.
        
        Returns:
            Student model instance
        """
        return self._student_model
    
    def get_tokenizer_info(self):
        """
        Get detailed information about tokenizer and models.
        
        Returns:
            Dictionary containing task info, model architecture details, and supported inputs
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
        
        # Try to infer model type from teacher
        if hasattr(self._teacher_model, 'config'):
            model_config = self._teacher_model.config
            if hasattr(model_config, 'model_type'):
                info['teacher_architecture'] = model_config.model_type
            if hasattr(model_config, 'vocab_size'):
                info['vocab_size'] = model_config.vocab_size
        
        # Student information if available
        if hasattr(self._student_model, 'config'):
            model_config = self._student_model.config
            if hasattr(model_config, 'model_type'):
                info['student_architecture'] = model_config.model_type
        
        return info
    
    def check_model_compatibility(self) -> bool:
        """
        Advanced compatibility check between teacher and student models.
        Tests with dummy inputs using only common supported inputs.
        
        Returns:
            True if models are compatible (same output shape), False otherwise
        """
        try:
            # Test with dummy input using only common inputs
            dummy_input = {}
            if 'input_ids' in self.common_inputs:
                dummy_input['input_ids'] = torch.randint(0, 1000, (2, 10))
            if 'attention_mask' in self.common_inputs:
                dummy_input['attention_mask'] = torch.ones(2, 10)
            if 'token_type_ids' in self.common_inputs:
                dummy_input['token_type_ids'] = torch.zeros(2, 10, dtype=torch.long)
            
            if not dummy_input:
                print(f"[TEXT_TASK] No common inputs found!")
                return False
            
            teacher_logits = self.forward_pass(self._teacher_model, dummy_input)
            student_logits = self.forward_pass(self._student_model, dummy_input)
            
            # Verify shape compatibility
            if teacher_logits.shape == student_logits.shape:
                print(f"[TEXT_TASK] Models compatible: {teacher_logits.shape}")
                print(f"[TEXT_TASK] Common inputs used: {sorted(dummy_input.keys())}")
                return True
            else:
                print(f"[TEXT_TASK] Shape mismatch: Teacher {teacher_logits.shape} vs Student {student_logits.shape}")
                return False
                
        except Exception as e:
            print(f"[TEXT_TASK] Compatibility error: {e}")
            return False
    
    def debug_compatibility(self):
        """
        Complete compatibility debug helper with detailed diagnostics.
        
        Returns:
            Dictionary containing compatibility test results and recommendations
        """
        print(f"\nDEBUG COMPATIBILITY")
        print(f"=" * 50)
        print(f"Teacher supports: {sorted(self.teacher_supported_inputs)}")
        print(f"Student supports: {sorted(self.student_supported_inputs)}")
        print(f"Common inputs: {sorted(self.common_inputs)}")
        print(f"Number of classes: {self.num_classes}")
        
        # Analyze specific inputs
        if 'token_type_ids' not in self.common_inputs:
            print("token_type_ids EXCLUDED from common (recommended for compatibility)")
        else:
            print("WARNING: token_type_ids INCLUDED in common (may cause errors)")
        
        # Test compatibility
        compatibility = self.check_model_compatibility()
        print(f"Compatibility test: {'PASS' if compatibility else 'FAIL'}")
        
        # Additional diagnostics
        essential_inputs = {'input_ids', 'attention_mask'}
        has_essentials = essential_inputs.issubset(self.common_inputs)
        print(f"Essential inputs present: {'YES' if has_essentials else 'NO'}")
        
        print(f"=" * 50)
        return {
            'compatibility': compatibility,
            'has_essentials': has_essentials,
            'common_inputs_count': len(self.common_inputs),
            'recommended': len(self.common_inputs) >= 2 and compatibility
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get useful information for training setup.
        
        Returns:
            Dictionary containing task configuration, compatibility info, and recommended settings
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