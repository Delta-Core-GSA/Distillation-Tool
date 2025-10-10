from typing import Dict, Any
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch


class ImageClassificationTask(BaseTask):
    """
    Task implementation for image classification with knowledge distillation.
    Handles forward passes, loss computation, and evaluation for image models.
    """
    
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        """
        Initialize image classification task with models.
        
        Args:
            config: Task configuration including num_classes, temperature, alpha
            teacher_model: Pre-trained teacher model for distillation
            student_model: Student model to be trained
        """
        super().__init__(config)
        self.task_type = TaskType.IMAGE_CLASSIFICATION
        self.num_classes = config.get('num_classes', 1000)
        
        self._teacher_model = teacher_model
        self._student_model = student_model
        
        print(f"[IMAGE_TASK] Initialized with {self.num_classes} classes")
    
    def prepare_dataset(self, dataset_adapter):
        """
        Prepare DataLoader for image classification.
        
        Args:
            dataset_adapter: Dataset adapter containing image data
            
        Returns:
            DataLoader configured for images
            
        Raises:
            ValueError: If dataset is not in image mode
        """
        if dataset_adapter.mode != "image":
            raise ValueError("Dataset must be in image mode for ImageClassificationTask")
        return dataset_adapter.get_image_loader()
    
    def forward_pass(self, model, inputs):
        """
        Execute forward pass with automatic device management.
        Ensures model and inputs are on the same device.
        
        Args:
            model: Neural network model (teacher or student)
            inputs: Input data (tensor or dictionary of tensors)
            
        Returns:
            Model logits as torch.Tensor
        """
        # Get model's device
        model_device = next(model.parameters()).device
        
        if isinstance(inputs, dict):
            # Move all tensor inputs to model's device
            device_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    device_inputs[k] = v.to(model_device)
                else:
                    device_inputs[k] = v
            outputs = model(**device_inputs)
        else:
            # Direct tensor input
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(model_device)
            outputs = model(inputs)
        
        # Extract logits safely
        if hasattr(outputs, 'logits'):
            return outputs.logits
        else:
            return outputs
    
    def compute_distillation_loss(self, teacher_logits, student_logits, labels, config):
        """
        Calculate distillation loss combining soft and hard losses.
        
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
        
        temperature = config.get('temperature', 4.0)
        alpha = config.get('alpha', 0.7)
        
        # Soft loss: KL divergence between teacher and student distributions
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean"
        ) * (temperature ** 2)
        
        # Hard loss: standard cross-entropy with ground truth
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Weighted combination of soft and hard losses
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
    def evaluate(self, model, dataloader):
        """
        Evaluate model accuracy with device management.
        
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
                    if isinstance(batch, (list, tuple)):
                        inputs, labels = batch
                        # Move to model's device
                        if isinstance(inputs, torch.Tensor):
                            inputs = inputs.to(model_device)
                        if isinstance(labels, torch.Tensor):
                            labels = labels.to(model_device)
                    else:
                        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch.items() if k != 'labels'}
                        labels = batch['labels'].to(model_device)
                    
                    logits = self.forward_pass(model, inputs)
                    predictions = torch.argmax(logits, dim=1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                except Exception as e:
                    print(f"[WARNING] Evaluation batch error: {e}")
                    continue
        
        model.train()
        
        if total > 0:
            return {
                'accuracy': correct / total,
                'correct': correct,
                'total': total
            }
        else:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}