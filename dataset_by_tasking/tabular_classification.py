from typing import Dict, Any
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch


class TabularClassificationTask(BaseTask):
    """
    Task implementation for tabular data classification with knowledge distillation.
    Handles structured/tabular data with numerical features.
    """
    
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        """
        Initialize tabular classification task with models.
        
        Args:
            config: Task configuration including num_classes, temperature, alpha
            teacher_model: Pre-trained teacher model for distillation
            student_model: Student model to be trained
        """
        super().__init__(config)
        self.task_type = TaskType.TABULAR_CLASSIFICATION
        self.num_classes = config.get('num_classes', 2)
        self._teacher_model = teacher_model
        self._student_model = student_model
        
    def prepare_dataset(self, dataset_adapter):
        """
        Prepare DataLoader for tabular classification.
        
        Args:
            dataset_adapter: Dataset adapter containing tabular data
            
        Returns:
            DataLoader configured for tabular data
            
        Raises:
            ValueError: If dataset is not in tabular mode
        """
        if dataset_adapter.mode != "tabular":
            raise ValueError("Dataset must be in tabular mode for TabularClassificationTask")
        return dataset_adapter.get_tabular_loader()
    
    def forward_pass(self, model, inputs):
        """
        Execute forward pass for tabular model.
        
        Args:
            model: Neural network model (teacher or student)
            inputs: Tabular input features as torch.Tensor
            
        Returns:
            Model outputs as torch.Tensor
        """
        return model(inputs)
    
    def get_teacher_model(self) -> torch.nn.Module:
        """
        Retrieve the teacher model.
        
        Returns:
            Teacher model instance passed during initialization
        """
        return self._teacher_model
    
    def get_student_model(self) -> torch.nn.Module:
        """
        Retrieve the student model.
        
        Returns:
            Student model instance passed during initialization
        """
        return self._student_model
    
    def compute_distillation_loss(self, teacher_outputs, student_outputs, targets, config=None):
        """
        Calculate distillation loss combining soft and hard losses for tabular data.
        
        Args:
            teacher_outputs: Teacher model output logits
            student_outputs: Student model output logits
            targets: Ground truth labels
            config: Optional configuration override (uses self.config if None)
            
        Returns:
            Combined distillation loss as torch.Tensor
        """
        import torch.nn.functional as F
        
        # Use provided config or fall back to instance config
        cfg = config if config is not None else self.config
        temperature = cfg.get('temperature', 3.0)
        alpha = cfg.get('alpha', 0.7)
        
        # Soft targets from teacher predictions
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        soft_predictions = F.log_softmax(student_outputs / temperature, dim=1)
        
        # Soft loss: KL divergence between teacher and student distributions
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Hard loss: standard cross-entropy with ground truth
        student_loss = F.cross_entropy(student_outputs, targets)
        
        # Weighted combination of distillation and student losses
        return alpha * distillation_loss + (1 - alpha) * student_loss
    
    def evaluate(self, model, dataloader):
        """
        Evaluate model accuracy on tabular data.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary containing accuracy metric
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return {'accuracy': correct / total}