from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any


class BaseTask(ABC):
    """
    Base interface for all distillation tasks.
    Defines the contract that all task implementations must follow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base task with configuration.
        
        Args:
            config: Task configuration dictionary containing parameters like num_classes
        """
        self.config = config
        self.task_type = None
        self.num_classes = config.get('num_classes', 2)
        
    @abstractmethod
    def prepare_dataset(self, dataset_adapter) -> DataLoader:
        """
        Prepare task-specific dataset and return DataLoader.
        
        Args:
            dataset_adapter: Dataset adapter containing the data
            
        Returns:
            DataLoader configured for this task
        """
        pass
    
    @abstractmethod
    def forward_pass(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        """
        Execute forward pass for the specific model.
        Handles differences between various architectures.
        
        Args:
            model: The model (teacher or student)
            inputs: Model-formatted inputs
            
        Returns:
            Model logits as torch.Tensor
        """
        pass
    
    @abstractmethod
    def compute_distillation_loss(self, teacher_logits: torch.Tensor, 
                                   student_logits: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   config: Dict[str, Any]) -> torch.Tensor:
        """
        Calculate task-specific distillation loss.
        
        Args:
            teacher_logits: Teacher model outputs
            student_logits: Student model outputs
            labels: Ground truth labels
            config: Configuration with temperature, alpha, and other parameters
            
        Returns:
            Computed distillation loss as torch.Tensor
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on this task.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary mapping metric names to values
        """
        pass
    
    def get_teacher_model(self) -> torch.nn.Module:
        """
        Retrieve the teacher model if stored in the task instance.
        Optional method that can be overridden in subclasses.
        
        Returns:
            Teacher model instance
            
        Raises:
            NotImplementedError: If teacher model is not stored or method not implemented
        """
        if hasattr(self, '_teacher_model'):
            return self._teacher_model
        else:
            raise NotImplementedError("Teacher model not stored in task or method not implemented")
    
    def get_student_model(self) -> torch.nn.Module:
        """
        Retrieve the student model if stored in the task instance.
        Optional method that can be overridden in subclasses.
        
        Returns:
            Student model instance
            
        Raises:
            NotImplementedError: If student model is not stored or method not implemented
        """
        if hasattr(self, '_student_model'):
            return self._student_model
        else:
            raise NotImplementedError("Student model not stored in task or method not implemented")
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task information and configuration.
        
        Returns:
            Dictionary containing task type, number of classes, and configuration
        """
        return {
            'task_type': self.task_type.value if self.task_type else 'unknown',
            'num_classes': self.num_classes,
            'config': self.config
        }