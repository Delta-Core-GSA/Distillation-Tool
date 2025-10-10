from typing import Dict, Any
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch


class TabularClassificationTask(BaseTask):
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        super().__init__(config)
        self.task_type = TaskType.TABULAR_CLASSIFICATION
        self.num_classes = config.get('num_classes', 2)
        self.input_dim = config.get('input_dim', 10)
        self._teacher_model = teacher_model
        self._student_model = student_model
        
    def prepare_dataset(self, dataset_adapter):
        """Prepara dataset per classificazione tabular"""
        if dataset_adapter.mode != "tabular":
            raise ValueError("Dataset must be in tabular mode for TabularClassificationTask")
        return dataset_adapter.get_tabular_loader()
    
    def get_teacher_model(self) -> torch.nn.Module:
        """Ritorna il modello teacher passato al costruttore"""
        return self._teacher_model
    
    def get_student_model(self) -> torch.nn.Module:
        """Ritorna il modello student passato al costruttore"""
        return self._student_model
    
    def compute_distillation_loss(self, teacher_outputs, student_outputs, targets):
        import torch.nn.functional as F
        
        temperature = self.config.get('temperature', 3.0)
        alpha = self.config.get('alpha', 0.7)
        
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        soft_predictions = F.log_softmax(student_outputs / temperature, dim=1)
        
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
        student_loss = F.cross_entropy(student_outputs, targets)
        
        return alpha * distillation_loss + (1 - alpha) * student_loss
    
    def evaluate(self, model, dataloader):
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