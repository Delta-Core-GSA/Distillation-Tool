from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

class BaseTask(ABC):
    """Interfaccia base per tutte le task di distillazione"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = None
        self.num_classes = config.get('num_classes', 2)  # Default generico
        
    @abstractmethod
    def prepare_dataset(self, dataset_adapter) -> DataLoader:
        """Prepara il dataset specifico per questa task"""
        pass
    
    @abstractmethod
    def forward_pass(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        """
        NUOVO: Esegue forward pass per il modello specifico della task
        Gestisce le differenze tra architetture diverse
        
        Args:
            model: Il modello (teacher o student)
            inputs: Input formattati per il modello
            
        Returns:
            torch.Tensor: Logits del modello
        """
        pass
    
    @abstractmethod
    def compute_distillation_loss(self, teacher_logits: torch.Tensor, 
                                student_logits: torch.Tensor, 
                                labels: torch.Tensor, 
                                config: Dict[str, Any]) -> torch.Tensor:
        """
        AGGIORNATO: Calcola la loss di distillazione specifica per questa task
        
        Args:
            teacher_logits: Output del teacher model
            student_logits: Output del student model  
            labels: Ground truth labels
            config: Configurazione con temperature, alpha, etc.
            
        Returns:
            torch.Tensor: Loss di distillazione
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Valuta il modello su questa task"""
        pass
    
    # =================== METODI HELPER OPZIONALI ===================
    
    def get_teacher_model(self) -> torch.nn.Module:
        """
        OPZIONALE: Ritorna il modello teacher se stored nella task
        Override se necessario
        """
        if hasattr(self, '_teacher_model'):
            return self._teacher_model
        else:
            raise NotImplementedError("Teacher model not stored in task or method not implemented")
    
    def get_student_model(self) -> torch.nn.Module:
        """
        OPZIONALE: Ritorna il modello student se stored nella task
        Override se necessario
        """
        if hasattr(self, '_student_model'):
            return self._student_model
        else:
            raise NotImplementedError("Student model not stored in task or method not implemented")
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        HELPER: Ritorna informazioni sulla task
        """
        return {
            'task_type': self.task_type.value if self.task_type else 'unknown',
            'num_classes': self.num_classes,
            'config': self.config
        }