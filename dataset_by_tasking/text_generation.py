from typing import Dict, Any
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch


class TextGenerationTask(BaseTask):
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        super().__init__(config)
        self.task_type = TaskType.TEXT_GENERATION
        self._teacher_model = teacher_model
        self._student_model = student_model
        
    def prepare_dataset(self, dataset_adapter):
        """Prepara dataset per text generation"""
        if dataset_adapter.mode != "text":
            raise ValueError("Dataset must be in text mode for TextGenerationTask")
        return dataset_adapter.get_generation_loader()
    
    def get_teacher_model(self) -> torch.nn.Module:
        """Ritorna il modello teacher passato al costruttore"""
        return self._teacher_model
    
    def get_student_model(self) -> torch.nn.Module:
        """Ritorna il modello student passato al costruttore"""
        return self._student_model
    
    def compute_distillation_loss(self, teacher_outputs, student_outputs, targets):
        import torch.nn.functional as F
        
        temperature = self.config.get('temperature', 2.0)
        alpha = self.config.get('alpha', 0.5)
        
        # Per generazione, usiamo la loss sui logits
        teacher_logits = teacher_outputs.logits
        student_logits = student_outputs.logits
        
        # Reshape per calcolare la loss
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        targets_flat = targets.view(-1)
        
        # KL Divergence
        soft_targets = F.softmax(teacher_logits_flat / temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits_flat / temperature, dim=1)
        
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
        student_loss = F.cross_entropy(student_logits_flat, targets_flat, ignore_index=-100)
        
        return alpha * distillation_loss + (1 - alpha) * student_loss
    
    def evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(input_ids=batch['input_ids'], 
                              attention_mask=batch['attention_mask'],
                              labels=batch['target_ids'])
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return {'perplexity': torch.exp(torch.tensor(total_loss / num_batches)).item()}