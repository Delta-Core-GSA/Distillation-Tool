from typing import Dict, Any
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch

class ImageClassificationTask(BaseTask):
    def __init__(self, config: Dict[str, Any], teacher_model: torch.nn.Module, student_model: torch.nn.Module):
        super().__init__(config)
        self.task_type = TaskType.IMAGE_CLASSIFICATION
        self.num_classes = config.get('num_classes', 1000)
        
        self._teacher_model = teacher_model
        self._student_model = student_model
        
        print(f"[IMAGE_TASK] ✅ Inizializzato con {self.num_classes} classi")
    
    def prepare_dataset(self, dataset_adapter):
        """Prepara il dataloader per image classification"""
        if dataset_adapter.mode != "image":
            raise ValueError("Dataset must be in image mode for ImageClassificationTask")
        return dataset_adapter.get_image_loader()
    
    def forward_pass(self, model, inputs):
        """
        AGGIORNATO: Forward pass con device safety
        """
        # Assicurati che model e inputs siano sullo stesso device
        model_device = next(model.parameters()).device
        
        if isinstance(inputs, dict):
            # Sposta tutti i tensor inputs sul device del modello
            device_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    device_inputs[k] = v.to(model_device)
                else:
                    device_inputs[k] = v
            outputs = model(**device_inputs)
        else:
            # Input tensor diretto
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(model_device)
            outputs = model(inputs)
        
        # Estrai logits in modo sicuro
        if hasattr(outputs, 'logits'):
            return outputs.logits
        else:
            return outputs
    
    def compute_distillation_loss(self, teacher_logits, student_logits, labels, config):
        """Loss di distillazione per image classification"""
        import torch.nn.functional as F
        
        # Assicurati che tutti i tensori siano sullo stesso device
        if teacher_logits.device != student_logits.device:
            teacher_logits = teacher_logits.to(student_logits.device)
        if labels.device != student_logits.device:
            labels = labels.to(student_logits.device)
        
        temperature = config.get('temperature', 4.0)
        alpha = config.get('alpha', 0.7)
        
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean"
        ) * (temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
    def evaluate(self, model, dataloader):
        """
        AGGIORNATO: Evaluation con device management
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
                        # Sposta su device del modello
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
                    print(f"[WARNING] Errore evaluation batch: {e}")
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