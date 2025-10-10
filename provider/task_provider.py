
from abc import ABC, abstractmethod
from typing import Dict, Any


from dataset_adapter.modular_dataset_adapter import ModularDatasetAdapter  # Import del modulo refactorato
from config.config import TaskConfig
from dataset_by_tasking.text_classification import TextClassificationTask
from dataset_by_tasking.image_classification import ImageClassificationTask




class TextClassificationProvider:
    """Provider per text classification tasks"""
    
    def create_task_handler(self, task_config: TaskConfig, 
                          teacher_model: Any, student_model: Any) -> Any:
        
        
        config_dict = {
            'num_classes': task_config.num_classes,
            'temperature': task_config.temperature,
            'alpha': task_config.alpha,
            'learning_rate': task_config.learning_rate,
            'epochs': task_config.epochs
        }
        
        return TextClassificationTask(config_dict, teacher_model, student_model)

class ImageClassificationProvider:
    """Provider per image classification tasks"""
    
    def create_task_handler(self, task_config: TaskConfig, 
                          teacher_model: Any, student_model: Any) -> Any:
        
        
        config_dict = {
            'num_classes': task_config.num_classes,
            'temperature': task_config.temperature,
            'alpha': task_config.alpha,
            'learning_rate': task_config.learning_rate,
            'epochs': task_config.epochs
        }
        
        return ImageClassificationTask(config_dict, teacher_model, student_model)
