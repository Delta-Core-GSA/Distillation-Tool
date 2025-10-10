from dataset_by_tasking.task_type import TaskType
from dataset_by_tasking.image_classification import ImageClassificationTask
from dataset_by_tasking.tabular_classification import TabularClassificationTask
from dataset_by_tasking.text_classification import TextClassificationTask
from dataset_by_tasking.text_generation import TextGenerationTask
from dataset_by_tasking.base_task import BaseTask
from typing import Dict,Any



class TaskFactory:
    """Factory per creare task basate sul tipo"""
    
    _task_registry = {
        TaskType.IMAGE_CLASSIFICATION: ImageClassificationTask,
        TaskType.TEXT_CLASSIFICATION: TextClassificationTask,
        TaskType.TEXT_GENERATION: TextGenerationTask,
        TaskType.TABULAR_CLASSIFICATION: TabularClassificationTask,
    }
    
    @classmethod
    def create_task(cls, task_type: TaskType, config: Dict[str, Any]) -> BaseTask:
        """Crea una task del tipo specificato"""
        if task_type not in cls._task_registry:
            raise ValueError(f"Task type {task_type} not supported")
        
        task_class = cls._task_registry[task_type]
        return task_class(config)
    
    @classmethod
    def register_task(cls, task_type: TaskType, task_class):
        """Registra una nuova task personalizzata"""
        cls._task_registry[task_type] = task_class
    
    @classmethod
    def get_supported_tasks(cls):
        """Ritorna lista di task supportate"""
        return list(cls._task_registry.keys())