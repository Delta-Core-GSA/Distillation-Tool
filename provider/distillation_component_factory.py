from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Type, Optional, Tuple
import os
from dataclasses import dataclass
from enum import Enum
from provider.interfaces import IDatasetProvider,IModelProvider
from provider.providers import ModularDatasetProvider,StandardModelProvider


from dataset_adapter.modular_dataset_adapter import ModularDatasetAdapter  # Import del modulo refactorato
from config.config import DatasetConfig,ModelConfig
from provider.task_type_provieder import TaskRegistry,TaskType
from provider.task_provider import TextClassificationProvider,ImageClassificationProvider
from config.config import TaskConfig




class DistillationComponentFactory:
    """Factory principale con dependency injection"""
    
    def __init__(self, 
                 dataset_provider: IDatasetProvider = None,
                 model_provider: IModelProvider = None):
        
        # Dependency injection con default providers
        self.dataset_provider = dataset_provider or ModularDatasetProvider()
        self.model_provider = model_provider or StandardModelProvider()
        
        # Registra i task providers di default
        self._register_default_tasks()
    
    def _register_default_tasks(self):
        """Registra i task providers di default"""
        TaskRegistry.register_task_provider(
            TaskType.TEXT_CLASSIFICATION, 
            TextClassificationProvider
        )
        TaskRegistry.register_task_provider(
            TaskType.IMAGE_CLASSIFICATION, 
            ImageClassificationProvider
        )
    
    def create_dataset_components(self, config: DatasetConfig) -> Tuple[Any, Dict[str, Any]]:
        """Crea adapter e info dataset"""
        return self.dataset_provider.create_adapter(config)
    
    def create_model(self, config: ModelConfig) -> Any:
        """Crea/carica modello"""
        return self.model_provider.load_model(config)
    
    def create_task_handler(self, task_type: TaskType, task_config: TaskConfig,
                          teacher_model: Any, student_model: Any) -> Any:
        """Crea task handler appropriato"""
        
        provider_class = TaskRegistry.get_task_provider(task_type)
        provider = provider_class()
        
        return provider.create_task_handler(task_config, teacher_model, student_model)
    
    def detect_task_type_from_info(self, dataset_info: Dict[str, Any]) -> TaskType:
        """Detect task type from dataset info"""
        task_type_str = dataset_info.get('task_type', 'unknown')
        
        task_mapping = {
            'text': TaskType.TEXT_CLASSIFICATION,
            'image': TaskType.IMAGE_CLASSIFICATION,
            'tabular': TaskType.TABULAR_CLASSIFICATION
        }
        
        if task_type_str in task_mapping:
            return task_mapping[task_type_str]
        else:
            raise ValueError(f"Unknown task type: {task_type_str}")