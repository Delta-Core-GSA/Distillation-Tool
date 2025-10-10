from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Type, Optional, Tuple
import os
from dataclasses import dataclass
from enum import Enum

from config.config import DatasetConfig,ModelConfig,TaskConfig



class IDatasetProvider(Protocol):
    """Interface per fornitori di dataset"""
    def create_adapter(self, config: DatasetConfig) -> Tuple[Any, Dict[str, Any]]:
        """Returns (adapter, dataset_info)"""
        ...

class IModelProvider(Protocol):
    """Interface per fornitori di modelli"""
    def load_model(self, config: ModelConfig) -> Any:
        """Returns loaded model"""
        ...

class ITaskProvider(Protocol):
    """Interface per fornitori di task handler"""
    def create_task_handler(self, task_config: TaskConfig, 
                          teacher_model: Any, student_model: Any) -> Any:
        """Returns task handler"""
        ...