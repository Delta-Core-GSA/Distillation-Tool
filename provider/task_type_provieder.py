from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Type, Optional, Tuple, Union
import os
from dataclasses import dataclass
from enum import Enum
from provider.interfaces import ITaskProvider

class TaskType(Enum):
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    TABULAR_CLASSIFICATION = "tabular_classification"
    TEXT_GENERATION = "text_generation"

class TaskRegistry:
    """Registry per mappare task types a provider classes - FIXED VERSION"""
    
    # 🔧 FIX: Ora accetta sia Enum che stringhe
    _providers: Dict[Union[TaskType, str], Type[ITaskProvider]] = {}
    
    @classmethod
    def register_task_provider(cls, task_type: Union[TaskType, str], provider_class: Type[ITaskProvider]):
        """Registra un provider per un task type - ACCEPTS BOTH ENUM AND STRING"""
        cls._providers[task_type] = provider_class
        
        # 🔧 FIX: Gestisce sia Enum che stringhe
        if hasattr(task_type, 'value'):
            # È un Enum
            task_name = task_type.value
        else:
            # È una stringa
            task_name = str(task_type)
        
        print(f"[REGISTRY] Registered {provider_class.__name__} for '{task_name}'")
    
    @classmethod
    def get_task_provider(cls, task_type: Union[TaskType, str]) -> Type[ITaskProvider]:
        """Ottiene il provider per un task type - ACCEPTS BOTH ENUM AND STRING"""
        
        # Cerca direttamente
        if task_type in cls._providers:
            return cls._providers[task_type]
        
        # Se non trovato, prova compatibilità incrociata
        if isinstance(task_type, str):
            # task_type è stringa, cerca per Enum con stesso valore
            for registered_key, provider in cls._providers.items():
                if hasattr(registered_key, 'value') and registered_key.value == task_type:
                    return provider
        elif isinstance(task_type, TaskType):
            # task_type è Enum, cerca per stringa con stesso valore
            if task_type.value in cls._providers:
                return cls._providers[task_type.value]
        
        # Non trovato
        task_name = task_type.value if hasattr(task_type, 'value') else str(task_type)
        raise ValueError(f"No provider registered for task type: '{task_name}'")
    
    @classmethod
    def list_registered_tasks(cls) -> list[Union[TaskType, str]]:
        """Lista dei task types registrati"""
        return list(cls._providers.keys())