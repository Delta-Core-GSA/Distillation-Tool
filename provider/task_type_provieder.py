from typing import Dict, Type, Union
from enum import Enum
from provider.interfaces import ITaskProvider


class TaskType(Enum):
    """
    Enumeration of supported task types for machine learning distillation.
    """
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    TABULAR_CLASSIFICATION = "tabular_classification"
    TEXT_GENERATION = "text_generation"


class TaskRegistry:
    """
    Registry for mapping task types to provider classes.
    Supports both TaskType enums and string representations for flexibility.
    """
    
    _providers: Dict[Union[TaskType, str], Type[ITaskProvider]] = {}
    
    @classmethod
    def register_task_provider(cls, task_type: Union[TaskType, str], provider_class: Type[ITaskProvider]):
        """
        Register a provider class for a specific task type.
        Accepts both TaskType enum and string representations.
        
        Args:
            task_type: Task type as TaskType enum or string
            provider_class: Provider class implementing ITaskProvider interface
        """
        cls._providers[task_type] = provider_class
        
        # Extract task name for logging
        if hasattr(task_type, 'value'):
            task_name = task_type.value
        else:
            task_name = str(task_type)
        
        print(f"[REGISTRY] Registered {provider_class.__name__} for '{task_name}'")
    
    @classmethod
    def get_task_provider(cls, task_type: Union[TaskType, str]) -> Type[ITaskProvider]:
        """
        Retrieve the provider class for a specific task type.
        Supports both TaskType enum and string with cross-compatibility.
        
        Args:
            task_type: Task type as TaskType enum or string
            
        Returns:
            Provider class registered for the task type
            
        Raises:
            ValueError: If no provider is registered for the given task type
        """
        # Direct lookup
        if task_type in cls._providers:
            return cls._providers[task_type]
        
        # Cross-compatibility lookup if not found directly
        if isinstance(task_type, str):
            # task_type is string, search for Enum with same value
            for registered_key, provider in cls._providers.items():
                if hasattr(registered_key, 'value') and registered_key.value == task_type:
                    return provider
        elif isinstance(task_type, TaskType):
            # task_type is Enum, search for string with same value
            if task_type.value in cls._providers:
                return cls._providers[task_type.value]
        
        # Not found
        task_name = task_type.value if hasattr(task_type, 'value') else str(task_type)
        raise ValueError(f"No provider registered for task type: '{task_name}'")
    
    @classmethod
    def list_registered_tasks(cls) -> list[Union[TaskType, str]]:
        """
        Get list of all registered task types.
        
        Returns:
            List of registered task types (may contain both enums and strings)
        """
        return list(cls._providers.keys())