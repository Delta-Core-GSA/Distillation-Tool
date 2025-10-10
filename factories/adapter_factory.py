import os
from adapters.dataset_adapter import BaseDatasetAdapter
from adapters.model_adapter import BaseModelAdapter
from dataset_by_tasking.task_type_detector import TaskDetector
from dataset_by_tasking.task_type import TaskType


class AdapterFactory:
    """
    Factory class for creating appropriate adapters for datasets, tasks, and models.
    Provides centralized adapter creation with automatic configuration.
    """
    
    @staticmethod
    def create_dataset_adapter(dataset_path, tokenizer_name=None, max_samples=None, 
                                imagenet_mapping_path=None):
        """
        Create dataset adapter and return it along with dataset information.
        
        Args:
            dataset_path: Path to CSV dataset file
            tokenizer_name: Name of HuggingFace tokenizer to use (optional)
            max_samples: Maximum number of samples to load (optional)
            imagenet_mapping_path: Path to ImageNet label mapping JSON (optional)
            
        Returns:
            Tuple of (BaseDatasetAdapter, dict): Dataset adapter and info dictionary containing
                task_type, num_classes, class_names, label_mapping, and num_samples
                
        Raises:
            FileNotFoundError: If dataset file does not exist
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        print(f"[FACTORY] Creating dataset adapter for: {dataset_path}")
        
        # Create the adapter
        adapter = BaseDatasetAdapter(
            csv_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            imagenet_mapping_path=imagenet_mapping_path
        )
        
        # Extract dataset information
        dataset_info = adapter.get_dataset_info()
        
        print(f"[FACTORY] Dataset adapter created - Task: {dataset_info['task_type']}")
        print(f"[FACTORY] Dataset shape: {adapter.df.shape}")
        print(f"[FACTORY] Number of classes detected: {dataset_info['num_classes']}")
        
        return adapter, dataset_info
    
    @staticmethod
    def create_task_adapter(dataset_path, config, teacher_model, student_model, dataset_info=None):
        """
        Create task-specific adapter based on detected or provided dataset information.
        
        Args:
            dataset_path: Path to CSV dataset file
            config: Configuration dictionary for task initialization
            teacher_model: Teacher model for distillation
            student_model: Student model to be trained
            dataset_info: Optional dataset information dict (if None, will auto-detect)
            
        Returns:
            Task adapter instance (TextClassificationTask, ImageClassificationTask, etc.)
            
        Raises:
            FileNotFoundError: If dataset file does not exist
            ValueError: If task type is not supported
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Use provided dataset info if available, otherwise auto-detect
        if dataset_info:
            task_type_str = dataset_info['task_type']
            num_classes = dataset_info['num_classes']
            print(f"[FACTORY] Using dataset info: {task_type_str}, {num_classes} classes")
            
            # Convert string to enum if necessary
            if isinstance(task_type_str, str):
                task_type = TaskType(task_type_str)
            else:
                task_type = task_type_str
        else:
            # Fallback to automatic detection
            import pandas as pd
            df = pd.read_csv(dataset_path)
            task_info = TaskDetector.detect_task_type(df)
            task_type = task_info['task_type']
            task_type_str = task_type.value
            num_classes = task_info.get('num_classes', 2)
        
        # Add num_classes to config
        config['num_classes'] = num_classes
        
        print(f"[FACTORY] Creating task adapter for: {task_type_str}")
        
        # Create appropriate task adapter based on type
        if task_type == TaskType.TEXT_CLASSIFICATION:
            from dataset_by_tasking.text_classification import TextClassificationTask
            return TextClassificationTask(config, teacher_model, student_model)
            
        elif task_type == TaskType.IMAGE_CLASSIFICATION:
            from dataset_by_tasking.image_classification import ImageClassificationTask
            return ImageClassificationTask(config, teacher_model, student_model)
            
        elif task_type == TaskType.TEXT_GENERATION:
            from dataset_by_tasking.text_generation import TextGenerationTask
            return TextGenerationTask(config, teacher_model, student_model)
            
        elif task_type == TaskType.TABULAR_CLASSIFICATION:
            from dataset_by_tasking.tabular_classification import TabularClassificationTask
            return TabularClassificationTask(config, teacher_model, student_model)
            
        else:
            raise ValueError(f"Unsupported task type: {task_type.value}")
  
    @staticmethod
    def create_model_adapter(model_path):
        """
        Create adapter for a saved model.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            BaseModelAdapter: Adapter instance for the model
            
        Raises:
            FileNotFoundError: If model file does not exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"[FACTORY] Creating model adapter for: {model_path}")
        
        adapter = BaseModelAdapter(model_path)
        
        print(f"[FACTORY] Model adapter created - Type: {type(adapter.model)}")
        return adapter

    @staticmethod
    def create_dataset_adapter_with_imagenet_mapping(dataset_path, imagenet_mapping_path, 
                                                      tokenizer_name=None, max_samples=None):
        """
        Convenience method to create dataset adapter with ImageNet mapping.
        
        Args:
            dataset_path: Path to CSV dataset file
            imagenet_mapping_path: Path to ImageNet label mapping JSON file
            tokenizer_name: Name of HuggingFace tokenizer to use (optional)
            max_samples: Maximum number of samples to load (optional)
            
        Returns:
            Tuple of (BaseDatasetAdapter, dict): Dataset adapter configured for ImageNet
                and its information dictionary
        """
        return AdapterFactory.create_dataset_adapter(
            dataset_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            imagenet_mapping_path=imagenet_mapping_path
        )