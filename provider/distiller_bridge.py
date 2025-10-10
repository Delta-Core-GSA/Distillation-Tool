from typing import Dict, Any
from distillation_component_factory import DistillationComponentFactory
from distillation_config_builder import DistillationConfigBuilder
from providers import ModularDatasetProvider, StandardModelProvider
from task_type_provieder import TaskType,TaskRegistry
from config.config import TaskConfig,DatasetConfig,ModelConfig
#from custom_task_handler import CustomTaskHandler
import os



class DistillerBridge:
    """DistillerBridge migliorato con dependency injection"""
    
    def __init__(self, configuration: Dict[str, Any], 
                 factory: DistillationComponentFactory = None):
        
        self.config = configuration
        self.factory = factory or DistillationComponentFactory()
        
        # Initialize components usando factory
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize tutti i componenti usando factory"""
        
        print("[BRIDGE] Initializing components with factory...")
        
        # 1. Dataset
        self.dataset_adapter, self.dataset_info = self.factory.create_dataset_components(
            self.config['dataset']
        )
        
        # 2. Models
        self.teacher_model = self.factory.create_model(self.config['teacher'])
        self.student_model = self.factory.create_model(self.config['student'])
        
        # 3. Task handler
        task_type = self.factory.detect_task_type_from_info(self.dataset_info)
        
        # Aggiorna task config con info da dataset
        task_config = self.config['task']
        task_config.num_classes = self.dataset_info['num_classes']
        
        self.task_handler = self.factory.create_task_handler(
            task_type, task_config, self.teacher_model, self.student_model
        )
        
        print(f"[BRIDGE] ✅ All components initialized")
        print(f"  - Task: {task_type.value}")
        print(f"  - Classes: {self.dataset_info['num_classes']}")
        print(f"  - Samples: {self.dataset_info['num_samples']}")
    
    def distill(self):
        """Main distillation process"""
        # Il processo di distillazione rimane simile ma ora usa
        # i componenti creati dalla factory
        pass

# =================== USAGE EXAMPLES ===================

def example_modern_usage():
    """Esempio di utilizzo con la nuova architettura"""
    
    # 1. Builder pattern per configurazione
    config = (DistillationConfigBuilder()
              .with_dataset("path/to/dataset.csv", tokenizer_name="bert-base-uncased")
              .with_teacher_model("path/to/teacher.pt")
              .with_student_model("path/to/student.pt") 
              .with_task_config(num_classes=2, temperature=4.0, alpha=0.7)
              .build())
    
    # 2. Factory con dependency injection personalizzata
    custom_factory = DistillationComponentFactory(
        dataset_provider=ModularDatasetProvider(),
        model_provider=StandardModelProvider()
    )
    
    # 3. Bridge migliorato
    bridge = DistillerBridge(config, custom_factory)
    bridge.distill()
''''
def example_custom_task_registration():
    """Esempio di registrazione custom task"""
    
    class CustomTaskProvider:
        def create_task_handler(self, task_config, teacher, student):
            # Custom implementation
            return CustomTaskHandler(task_config, teacher, student)
    
    # Registra nuovo task type
    custom_task_type = TaskType.TEXT_GENERATION
    TaskRegistry.register_task_provider(custom_task_type, CustomTaskProvider)
    
    print(f"Available tasks: {[t.value for t in TaskRegistry.list_registered_tasks()]}")
'''
# =================== TESTING AND VALIDATION ===================

class ComponentValidator:
    """Validator per verificare la corretta configurazione dei componenti"""
    
    @staticmethod
    def validate_dataset_config(config: DatasetConfig) -> list[str]:
        """Valida configurazione dataset"""
        errors = []
        
        if not os.path.exists(config.csv_path):
            errors.append(f"Dataset file not found: {config.csv_path}")
        
        if config.max_samples and config.max_samples <= 0:
            errors.append("max_samples must be positive")
        
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        return errors
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> list[str]:
        """Valida configurazione modello"""
        errors = []
        
        if not os.path.exists(config.model_path):
            errors.append(f"Model file not found: {config.model_path}")
        
        if config.device and config.device not in ['cpu', 'cuda']:
            if not config.device.startswith('cuda:'):
                errors.append(f"Invalid device: {config.device}")
        
        return errors
    
    @staticmethod
    def validate_task_config(config: TaskConfig) -> list[str]:
        """Valida configurazione task"""
        errors = []
        
        if config.num_classes <= 0:
            errors.append("num_classes must be positive")
        
        if not 0 < config.alpha < 1:
            errors.append("alpha must be between 0 and 1")
        
        if config.temperature <= 0:
            errors.append("temperature must be positive")
        
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        return errors
    
    @classmethod
    def validate_full_config(cls, config: Dict[str, Any]) -> Dict[str, list[str]]:
        """Valida configurazione completa"""
        validation_results = {}
        
        if 'dataset' in config:
            validation_results['dataset'] = cls.validate_dataset_config(config['dataset'])
        
        if 'teacher' in config:
            validation_results['teacher'] = cls.validate_model_config(config['teacher'])
        
        if 'student' in config:
            validation_results['student'] = cls.validate_model_config(config['student'])
        
        if 'task' in config:
            validation_results['task'] = cls.validate_task_config(config['task'])
        
        return {k: v for k, v in validation_results.items() if v}  # Solo errori