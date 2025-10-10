from config.config import DatasetConfig,ModelConfig,TaskConfig
from typing import Dict, Any



class DistillationConfigBuilder:
    """Builder per configurazioni complesse"""
    
    def __init__(self):
        self.dataset_config = None
        self.teacher_config = None
        self.student_config = None
        self.task_config = None
    
    def with_dataset(self, csv_path: str, **kwargs) -> 'DistillationConfigBuilder':
        self.dataset_config = DatasetConfig(csv_path=csv_path, **kwargs)
        return self
    
    def with_teacher_model(self, model_path: str, **kwargs) -> 'DistillationConfigBuilder':
        self.teacher_config = ModelConfig(model_path=model_path, **kwargs)
        return self
    
    def with_student_model(self, model_path: str, **kwargs) -> 'DistillationConfigBuilder':
        self.student_config = ModelConfig(model_path=model_path, **kwargs)
        return self
    
    def with_task_config(self, num_classes: int, **kwargs) -> 'DistillationConfigBuilder':
        self.task_config = TaskConfig(num_classes=num_classes, **kwargs)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Constructs final configuration"""
        if not all([self.dataset_config, self.teacher_config, 
                   self.student_config, self.task_config]):
            raise ValueError("All configurations must be set")
        
        return {
            'dataset': self.dataset_config,
            'teacher': self.teacher_config,
            'student': self.student_config,
            'task': self.task_config
        }