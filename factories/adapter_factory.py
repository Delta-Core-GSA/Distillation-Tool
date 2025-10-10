import os
from adapters.dataset_adapter import BaseDatasetAdapter
from adapters.model_adapter import BaseModelAdapter
from dataset_by_tasking.task_type_detector import TaskDetector
from dataset_by_tasking.task_type import TaskType


class AdapterFactory:
    """Factory per creare gli adapter appropriati"""
    
    @staticmethod
    def create_dataset_adapter(dataset_path, tokenizer_name=None, max_samples=None, 
                             imagenet_mapping_path=None, task_type=None):
        """
        MODIFICATO: Ora ritorna anche le informazioni del dataset
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")
        
        print(f"[FACTORY] Creando dataset adapter per: {dataset_path}")
        
        # Crea l'adapter (codice originale)
        adapter = BaseDatasetAdapter(
            csv_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            #task_type=task_type,
        )
        
        # NUOVO: Estrai le informazioni del dataset
        dataset_info = adapter.get_dataset_info()
        
        print(f"[FACTORY] Dataset adapter creato - Task: {dataset_info['task_type']}")
        print(f"[FACTORY] Dataset shape: {adapter.df.shape}")
        print(f"[FACTORY] Numero classi rilevate: {dataset_info['num_classes']}")
        
        return adapter, dataset_info  # ← CAMBIATO: ritorna anche le info
    
    @staticmethod
    def create_task_adapter(dataset_path, config, teacher_model, student_model, dataset_info=None):
        """
        MODIFICATO: Ora accetta dataset_info opzionale
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")
        
        # Se abbiamo le info del dataset, usale
        if dataset_info:
            task_type_str = dataset_info['task_type']
            num_classes = dataset_info['num_classes']
            print(f"[FACTORY] Usando info dataset: {task_type_str}, {num_classes} classi")
        else:
            # Fallback al metodo originale
            import pandas as pd
            df = pd.read_csv(dataset_path)
            task_info = TaskDetector.detect_task_type(df)
            task_type = task_info['task_type']
            task_type_str = task_type.value
            num_classes = 2  # Default
        
        # Aggiungi num_classes al config
        config['num_classes'] = num_classes
        
        print(f"[FACTORY] Creando task adapter per: {task_type_str}")
        
        # Converti stringa in enum se necessario
        
         # Crea l'adapter appropriato
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
            raise ValueError(f"Task type non supportato: {task_type.value}")
  
        
    @staticmethod
    def create_model_adapter(model_path):
        """
        Crea un adapter per il modello
        
        Args:
            model_path (str): Path al modello salvato
            
        Returns:
            BaseModelAdapter: Adapter per il modello
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        print(f"[FACTORY] Creando model adapter per: {model_path}")
        
        adapter = BaseModelAdapter(model_path)
        
        print(f"[FACTORY] Model adapter creato - Type: {type(adapter.model)}")
        return adapter

   
    @staticmethod
    def create_dataset_adapter_with_imagenet_mapping(dataset_path, imagenet_mapping_path, 
                                                   tokenizer_name=None, max_samples=None):
        """
        Metodo di convenienza per creare un dataset adapter con mapping ImageNet
        
        Args:
            dataset_path (str): Path al file CSV del dataset
            imagenet_mapping_path (str): Path al file di mapping ImageNet
            tokenizer_name (str): Nome del tokenizer (opzionale)
            max_samples (int): Numero massimo di campioni da caricare
            
        Returns:
            BaseDatasetAdapter: Adapter configurato per ImageNet
        """
        return AdapterFactory.create_dataset_adapter(
            dataset_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            imagenet_mapping_path=imagenet_mapping_path,
            task_type=TaskType.IMAGE_CLASSIFICATION
        )