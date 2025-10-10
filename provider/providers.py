from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Type, Optional, Tuple
import os
from dataclasses import dataclass
from enum import Enum


from dataset_adapter.modular_dataset_adapter import ModularDatasetAdapter  # Import del modulo refactorato
from config.config import DatasetConfig,ModelConfig

class ModularDatasetProvider:
    """Provider modulare per dataset"""
    
    def create_adapter(self, config: DatasetConfig) -> Tuple[Any, Dict[str, Any]]:
        if not os.path.exists(config.csv_path):
            raise FileNotFoundError(f"Dataset not found: {config.csv_path}")
        
        # Usa la nuova ModularDatasetAdapter
        from dataset_adapter.modular_dataset_adapter import ModularDatasetAdapter  # Import del modulo refactorato
        
        adapter = ModularDatasetAdapter(
            csv_path=config.csv_path,
            tokenizer_name=config.tokenizer_name,
            max_samples=config.max_samples,
            imagenet_mapping_path=config.imagenet_mapping_path
        )
        
        dataset_info = adapter.get_dataset_info()
        
        print(f"[DATASET_PROVIDER] Created adapter for {dataset_info['task_type']} "
              f"with {dataset_info['num_classes']} classes")
        
        return adapter, dataset_info

class StandardModelProvider:
    """Provider standard per modelli PyTorch"""
    
    def load_model(self, config: ModelConfig) -> Any:
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model not found: {config.model_path}")
        
        import torch
        
        # Carica modello
        model = torch.load(config.model_path, map_location='cpu')
        
        # Gestione device
        if config.device:
            device = torch.device(config.device)
            model = model.to(device)
        
        print(f"[MODEL_PROVIDER] Loaded model from {config.model_path}")
        return model