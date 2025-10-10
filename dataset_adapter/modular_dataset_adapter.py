from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set
import pandas as pd
import torch
from torch.utils.data import DataLoader


from dataset_adapter.implementation import AutomaticLabelManager, SmartTokenizerManager
from dataset_adapter.interfaces import IDatasetProcessor
from dataset_adapter.processor import TextDatasetProcessor, ImageDatasetProcessor




class ModularDatasetAdapter:
    """Adapter modulare che compone i vari manager"""
    
    def __init__(self, csv_path: str, tokenizer_name: Optional[str] = None,
                 max_samples: Optional[int] = None, 
                 imagenet_mapping_path: Optional[str] = None):
        
        # Core data
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        if max_samples:
            self.df = self.df.head(max_samples)
        
        # Task detection
        self.mode = self._infer_mode()
        
        # Initialize managers
        self.label_manager = AutomaticLabelManager(imagenet_mapping_path)
        
        # Setup tokenizer manager se necessario
        self.tokenizer_manager = None
        if tokenizer_name and self.mode == "text":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer_manager = SmartTokenizerManager(tokenizer)
        
        # Setup processor specifico
        self.processor = self._create_processor()
        
        # Setup labels
        self._setup_labels()
    
    def _infer_mode(self) -> str:
        first_col = self.df.iloc[:, 0].astype(str)
        
        if (first_col.str.contains(r'\.(jpg|jpeg|png)$', case=False).any() or
            first_col.str.startswith("data:image/").any()):
            return "image"
        elif first_col.str.len().mean() > 30:
            return "text"
        else:
            return "tabular"
    
    def _create_processor(self) -> IDatasetProcessor:
        """Factory method per creare il processore appropriato"""
        if self.mode == "text":
            if not self.tokenizer_manager:
                raise ValueError("Tokenizer manager required for text mode")
            return TextDatasetProcessor(self.tokenizer_manager)
        elif self.mode == "image":
            return ImageDatasetProcessor()
        else:
            # Implementare TabularDatasetProcessor se necessario
            raise NotImplementedError("Tabular processor not implemented yet")
    
    def _setup_labels(self):
        label_column = self.df.iloc[:, 1]
        self.label_manager.create_mapping(label_column)
        self.df.iloc[:, 1] = self.label_manager.apply_mapping(label_column)
    
    def set_supported_inputs(self, supported_inputs: Set[str]):
        """Imposta input supportati per il tokenizer"""
        if self.tokenizer_manager:
            self.tokenizer_manager.supported_inputs = supported_inputs
    
    def get_dataloader(self) -> DataLoader:
        """Metodo principale per ottenere il dataloader"""
        processed_data = self.processor.process_data(self.df)
        return self.processor.create_dataloader(processed_data)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Informazioni sul dataset"""
        return {
            'num_classes': self.label_manager.get_num_classes(),
            'class_names': list(self.label_manager.label_to_idx.keys()),
            'label_mapping': self.label_manager.label_to_idx,
            'task_type': self.mode,
            'num_samples': len(self.df),
            'tokenizer_capabilities': (self.tokenizer_manager.capabilities 
                                     if self.tokenizer_manager else None)
        }

# =================== USAGE EXAMPLE ===================

def main():
    # Creazione adapter modulare
    adapter = ModularDatasetAdapter(
        csv_path="datasets/CIFAR10_tiny/train.csv",
        #tokenizer_name="bert-base-uncased"  # solo se text
    )
    
    # Configurazione input supportati (da task handler)
    adapter.set_supported_inputs({'input_ids', 'attention_mask'})
    
    # Ottenimento dataloader
    dataloader = adapter.get_dataloader()
    
    # Informazioni dataset
    info = adapter.get_dataset_info()
    print(f"Dataset info: {info}")



if __name__ == "__main__":
    main()