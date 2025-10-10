from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set
import pandas as pd
import torch
from torch.utils.data import DataLoader

# =================== CORE INTERFACES ===================

class IDatasetProcessor(ABC):
    """Interface per processori di dataset specifici"""
    
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> Any:
        pass
    
    @abstractmethod
    def create_dataloader(self, processed_data: Any) -> DataLoader:
        pass

class ILabelManager(ABC):
    """Interface per gestione label e mapping"""
    
    @abstractmethod
    def create_mapping(self, labels: pd.Series) -> Dict[str, int]:
        pass
    
    @abstractmethod
    def apply_mapping(self, labels: pd.Series) -> pd.Series:
        pass

class ITokenizerManager(ABC):
    """Interface per gestione tokenizer"""
    
    @abstractmethod
    def detect_capabilities(self) -> Dict[str, bool]:
        pass
    
    @abstractmethod
    def filter_inputs(self, tokenized_output: Dict) -> Dict:
        pass