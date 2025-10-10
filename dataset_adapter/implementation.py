from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset_adapter.interfaces import ILabelManager,ITokenizerManager



class AutomaticLabelManager(ILabelManager):
    """Gestione automatica delle label"""
    
    def __init__(self, imagenet_mapping_path: Optional[str] = None):
        self.imagenet_mapping_path = imagenet_mapping_path
        self.label_to_idx = {}
        self.idx_to_label = {}
    
    def create_mapping(self, labels: pd.Series) -> Dict[str, int]:
        if self.imagenet_mapping_path:
            return self._load_imagenet_mapping()
        else:
            return self._create_automatic_mapping(labels)
    
    def _create_automatic_mapping(self, labels: pd.Series) -> Dict[str, int]:
        unique_labels = sorted(labels.astype(str).unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        return self.label_to_idx
    
    def apply_mapping(self, labels: pd.Series) -> pd.Series:
        return labels.astype(str).map(self.label_to_idx).fillna(0).astype(int)
    
    def get_num_classes(self) -> int:
        return len(self.label_to_idx)

class SmartTokenizerManager(ITokenizerManager):
    """Gestione intelligente dei tokenizer"""
    
    def __init__(self, tokenizer, supported_inputs: Optional[Set[str]] = None):
        self.tokenizer = tokenizer
        self.supported_inputs = supported_inputs
        self.capabilities = self.detect_capabilities() if tokenizer else {}
    
    def detect_capabilities(self) -> Dict[str, bool]:
        if not self.tokenizer:
            return {}
        
        test_text = ["Test sentence for capability detection."]
        capabilities = {}
        
        try:
            output = self.tokenizer(
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            
            for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                capabilities[key] = key in output
                
        except Exception as e:
            print(f"[TOKENIZER] Error detecting capabilities: {e}")
            capabilities = {'input_ids': True, 'attention_mask': True}
        
        return capabilities
    
    def filter_inputs(self, tokenized_output: Dict) -> Dict:
        if not self.supported_inputs:
            return self._conservative_filter(tokenized_output)
        
        filtered = {}
        for key, value in tokenized_output.items():
            if key in self.supported_inputs or key in ['input_ids', 'attention_mask']:
                filtered[key] = value
        
        return filtered
    
    def _conservative_filter(self, tokenized_output: Dict) -> Dict:
        safe_inputs = ['input_ids', 'attention_mask']
        return {k: v for k, v in tokenized_output.items() if k in safe_inputs}
